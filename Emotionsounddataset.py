import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from AudioAugmentation import CombinedAugmentation 
import logging
import numpy as np

# 设置日志
logger = logging.getLogger(__name__)

class EmotionSoundDataset(Dataset):
    def __init__(self,
                 annotations_file,    # 文件信息路径
                 audio_dir,           # 音频根目录
                 transformation=None, # 数据变换
                 target_sample_rate=22050,
                 num_samples=220500,
                 device='cpu',
                 use_log_mel=True,
                 top_n_classes=10,
                 augment=False,
                 # 🎯 新增：细粒度增强控制
                 audio_augment=True,  
                 spec_augment=True,
                 # 🎯 新增：确定性增强控制
                 deterministic_augment=True,
                 augment_seed=42
                 ):
        
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.use_log_mel = use_log_mel
        self.audio_dir = audio_dir
        self.augment = augment
        self.audio_augment = audio_augment
        self.spec_augment = spec_augment
        self.deterministic_augment = deterministic_augment
        self.augment_seed = augment_seed

        # 🎯 优化：更清晰的增强初始化
        self.combined_augmentation = None

        if self.augment:
            self.combined_augmentation = CombinedAugmentation(
                sample_rate=target_sample_rate, 
                device=device,
                deterministic=deterministic_augment,  # 🎯 使用确定性模式
                seed=augment_seed,
                audio_augment=self.audio_augment,    # 🎯 新增：传递音频增强控制
                spec_augment=self.spec_augment       # 🎯 新增：传递频谱增强控制
            )
            mode = "确定性" if deterministic_augment else "随机"
            logger.info(f"✅ 初始化增强: 音频增强={audio_augment}, 频谱增强={spec_augment}, 模式={mode}, 种子={augment_seed}")

        # 前 10 类标签
        self.top_labels = [
            'melodic', 'energetic', 'dark', 'film', 'relaxing',
            'dream', 'ambiental', 'love', 'soundscape', 'emotional'
        ][:top_n_classes]

        # 读取文件信息
        self.annotations = pd.read_csv(annotations_file, sep='\t')

        # 只保留前 n 类的样本
        self.filtered_annotations = self.annotations[
            self.annotations['TAG'].apply(
                lambda x: any(tag.replace('mood/theme---','').strip() in self.top_labels for tag in x.split(','))
            )
        ].reset_index(drop=True)

        # mel 变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)

        print(f"🎯 总样本数: {len(self.filtered_annotations)}")
        for label in self.top_labels:
            count = len(self.filtered_annotations[self.filtered_annotations['TAG'].str.contains(label)])
            print(f"✅ mood/theme---{label}: 找到 {count} 个 wav 文件")

    def __len__(self):
        return len(self.filtered_annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        if label == -1:
            raise ValueError(f"Sample {index} not in top {len(self.top_labels)} labels")

        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        # 🎯 优化：统一的预处理流程
        signal = self._preprocess_audio(signal, sr)

        # 🎯 优化：更清晰的增强逻辑
        if self.augment and self.combined_augmentation:
            # 应用音频增强
            if self.audio_augment:
                try:
                    # 🎯 修改：直接使用新的确定性增强接口
                    signal = self.combined_augmentation(signal, index=index)
                except Exception as e:
                    logger.warning(f"音频增强失败，使用原始信号: {e}")
            
            # 转换为梅尔频谱图
            mel_spectrogram = self.mel_transform(signal)
            features = torch.log(mel_spectrogram + 1e-9)
            
            # 应用频谱增强
            # if self.spec_augment:
            #     try:
            #         # 🎯 频谱增强保持随机性（在训练时应用）
            #         features = self.combined_augmentation(features)
            #     except Exception as e:
            #         logger.warning(f"频谱增强失败，使用原始频谱: {e}")
        else:
            # 无增强路径
            mel_spectrogram = self.mel_transform(signal)
            features = torch.log(mel_spectrogram + 1e-9)

        # 原有的变换
        if self.transformation:
            features = self.transformation(features)

        return features, label

    # 🎯 新增：统一的预处理方法
    def _preprocess_audio(self, signal, sr):
        """统一的音频预处理流程"""
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

    # -------------------- 辅助函数 --------------------
    def _get_audio_sample_label(self, index):
        tags_str = self.filtered_annotations.iloc[index]['TAG']
        tags = tags_str.split(',')
        for tag in tags:
            clean_tag = tag.replace('mood/theme---', '').strip()
            if clean_tag in self.top_labels:
                return self.top_labels.index(clean_tag)
        return -1

    def _get_audio_sample_path(self, index):
        tags_str = self.filtered_annotations.iloc[index]['TAG']
        tags = tags_str.split(',')
        chosen_tag = None
        for tag in tags:
            clean_tag = tag.replace('mood/theme---', '').strip()
            if clean_tag in self.top_labels:
                chosen_tag = clean_tag
                break
        if chosen_tag is None:
            raise ValueError(f"Sample {index} has no valid top label")
        filename = self.filtered_annotations.iloc[index]['PATH']
        full_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"找不到文件: {full_path}")
        return full_path
    
    def _cut_if_necessary(self, signal):
        return signal[:, :self.num_samples] if signal.shape[1] > self.num_samples else signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            pad = self.num_samples - signal.shape[1]
            signal = nn.functional.pad(signal, (0, pad))
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    # 🎯 优化：增强的调试方法
    def get_audio_and_mel(self, index, apply_augmentation=True):
        """分别获取音频和梅尔频谱图，用于调试和可视化"""
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        if label == -1:
            raise ValueError(f"Sample {index} not in top {len(self.top_labels)} labels")
    
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
    
        # 预处理
        original_signal = self._preprocess_audio(signal, sr)
        
        # 应用音频增强
        if apply_augmentation and self.augment and self.audio_augment:
            try:
                # 🎯 修改：使用新的确定性增强接口
                augmented_signal = self.combined_augmentation(original_signal.clone(), index=index)
            except Exception as e:
                logger.warning(f"调试模式音频增强失败: {e}")
                augmented_signal = original_signal
        else:
            augmented_signal = original_signal
    
        # 转换为梅尔频谱图
        mel_spectrogram = self.mel_transform(augmented_signal)
        features = torch.log(mel_spectrogram + 1e-9)
    
        # 应用频谱图增强
        if apply_augmentation and self.augment and self.spec_augment:
            try:
                features = self.combined_augmentation(features)
            except Exception as e:
                logger.warning(f"调试模式频谱增强失败: {e}")
    
        return original_signal, augmented_signal, features, label
    
    # 🎯 新增：切换增强模式的方法
    def set_augmentation_mode(self, deterministic=True, seed=42):
        """切换增强模式（确定性/随机）"""
        if self.combined_augmentation:
            self.combined_augmentation.toggle_deterministic(deterministic, seed)
            self.deterministic_augment = deterministic
            self.augment_seed = seed
            mode = "确定性" if deterministic else "随机"
            logger.info(f"🎯 切换增强模式: {mode}, 种子: {seed}")
    

    
    # 🎯 新增：获取数据集信息
    def get_dataset_info(self):
        """返回数据集统计信息"""
        info = {
            'total_samples': len(self.filtered_annotations),
            'sample_rate': self.target_sample_rate,
            'num_samples': self.num_samples,
            'augment_enabled': self.augment,
            'audio_augment': self.audio_augment,
            'spec_augment': self.spec_augment,
            'deterministic_augment': self.deterministic_augment,
            'augment_seed': self.augment_seed,
            'labels_distribution': {}
        }
        
        for label in self.top_labels:
            count = len(self.filtered_annotations[self.filtered_annotations['TAG'].str.contains(label)])
            info['labels_distribution'][label] = count
            
        return info