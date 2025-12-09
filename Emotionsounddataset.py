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


logger = logging.getLogger(__name__)

class EmotionSoundDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation=None,
                 target_sample_rate=22050,
                 num_samples=220500,
                 device='cpu',
                 use_log_mel=True,
                 top_n_classes=10,
                 augment=False,
                 audio_augment=True,  
                 spec_augment=True,
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

        self.combined_augmentation = None

        if self.augment:
            self.combined_augmentation = CombinedAugmentation(
                sample_rate=target_sample_rate, 
                device=device,
                deterministic=deterministic_augment,
                seed=augment_seed,
                audio_augment=self.audio_augment,
                spec_augment=self.spec_augment
            )
            mode = "deterministic" if deterministic_augment else "random"
            logger.info(f"Initialized augmentation: audio={audio_augment}, spectrogram={spec_augment}, mode={mode}, seed={augment_seed}")

        self.top_labels = [
            'melodic', 'energetic', 'dark', 'film', 'relaxing',
            'dream', 'ambiental', 'love', 'soundscape', 'emotional'
        ][:top_n_classes]

        self.annotations = pd.read_csv(annotations_file, sep='\t')

        self.filtered_annotations = self.annotations[
            self.annotations['TAG'].apply(
                lambda x: any(tag.replace('mood/theme---','').strip() in self.top_labels for tag in x.split(','))
            )
        ].reset_index(drop=True)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)

        print(f"Total samples: {len(self.filtered_annotations)}")
        for label in self.top_labels:
            count = len(self.filtered_annotations[self.filtered_annotations['TAG'].str.contains(label)])
            print(f"mood/theme---{label}: found {count} wav files")

    def __len__(self):
        return len(self.filtered_annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        if label == -1:
            raise ValueError(f"Sample {index} not in top {len(self.top_labels)} labels")

        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._preprocess_audio(signal, sr)

        if self.augment and self.combined_augmentation:
            if self.audio_augment:
                try:
                    signal = self.combined_augmentation(signal, index=index)
                except Exception as e:
                    logger.warning(f"Audio augmentation failed, using original signal: {e}")
            
            mel_spectrogram = self.mel_transform(signal)
            features = torch.log(mel_spectrogram + 1e-9)
        else:
            mel_spectrogram = self.mel_transform(signal)
            features = torch.log(mel_spectrogram + 1e-9)

        if self.transformation:
            features = self.transformation(features)

        return features, label

    def _preprocess_audio(self, signal, sr):
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

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
            raise FileNotFoundError(f"File not found: {full_path}")
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
    
    def get_audio_and_mel(self, index, apply_augmentation=True):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        if label == -1:
            raise ValueError(f"Sample {index} not in top {len(self.top_labels)} labels")
    
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
    
        original_signal = self._preprocess_audio(signal, sr)
        
        if apply_augmentation and self.augment and self.audio_augment:
            try:
                augmented_signal = self.combined_augmentation(original_signal.clone(), index=index)
            except Exception as e:
                logger.warning(f"Debug mode audio augmentation failed: {e}")
                augmented_signal = original_signal
        else:
            augmented_signal = original_signal
    
        mel_spectrogram = self.mel_transform(augmented_signal)
        features = torch.log(mel_spectrogram + 1e-9)
    
        if apply_augmentation and self.augment and self.spec_augment:
            try:
                features = self.combined_augmentation(features)
            except Exception as e:
                logger.warning(f"Debug mode spectrogram augmentation failed: {e}")
    
        return original_signal, augmented_signal, features, label
    
    def set_augmentation_mode(self, deterministic=True, seed=42):
        if self.combined_augmentation:
            self.combined_augmentation.toggle_deterministic(deterministic, seed)
            self.deterministic_augment = deterministic
            self.augment_seed = seed
            mode = "deterministic" if deterministic else "random"
            logger.info(f"Switched augmentation mode: {mode}, seed: {seed}")
    
    def get_dataset_info(self):
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
