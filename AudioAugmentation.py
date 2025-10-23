import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np

# ==========================================================
# 🔧 全局增强参数配置（统一入口）
# ==========================================================

AUGMENT_CONFIG = {
    # ---- Audio ----
    "noise_prob": 0,            # 添加高斯噪声概率
    "noise_level": 0,         # 高斯噪声强度
    "speed_prob": 0,            # 速度扰动概率
    "speed_factors": [1.0],  # 速度扰动系数
    "pitch_prob": 0,            # 音高偏移概率
    "pitch_steps": 0,             # 音高变化步长（半音）

    # ---- Spectrogram ----
    "freq_mask_prob": 0.2,        # 频率掩码概率
    "time_mask_prob": 0.2,        # 时间掩码概率
    "max_freq_mask": 8,          # 最大频率掩码宽度
    "max_time_mask": 12,          # 最大时间掩码宽度
    "max_num_masks": 1,           # 每次掩码数量

    # ---- Combined ----
    "global_prob": 0.8,           # 样本增强概率
    "inner_prob": 0.5             # 内层增强概率（每个模块）
    
}

# ==========================================================
# 🎧 音频增强
# ==========================================================

class AudioAugmentation:
    """音频数据增强类（支持确定性和随机模式）"""
    
    def __init__(self, sample_rate=22050, device='cuda'):
        self.sample_rate = sample_rate
        self.device = device
        
        self.noise_prob = AUGMENT_CONFIG["noise_prob"]
        self.noise_level = AUGMENT_CONFIG["noise_level"]
        self.speed_prob = AUGMENT_CONFIG["speed_prob"]
        self.speed_factors = AUGMENT_CONFIG["speed_factors"]
        self.pitch_prob = AUGMENT_CONFIG["pitch_prob"]
        
        self.pitch_shift = T.PitchShift(sample_rate=sample_rate, n_steps=AUGMENT_CONFIG["pitch_steps"]).to(device)
        
    def apply_augmentations(self, signal):
        """随机音频增强"""
        augmented = signal.clone()
        return self._audio_augmentation(augmented)
    
    def apply_deterministic_augmentations(self, signal, index, seed=42):
        """确定性音频增强"""
        # 使用固定种子+索引创建确定性随机状态
        rng = np.random.RandomState(seed + index)
        augmented = signal.clone()
        
        # 确定性增强决策
        if rng.rand() < self.noise_prob:
            augmented = self._add_deterministic_noise(augmented, rng)
        if rng.rand() < self.speed_prob:
            augmented = self._deterministic_speed_perturb(augmented, rng)
        if rng.rand() < self.pitch_prob:
            augmented = self._deterministic_pitch_shift(augmented, rng)
            
        return augmented
    
    def _audio_augmentation(self, signal):
        """音频信号增强（随机模式）"""
        augmented = signal.clone()
        aug_methods = []
        
        if random.random() < self.noise_prob:
            aug_methods.append('noise')
        if random.random() < self.speed_prob:
            aug_methods.append('speed_perturb')
        if random.random() < self.pitch_prob:
            aug_methods.append('pitch_shift')
        
        random.shuffle(aug_methods)
        
        for method in aug_methods:
            if method == 'noise':
                augmented = self._add_gaussian_noise(augmented)
            elif method == 'speed_perturb':
                augmented = self._speed_perturbation(augmented)
            elif method == 'pitch_shift':
                augmented = self._pitch_shift(augmented)
        
        return augmented
    
    def _add_gaussian_noise(self, signal):
        """添加高斯噪声（随机模式）"""
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise
    
    def _add_deterministic_noise(self, signal, rng):
        """添加高斯噪声（确定性模式）"""
        noise = torch.from_numpy(rng.randn(*signal.shape).astype(np.float32)).to(self.device)
        noise = noise * self.noise_level
        return signal + noise
    
    def _speed_perturbation(self, audio):
        """速度扰动（随机模式）"""
        try:
            factor = random.choice(self.speed_factors)
            return self._apply_speed_perturb(audio, factor)
        except Exception as e:
            print(f"速度扰动失败: {e}")
            return audio
    
    def _deterministic_speed_perturb(self, audio, rng):
        """速度扰动（确定性模式）"""
        try:
            factor = self.speed_factors[rng.randint(0, len(self.speed_factors))]
            return self._apply_speed_perturb(audio, factor)
        except Exception as e:
            print(f"速度扰动失败: {e}")
            return audio
    
    def _apply_speed_perturb(self, audio, factor):
        """应用速度扰动（通用实现）"""
        new_length = int(audio.shape[1] / factor)
        stretched = torch.nn.functional.interpolate(
            audio.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        if stretched.shape[1] > audio.shape[1]:
            start = random.randint(0, stretched.shape[1] - audio.shape[1])
            stretched = stretched[:, start:start + audio.shape[1]]
        else:
            pad_length = audio.shape[1] - stretched.shape[1]
            stretched = torch.nn.functional.pad(stretched, (0, pad_length))
        
        return stretched
    
    def _pitch_shift(self, signal):
        """音高偏移（随机模式）"""
        try:
            return self.pitch_shift(signal)
        except Exception as e:
            print(f"音高偏移失败: {e}")
            return signal
    
    def _deterministic_pitch_shift(self, signal, rng):
        """音高偏移（确定性模式）"""
        try:
            # 使用固定的音高偏移步长选项
            pitch_steps = rng.choice([-2, -1, 1, 2])
            pitch_shift = T.PitchShift(
                                        sample_rate=self.sample_rate, 
                                        n_steps=pitch_steps
                                        ).to(self.device)
            return pitch_shift(signal)
        except Exception as e:
            print(f"音高偏移失败: {e}")
            return signal


# ==========================================================
# 🎼 频谱增强
# ==========================================================

class SpectrogramAugmentation:
    """频谱图数据增强类"""
    
    def __init__(self):
        self.freq_mask_prob = AUGMENT_CONFIG["freq_mask_prob"]
        self.time_mask_prob = AUGMENT_CONFIG["time_mask_prob"]
        self.max_freq_mask = AUGMENT_CONFIG["max_freq_mask"]
        self.max_time_mask = AUGMENT_CONFIG["max_time_mask"]
        self.max_num_masks = AUGMENT_CONFIG["max_num_masks"]
    
    def apply_augmentations(self, spectrogram):
        augmented = spectrogram.clone()
        
        # 修改掩码下限
        num_freq_masks = random.randint(0, self.max_num_masks)
        num_time_masks = random.randint(0, self.max_num_masks)
        for _ in range(num_freq_masks):
            if random.random() < self.freq_mask_prob:
                augmented = self._frequency_masking(augmented)
        for _ in range(num_time_masks):
            if random.random() < self.time_mask_prob:
                augmented = self._time_masking(augmented)
        return augmented
    
    def _frequency_masking(self, spectrogram):
        n_mels = spectrogram.shape[1]
        max_mask = min(self.max_freq_mask, n_mels // 3)
        f_mask_size = random.randint(1, max_mask)
        f_start = random.randint(0, n_mels - f_mask_size)
        masked = spectrogram.clone()
        masked[:, f_start:f_start+f_mask_size, :] = 0
        return masked
    
    def _time_masking(self, spectrogram):
        time_steps = spectrogram.shape[2]
        max_mask = min(self.max_time_mask, time_steps // 4)
        t_mask_size = random.randint(1, max_mask)
        t_start = random.randint(0, time_steps - t_mask_size)
        masked = spectrogram.clone()
        masked[:, :, t_start:t_start+t_mask_size] = 0
        return masked


# ==========================================================
# 🔀 组合增强（统一调度）
# ==========================================================

class CombinedAugmentation:
    """组合音频和频谱图增强（支持确定性和随机模式）"""
    
    def __init__(self, 
                 sample_rate=22050, 
                 device='cuda', 
                 deterministic=False, 
                 seed=42, 
                 audio_augment=True, 
                 spec_augment=True
                 ):  # 🎯 新增控制参数
        self.audio_aug = AudioAugmentation(sample_rate, device)
        self.spec_aug = SpectrogramAugmentation()
        self.global_prob = AUGMENT_CONFIG["global_prob"]
        self.inner_prob = AUGMENT_CONFIG["inner_prob"]
        
        # 🎯 新增：增强控制参数
        self.audio_augment = audio_augment
        self.spec_augment = spec_augment
        
        # 🎯 新增：确定性模式配置
        self.deterministic = deterministic
        self.seed = seed
        
        print(f"🎯 初始化增强器 - 音频增强: {audio_augment}, 频谱增强: {spec_augment}, 确定性: {deterministic}")
    
    def apply_audio_only(self, signal, index=None):
        """应用音频增强"""
        if not self.audio_augment or random.random() > self.inner_prob:
            return signal
        
        if self.deterministic and index is not None:
            return self.audio_aug.apply_deterministic_augmentations(signal, index, self.seed)
        else:
            return self.audio_aug.apply_augmentations(signal)
    
    def apply_spec_only(self, spectrogram):
        """应用频谱增强（目前只支持随机模式）"""
        if not self.spec_augment or random.random() > self.inner_prob:
            return spectrogram
        return self.spec_aug.apply_augmentations(spectrogram)
    
    def apply_both(self, signal, spectrogram, index=None):
        """同时应用音频和频谱增强"""
        if self.audio_augment and random.random() < self.inner_prob:
            if self.deterministic and index is not None:
                signal = self.audio_aug.apply_deterministic_augmentations(signal, index, self.seed)
            else:
                signal = self.audio_aug.apply_augmentations(signal)
        if self.spec_augment and random.random() < self.inner_prob:
            spectrogram = self.spec_aug.apply_augmentations(spectrogram)
        return signal, spectrogram
    
    def __call__(self, data, index=None):
        """
        增强调用接口
        Args:
            data: 输入数据（音频信号或频谱图）
            index: 样本索引（用于确定性模式）
        """
        # 全局增强概率检查
        if random.random() > self.global_prob:
            return data
            
        if isinstance(data, tuple):
            return self.apply_both(data[0], data[1], index)
        else:
            if len(data.shape) == 3 and self.spec_augment:  # 频谱图
                return self.apply_spec_only(data)
            elif len(data.shape) != 3 and self.audio_augment:  # 音频信号
                return self.apply_audio_only(data, index)
            else:
                return data


# ==========================================================
# 🧩 工厂函数
# ==========================================================

def create_augmentation_transforms(sample_rate=22050, device='cuda', deterministic=False, seed=42, audio_augment=True, spec_augment=True):
    """创建增强转换器"""
    return CombinedAugmentation(sample_rate, device, deterministic, seed, audio_augment, spec_augment)

if __name__ == "__main__":
    # 测试代码
    print("✅ AudioAugmentation.py 加载成功 - 支持确定性和随机模式")
    print("📊 当前增强配置:")
    for k, v in AUGMENT_CONFIG.items():
        print(f"   - {k}: {v}")
    
    print("\n🎯 使用示例:")
    print("   # 随机增强（训练时）")
    print("   augmentor = CombinedAugmentation(deterministic=False)")
    print("   augmented_signal = augmentor(signal)")
    print("")
    print("   # 确定性增强（预计算时）") 
    print("   augmentor = CombinedAugmentation(deterministic=True, seed=42)")
    print("   augmented_signal = augmentor(signal, index=sample_index)")