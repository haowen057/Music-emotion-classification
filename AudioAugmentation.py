import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np

AUGMENT_CONFIG = {
    "noise_prob": 0,
    "noise_level": 0,
    "speed_prob": 0,
    "speed_factors": [1.0],
    "pitch_prob": 0,
    "pitch_steps": 0,
    "freq_mask_prob": 0.2,
    "time_mask_prob": 0.2,
    "max_freq_mask": 8,
    "max_time_mask": 12,
    "max_num_masks": 1,
    "global_prob": 0.8,
    "inner_prob": 0.5
}

class AudioAugmentation:
    
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
        augmented = signal.clone()
        return self._audio_augmentation(augmented)
    
    def apply_deterministic_augmentations(self, signal, index, seed=42):
        rng = np.random.RandomState(seed + index)
        augmented = signal.clone()
        
        if rng.rand() < self.noise_prob:
            augmented = self._add_deterministic_noise(augmented, rng)
        if rng.rand() < self.speed_prob:
            augmented = self._deterministic_speed_perturb(augmented, rng)
        if rng.rand() < self.pitch_prob:
            augmented = self._deterministic_pitch_shift(augmented, rng)
            
        return augmented
    
    def _audio_augmentation(self, signal):
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
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise
    
    def _add_deterministic_noise(self, signal, rng):
        noise = torch.from_numpy(rng.randn(*signal.shape).astype(np.float32)).to(self.device)
        noise = noise * self.noise_level
        return signal + noise
    
    def _speed_perturbation(self, audio):
        try:
            factor = random.choice(self.speed_factors)
            return self._apply_speed_perturb(audio, factor)
        except Exception as e:
            print(f"Speed perturbation failed: {e}")
            return audio
    
    def _deterministic_speed_perturb(self, audio, rng):
        try:
            factor = self.speed_factors[rng.randint(0, len(self.speed_factors))]
            return self._apply_speed_perturb(audio, factor)
        except Exception as e:
            print(f"Speed perturbation failed: {e}")
            return audio
    
    def _apply_speed_perturb(self, audio, factor):
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
        try:
            return self.pitch_shift(signal)
        except Exception as e:
            print(f"Pitch shift failed: {e}")
            return signal
    
    def _deterministic_pitch_shift(self, signal, rng):
        try:
            pitch_steps = rng.choice([-2, -1, 1, 2])
            pitch_shift = T.PitchShift(
                                        sample_rate=self.sample_rate, 
                                        n_steps=pitch_steps
                                        ).to(self.device)
            return pitch_shift(signal)
        except Exception as e:
            print(f"Pitch shift failed: {e}")
            return signal

class SpectrogramAugmentation:
    
    def __init__(self):
        self.freq_mask_prob = AUGMENT_CONFIG["freq_mask_prob"]
        self.time_mask_prob = AUGMENT_CONFIG["time_mask_prob"]
        self.max_freq_mask = AUGMENT_CONFIG["max_freq_mask"]
        self.max_time_mask = AUGMENT_CONFIG["max_time_mask"]
        self.max_num_masks = AUGMENT_CONFIG["max_num_masks"]
    
    def apply_augmentations(self, spectrogram):
        augmented = spectrogram.clone()
        
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

class CombinedAugmentation:
    
    def __init__(self, 
                 sample_rate=22050, 
                 device='cuda', 
                 deterministic=False, 
                 seed=42, 
                 audio_augment=True, 
                 spec_augment=True
                 ):
        self.audio_aug = AudioAugmentation(sample_rate, device)
        self.spec_aug = SpectrogramAugmentation()
        self.global_prob = AUGMENT_CONFIG["global_prob"]
        self.inner_prob = AUGMENT_CONFIG["inner_prob"]
        
        self.audio_augment = audio_augment
        self.spec_augment = spec_augment
        
        self.deterministic = deterministic
        self.seed = seed
        
        print(f"Initialized augmenter - Audio: {audio_augment}, Spectrogram: {spec_augment}, Deterministic: {deterministic}")
    
    def apply_audio_only(self, signal, index=None):
        if not self.audio_augment or random.random() > self.inner_prob:
            return signal
        
        if self.deterministic and index is not None:
            return self.audio_aug.apply_deterministic_augmentations(signal, index, self.seed)
        else:
            return self.audio_aug.apply_augmentations(signal)
    
    def apply_spec_only(self, spectrogram):
        if not self.spec_augment or random.random() > self.inner_prob:
            return spectrogram
        return self.spec_aug.apply_augmentations(spectrogram)
    
    def apply_both(self, signal, spectrogram, index=None):
        if self.audio_augment and random.random() < self.inner_prob:
            if self.deterministic and index is not None:
                signal = self.audio_aug.apply_deterministic_augmentations(signal, index, self.seed)
            else:
                signal = self.audio_aug.apply_augmentations(signal)
        if self.spec_augment and random.random() < self.inner_prob:
            spectrogram = self.spec_aug.apply_augmentations(spectrogram)
        return signal, spectrogram
    
    def __call__(self, data, index=None):
        if random.random() > self.global_prob:
            return data
            
        if isinstance(data, tuple):
            return self.apply_both(data[0], data[1], index)
        else:
            if len(data.shape) == 3 and self.spec_augment:
                return self.apply_spec_only(data)
            elif len(data.shape) != 3 and self.audio_augment:
                return self.apply_audio_only(data, index)
            else:
                return data

def create_augmentation_transforms(sample_rate=22050, device='cuda', deterministic=False, seed=42, audio_augment=True, spec_augment=True):
    return CombinedAugmentation(sample_rate, device, deterministic, seed, audio_augment, spec_augment)

if __name__ == "__main__":
    print("AudioAugmentation.py loaded successfully - supports deterministic and random modes")
    print("Current augmentation configuration:")
    for k, v in AUGMENT_CONFIG.items():
        print(f"   - {k}: {v}")
    
    print("\nUsage examples:")
    print("   # Random augmentation (for training)")
    print("   augmentor = CombinedAugmentation(deterministic=False)")
    print("   augmented_signal = augmentor(signal)")
    print("")
    print("   # Deterministic augmentation (for pre-computation)") 
    print("   augmentor = CombinedAugmentation(deterministic=True, seed=42)")
    print("   augmented_signal = augmentor(signal, index=sample_index)")
