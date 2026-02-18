"""
Audio-specific augmentations for SED tasks.
Optimized to strictly preserve temporal alignment (no time-stretching) 
so bounding box targets remain perfectly accurate.
"""

import random
import torch
import torchaudio
import torchaudio.functional as F
import os

class AudioAugmentations:
    """
    Composable, box-safe audio augmentations pipeline.
    """
    def __init__(
        self,
        prob: float = 0.5,
        sample_rate: int = 44100,
        pitch_shift_range: tuple = (-3, 3),
        noise_snr_range: tuple = (10, 25),
        background_noise_dir: str = None,
    ):
        self.prob = prob
        self.sr = sample_rate
        self.pitch_shift_range = pitch_shift_range
        self.noise_snr_range = noise_snr_range

        self.background_files = []
        if background_noise_dir and os.path.exists(background_noise_dir):
            self.background_files = [
                os.path.join(background_noise_dir, f)
                for f in os.listdir(background_noise_dir)
                if f.lower().endswith(('.wav', '.flac', '.ogg'))
            ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [1, T] tensor.
        """
        if random.random() > self.prob:
            return waveform

        # Keep a reference to original length to ensure no shape changes
        orig_len = waveform.shape[-1]
        aug = waveform.clone()

        # 1. Pitch shifting (Changes frequency pitch, preserves duration)
        # Safe for boxes because our freq bounds are generally [0.0, 1.0] for broad sounds
        if random.random() < 0.3:
            n_steps = random.randint(*self.pitch_shift_range)
            # torchaudio pitch_shift can be slow on CPU, use with caution
            aug = F.pitch_shift(aug, self.sr, n_steps)

        # 2. Add Gaussian noise (Simulates sensor noise)
        if random.random() < 0.5:
            snr_db = random.uniform(*self.noise_snr_range)
            noise = torch.randn_like(aug)
            
            # Avoid divide by zero
            signal_power = aug.pow(2).mean().clamp(min=1e-7)
            noise_power = noise.pow(2).mean().clamp(min=1e-7)
            
            scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            aug = aug + noise * scale

        # 3. Background noise mix (Real-world environmental noise)
        if self.background_files and random.random() < 0.3:
            bg_path = random.choice(self.background_files)
            try:
                bg, bg_sr = torchaudio.load(bg_path)
                bg = bg.mean(0, keepdim=True)  # force mono
                
                if bg_sr != self.sr:
                    bg = F.resample(bg, bg_sr, self.sr)

                bg_len = bg.shape[-1]
                if bg_len > orig_len:
                    start = random.randint(0, bg_len - orig_len)
                    bg = bg[:, start:start + orig_len]
                else:
                    # Repeat background if it's too short
                    repeats = (orig_len // bg_len) + 1
                    bg = bg.repeat(1, repeats)[:, :orig_len]

                bg_scale = random.uniform(0.1, 0.4)
                aug = aug + bg * bg_scale
            except Exception:
                pass # Silently skip if background file is corrupted

        # 4. Soft clipping / Normalization to prevent audio distortion
        max_val = aug.abs().max()
        if max_val > 0.95:
            aug = aug / (max_val * 1.05)

        # Final safety check to guarantee tensor shape hasn't changed
        assert aug.shape[-1] == orig_len, "Augmentation modified temporal length!"
        
        return aug

def audio_augment(waveform: torch.Tensor, prob: float = 0.5, sample_rate: int = 44100) -> torch.Tensor:
    augmenter = AudioAugmentations(prob=prob, sample_rate=sample_rate)
    return augmenter(waveform)
