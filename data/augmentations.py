"""
Audio-specific augmentations for SED / segmentation tasks.
All operations are differentiable where possible and applied in waveform domain.
"""

import random
import torch
import torchaudio.transforms as T
import torchaudio.functional as F

class AudioAugmentations:
    """
    Composable audio augmentations pipeline.
    Designed for polyphonic SED: preserves event timing and frequency content as much as possible.
    """

    def __init__(
        self,
        prob: float = 0.6,
        sample_rate: int = 44100,
        time_stretch_range: tuple = (0.8, 1.25),
        pitch_shift_range: tuple = (-4, 4),
        noise_snr_range: tuple = (10, 30),
        mixup_alpha: float = 0.4,
        add_background_prob: float = 0.3,
        background_noise_dir: str = None,  # Optional path to background noises
    ):
        self.prob = prob
        self.sr = sample_rate

        # Time-domain augmentations
        self.time_stretch = T.TimeStretch(
            n_freq=1025,  # Default STFT
            hop_length=512,
            fixed_rate=None  # Will be randomized
        )

        self.pitch_shift = T.PitchShift(sample_rate, n_steps=0)

        # Noise addition
        self.noise_snr_range = noise_snr_range

        # Mixup (simple waveform-level)
        self.mixup_alpha = mixup_alpha

        # Background noise (if provided)
        self.background_noise_dir = background_noise_dir
        self.background_files = []
        if background_noise_dir:
            import os
            self.background_files = [
                os.path.join(background_noise_dir, f)
                for f in os.listdir(background_noise_dir)
                if f.lower().endswith(('.wav', '.flac', '.ogg'))
            ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations with probability self.prob.
        waveform: [C, T] or [T] (mono)
        Returns augmented waveform (same shape)
        """
        if random.random() > self.prob:
            return waveform

        # Ensure mono or stereo consistency
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 2 and waveform.shape[0] > 2:
            waveform = waveform[:1]  # Take first channel

        aug = waveform.clone()

        # 1. Time stretching (changes duration, but preserves pitch)
        if random.random() < 0.5:
            rate = random.uniform(*self.time_stretch_range)
            aug = F.time_stretch(
                aug,
                rate=rate,
                n_freq=1025,
                hop_length=512
            )
            # Trim/pad to original length
            orig_len = waveform.shape[-1]
            aug_len = aug.shape[-1]
            if aug_len > orig_len:
                aug = aug[..., :orig_len]
            elif aug_len < orig_len:
                aug = torch.nn.functional.pad(aug, (0, orig_len - aug_len))

        # 2. Pitch shifting (changes pitch, preserves duration)
        if random.random() < 0.4:
            n_steps = random.randint(*self.pitch_shift_range)
            aug = F.pitch_shift(aug, self.sr, n_steps)

        # 3. Add Gaussian noise (SNR-based)
        if random.random() < 0.6:
            snr_db = random.uniform(*self.noise_snr_range)
            noise = torch.randn_like(aug)
            noise_power = noise.pow(2).mean()
            signal_power = aug.pow(2).mean()
            scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            aug = aug + noise * scale

        # 4. Background noise mix (real-world noise)
        if self.background_files and random.random() < 0.35:
            bg_path = random.choice(self.background_files)
            try:
                bg, bg_sr = torchaudio.load(bg_path)
                if bg_sr != self.sr:
                    bg = F.resample(bg, bg_sr, self.sr)
                bg = bg.mean(0, keepdim=True)  # mono

                # Random crop to match length
                bg_len = bg.shape[-1]
                wav_len = aug.shape[-1]
                if bg_len > wav_len:
                    start = random.randint(0, bg_len - wav_len)
                    bg = bg[..., start:start + wav_len]
                else:
                    bg = F.pad(bg, (0, wav_len - bg_len))

                # Random volume
                bg_scale = random.uniform(0.1, 0.6)
                aug = aug + bg * bg_scale
            except Exception:
                pass  # silent fail if file broken

        # 5. Mixup (simple waveform mixup with another sample - requires batch)
        # Note: full mixup usually done in collate_fn, here only intra-sample simulation

        # Normalize to prevent clipping
        max_val = aug.abs().max() + 1e-8
        if max_val > 1.0:
            aug /= max_val * 1.05  # slight headroom

        return aug.squeeze(0) if waveform.dim() == 1 else aug


# Convenience function for dataset usage
def audio_augment(waveform: torch.Tensor, prob: float = 0.6, sample_rate: int = 44100) -> torch.Tensor:
    """
    Simple wrapper for quick usage in datasets.
    """
    augmenter = AudioAugmentations(prob=prob, sample_rate=sample_rate)
    return augmenter(waveform)
