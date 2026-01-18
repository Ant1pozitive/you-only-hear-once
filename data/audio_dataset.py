import torch
from torch.utils.data import Dataset
import torchaudio
import os
from typing import Tuple

class AudioSEDDataset(Dataset):
    """Example for ESC-50; adapt for others."""
    def __init__(self, root: str, augment_prob: float = 0.0):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith('.wav')]
        self.augment_prob = augment_prob
        # Labels: Assume annotation files with events (start, end, class, freq_range)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        path = os.path.join(self.root, self.files[idx])
        waveform, sr = torchaudio.load(path)
        waveform = audio_augment(waveform, self.augment_prob)
        # Labels: dict with 'boxes': [N,4] (t_start, f_start, t_end, f_end), 'classes': [N], 'masks': [N, H, W] if seg
        labels = {"boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]), "classes": torch.tensor([1]), "masks": None}
        return waveform.squeeze(0), labels
