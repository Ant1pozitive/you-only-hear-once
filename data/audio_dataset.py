import torch
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
from .augmentations import audio_augment

class AudioSEDDataset(Dataset):
    """ESC-50 dataset with pseudo-SED labels (whole clip as event)"""
    def __init__(self, root, meta_csv='esc50.csv', augment_prob=0.5):
        self.root = os.path.join(root, 'audio')
        self.meta = pd.read_csv(os.path.join(root, meta_csv))
        self.files = self.meta['filename'].tolist()
        self.labels = self.meta['target'].tolist()  # Class indices
        self.augment_prob = augment_prob
        self.sr = 44100  # ESC-50 sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        waveform, sr = torchaudio.load(path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = audio_augment(waveform, self.augment_prob)
        
        # Pseudo-labels: whole clip [t_start=0, f_start=0, t_end=1.0 (norm), f_end=1.0], class
        duration = waveform.shape[-1] / self.sr
        boxes = torch.tensor([[0.0, 0.0, duration, 8000.0]])  # t_start, f_start, t_end, f_end (Hz)
        cls = torch.tensor([self.labels[idx]])
        masks = torch.ones(1, 128, waveform.shape[-1] // 512)  # Pseudo-mask full spec
        
        labels = {"boxes": boxes, "cls": cls, "masks": masks}
        return waveform.squeeze(0), labels
