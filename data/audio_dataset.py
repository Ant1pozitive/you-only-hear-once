import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import pandas as pd
import random
from typing import Dict, List, Tuple
from .augmentations import AudioAugmentations

class AudioSEDDataset(Dataset):
    def __init__(self, root: str, meta_csv: str = 'esc50.csv',
                 augment_prob: float = 0.5, sr: int = 44100,
                 max_dur_sec: float = 5.0):
        self.root = os.path.join(root, 'audio')
        self.meta = pd.read_csv(os.path.join(root, meta_csv))
        self.files = self.meta['filename'].tolist()
        self.labels = self.meta['target'].tolist() if 'target' in self.meta.columns else None
        self.sr = sr
        self.max_dur_sec = max_dur_sec
        self.max_length = int(sr * max_dur_sec)
        self.augmenter = AudioAugmentations(prob=augment_prob, sample_rate=sr)

    def __len__(self) -> int:
        return len(self.files)

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length > self.max_length:
            start = random.randint(0, length - self.max_length)
            return waveform[:, start:start + self.max_length]
        elif length < self.max_length:
            return torch.nn.functional.pad(waveform, (0, self.max_length - length))
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        path = os.path.join(self.root, self.files[idx])
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        except Exception:
            waveform = torch.zeros((1, self.max_length))

        waveform = self._pad_or_trim(waveform)
        if self.augmenter.prob > 0:
            waveform = self.augmenter(waveform)

        # normalized boxes [0,1] x [0,1]
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        cls = torch.tensor([self.labels[idx]], dtype=torch.long) if self.labels is not None else torch.empty((0,), dtype=torch.long)

        labels = {"boxes": boxes, "labels": cls}
        return waveform.squeeze(0), labels


def yoho_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    audio_list = [item[0] for item in batch]
    boxes_list = [item[1]['boxes'] for item in batch]
    labels_list = [item[1]['labels'] for item in batch]

    audio_batch = torch.stack(audio_list, dim=0)
    targets = {"boxes": boxes_list, "labels": labels_list}
    return audio_batch, targets


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=yoho_collate_fn,
        pin_memory=True
    )
