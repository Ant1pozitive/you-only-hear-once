import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import pandas as pd
import random
from typing import Dict, List, Tuple
from .augmentations import AudioAugmentations

class AudioSEDDataset(Dataset):
    """
    Base Dataset for Sound Event Detection.
    Handles variable length audio and formats boxes for YOHO.
    """
    def __init__(self, root: str, meta_csv: str = 'esc50.csv', 
                 augment_prob: float = 0.5, sr: int = 44100, 
                 max_duration_sec: float = 5.0):
        self.root = os.path.join(root, 'audio')
        self.meta = pd.read_csv(os.path.join(root, meta_csv))
        self.files = self.meta['filename'].tolist()
        
        # Handle 'target' column for ESC50 or 'event_label' for DESED
        if 'target' in self.meta.columns:
            self.labels = self.meta['target'].tolist()
            self.is_pseudo_sed = True
        else:
            self.labels = None  # Complex parsing handled in specific subclasses
            self.is_pseudo_sed = False
            
        self.sr = sr
        self.max_length = int(sr * max_duration_sec)
        self.augmenter = AudioAugmentations(prob=augment_prob, sample_rate=sr)

    def __len__(self) -> int:
        return len(self.files)

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure audio fits within max_duration without breaking batches"""
        length = waveform.shape[-1]
        if length > self.max_length:
            # Random crop if too long
            start = random.randint(0, length - self.max_length)
            return waveform[:, start:start + self.max_length]
        elif length < self.max_length:
            # Pad with zeros if too short
            return torch.nn.functional.pad(waveform, (0, self.max_length - length))
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        path = os.path.join(self.root, self.files[idx])
        
        try:
            waveform, sr = torchaudio.load(path)
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        except Exception as e:
            # Fallback for broken files
            print(f"Warning: Failed to load {path}. Using silence.")
            waveform = torch.zeros((1, self.max_length))
            
        # Apply temporal crop/pad before augmentations
        waveform = self._pad_or_trim(waveform)
        
        # Apply safe augmentations (noise, pitch)
        if self.augmenter.prob > 0:
            waveform = self.augmenter(waveform)
            
        # Format labels
        if self.is_pseudo_sed:
            # Pseudo-SED for classification datasets (ESC-50)
            # Box format: [time_start(sec), freq_start(norm), time_end(sec), freq_end(norm)]
            duration = waveform.shape[-1] / self.sr
            boxes = torch.tensor([[0.0, 0.0, duration, 1.0]], dtype=torch.float32)
            cls = torch.tensor([self.labels[idx]], dtype=torch.long)
        else:
            # Placeholder for real SED datasets like DESED. 
            # Override this method in a subclass.
            boxes = torch.empty((0, 4), dtype=torch.float32)
            cls = torch.empty((0,), dtype=torch.long)
            
        labels = {"boxes": boxes, "cls": cls}
        
        # Squeeze channel dim for dataloader (will be added back in model)
        return waveform.squeeze(0), labels


def yoho_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    """
    Custom collate function to handle variable numbers of bounding boxes per audio clip.
    Required because torch.utils.data.DataLoader cannot stack tensors of different sizes.
    """
    audio_list = []
    boxes_list = []
    cls_list = []
    
    for audio, labels in batch:
        audio_list.append(audio)
        boxes_list.append(labels['boxes'])
        cls_list.append(labels['cls'])
        
    # Audio is already padded to max_length in __getitem__, so stack is safe
    audio_batch = torch.stack(audio_list, dim=0)  # [B, T]
    
    # We return lists for targets because the Assigner in heads.py handles list padding
    targets = {
        "boxes": boxes_list,  # List of [N_i, 4] tensors
        "cls": cls_list       # List of [N_i] tensors
    }
    
    return audio_batch, targets


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Helper to initialize dataloader with the correct collate_fn"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=yoho_collate_fn,
        pin_memory=True  # Speeds up GPU transfer
    )
