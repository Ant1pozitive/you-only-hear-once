"""
Data loader for DESED dataset (strong/weak labeled)
Assumes DESED public structure: audio/ + metadata/ with .tsv
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict
import torchaudio
import os
import pandas as pd
from .augmentations import AudioAugmentations

class DESEDDataset(Dataset):
    """DESED dataset loader"""
    def __init__(self, root: str, mode: str = 'weak', augment_prob: float = 0.5):
        """
        mode: 'weak', 'strong', 'synthetic'
        """
        self.root = root
        self.mode = mode
        self.augment_prob = augment_prob
        self.sr = 44100
        
        meta_dir = os.path.join(root, 'metadata')
        audio_dir = os.path.join(root, 'audio')
        
        if mode == 'weak':
            meta_file = os.path.join(meta_dir, 'public.tsv')
            audio_subdir = 'public'
        elif mode == 'strong':
            meta_file = os.path.join(meta_dir, 'public_strong.tsv')
            audio_subdir = 'public'
        elif mode == 'synthetic':
            meta_file = None  # or specific
            audio_subdir = 'synthetic20_strong'
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.meta = pd.read_csv(meta_file, sep='\t')
        self.files = self.meta['filename'].tolist()
        self.audio_dir = os.path.join(audio_dir, audio_subdir)
        
        self.augmenter = AudioAugmentations(
            prob=augment_prob,
            sample_rate=self.sr
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        path = os.path.join(self.audio_dir, self.files[idx])
        waveform, sr = torchaudio.load(path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = waveform.mean(0)  # mono
        
        waveform = self.augmenter(waveform)
        
        # Labels from meta
        row = self.meta.iloc[idx]
        if self.mode == 'weak':
            events = row['event_labels'].split(',') if pd.notna(row['event_labels']) else []
            # Pseudo strong: whole clip
            boxes = torch.tensor([[0.0, 0.0, 10.0, 8000.0]] * len(events))  # 10sec clip
            cls = torch.tensor([self.event_to_id(e) for e in events])  # need event_to_id map
            masks = torch.ones(len(events), 128, waveform.shape[0] // 512)  # pseudo
        else:  # strong
            # Parse onset, offset, event_label
            onsets = row['onset'].split(',') if pd.notna(row['onset']) else []
            offsets = row['offset'].split(',') if pd.notna(row['offset']) else []
            events = row['event_label'].split(',') if pd.notna(row['event_label']) else []
            
            boxes = torch.tensor([[float(on), 0.0, float(off), 8000.0] for on, off in zip(onsets, offsets)])
            cls = torch.tensor([self.event_to_id(e) for e in events])
            masks = self.generate_masks(boxes, waveform.shape[0] // 512)  # implement binary masks
        
        labels = {"boxes": boxes, "cls": cls, "masks": masks}
        
        return waveform, labels
    
    def event_to_id(self, event: str) -> int:
        """Map event label to ID (implement class map)"""
        class_map = {'Alarm_bell_ringing': 0, 'Blender': 1, 'Cat': 2, 'Dishes': 3, 'Dog': 4, 
                     'Electric_shaver_toothbrush': 5, 'Frying': 6, 'Running_water': 7, 'Speech': 8, 'Vacuum_cleaner': 9}
        return class_map.get(event, -1)
    
    def generate_masks(self, boxes: torch.Tensor, time_frames: int) -> torch.Tensor:
        """Generate binary masks for events"""
        masks = torch.zeros(len(boxes), self.cfg.spec.n_mels, time_frames)
        # Fill masks based on time/freq (pseudo for now)
        return masks

    def collate_fn(batch):
        """Custom collate for variable lengths"""
        audios, labels = zip(*batch)
        return torch.nn.utils.rnn.pad_sequence(audios, batch_first=True), labels  # labels as list