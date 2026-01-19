from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SpecConfig:
    n_mels: int = 128
    hop_length: int = 512
    f_max: int = 8000
    normalized: bool = True

@dataclass
class ModelConfig:
    input_channels: int = 1  # Mono spectrogram
    num_classes: int = 10  # e.g., for ESC-50
    backbone_scales: List[int] = [8, 16, 32]  # Multi-scale
    attention_heads: int = 4
    memory_slots: int = 256
    use_memory: bool = True
    seg_enabled: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-4
    patience: int = 10  # Early stopping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: Optional[str] = "yoho-audio"

@dataclass
class DataConfig:
    dataset: str = "esc50"  # or "urban8k", "audioset"
    root_dir: str = "./data"
    augment_prob: float = 0.5
    spec: SpecConfig = SpecConfig()

    curriculum_enabled: bool = True
    curriculum_start_epoch: int = 0
    curriculum_ramp_epochs: int = 30  # gradual increase difficulty
    complexity_metric: str = "num_events"  # or "overlap_ratio", etc.

@dataclass
class YOHOConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()

cfg = YOHOConfig()
