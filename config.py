from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class SpecConfig:
    sample_rate: int = 44100
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 1024
    f_max: int = 8000
    normalized: bool = True

@dataclass
class AugmentConfig:
    mixup_prob: float = 0.5
    mixup_alpha: float = 1.5
    time_mask_param: int = 30
    freq_mask_param: int = 15

@dataclass
class ModelConfig:
    input_channels: int = 1  
    num_classes: int = 10  
    base_channels: int = 32
    scales: List[int] = field(default_factory=lambda: [4, 8, 16])
    bifpn_layers: int = 2     # Number of BiFPN repeats
    attention_heads: int = 4
    memory_slots: int = 256
    use_memory: bool = False  
    seg_enabled: bool = False
    reg_max: int = 16

@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-4
    clip_grad: float = 5.0
    amp: bool = True
    ema_decay: float = 0.9999
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: Optional[str] = "yoho-audio"

@dataclass
class DataConfig:
    dataset: str = "esc50"  
    root_dir: str = "./data"
    augment_prob: float = 0.5
    spec: SpecConfig = field(default_factory=SpecConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    curriculum_enabled: bool = True
    curriculum_start_epoch: int = 0
    curriculum_ramp_epochs: int = 30  
    complexity_metric: str = "num_events"  

@dataclass
class YOHOConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

cfg = YOHOConfig()
