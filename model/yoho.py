import torch
import torch.nn as nn
import torchaudio.transforms as T
from .backbone import MultiScaleBackbone
from .heads import DetectionHead, SegmentationHead
from .memory_aug import AudioMemoryBank

class YOHO(nn.Module):
    """You Only Hear Once: Audio Detection and Segmentation."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.spec_transform = T.MelSpectrogram(
            n_mels=cfg.spec.n_mels, hop_length=cfg.spec.hop_length, f_max=cfg.spec.f_max, normalized=cfg.spec.normalized
        )
        self.backbone = MultiScaleBackbone(cfg.backbone_scales)
        channels = 64  # From backbone
        anchors = [[10,13], [16,30], [33,23]]  # Time-freq anchors (example)
        self.det_head = DetectionHead(channels, cfg.num_classes, anchors)
        self.seg_head = SegmentationHead(channels, cfg.num_classes) if cfg.seg_enabled else None
        self.memory = AudioMemoryBank(cfg.memory_slots, channels) if cfg.use_memory else None
        self.attention = nn.MultiheadAttention(channels, cfg.attention_heads)  # Auditory attention

    def forward(self, audio: torch.Tensor, is_train: bool = True) -> dict:
        # Audio: [B, Time] -> Spectrogram [B, 1, Freq, Time//hop]
        spec = self.spec_transform(audio).unsqueeze(1)
        features = self.backbone(spec)  # [B, C, Freq, Time]
        
        # Auditory attention: Attend over freq-time
        features_flat = features.flatten(2).permute(2, 0, 1)  # [Freq*Time, B, C]
        attn_out, _ = self.attention(features_flat, features_flat, features_flat)
        features = attn_out.permute(1, 2, 0).view_as(features)
        
        # Detection
        det_out = self.det_head(features)
        
        # Segmentation (if enabled)
        seg_out = self.seg_head(features) if self.seg_head else None
        
        # Memory augmentation (for continual)
        if self.memory:
            query = features.mean([2,3]).unsqueeze(1)  # Global query
            mem_vec = self.memory.read(query)
            features += mem_vec.unsqueeze(2).unsqueeze(3)
            if is_train:
                self.memory.write(query, features.mean([2,3]).unsqueeze(1))
                self.memory.synthesize()
        
        return {"det": det_out, "seg": seg_out}
    
    @torch.inference_mode()
    def infer(self, audio: torch.Tensor, nms_thresh: float = 0.5):
        out = self(audio, is_train=False)
        # Post-process: NMS on det, mask activation on seg (simplified)
        # Implement full post-processing as needed
        return out
