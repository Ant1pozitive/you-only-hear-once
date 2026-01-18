import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    """YOLO-style detection head for bounding boxes on spectrogram."""
    def __init__(self, channels: int, num_classes: int, anchors: List[List[int]]):
        super().__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        # Output: (x,y,w,h,conf) + classes per anchor
        out_dim = self.num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(channels, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, Out, Freq//stride, Time//stride]

class SegmentationHead(nn.Module):
    """Mask head for instance segmentation on spectrogram."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.proto = nn.Conv2d(channels, 32, 1)  # Proto coefficients
        self.mask = nn.Conv2d(channels, num_classes, 1)  # Class masks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proto = self.proto(x)  # [B, 32, H, W]
        mask = self.mask(x)    # [B, Classes, H, W]
        return proto, mask
