import torch
import torch.nn as nn
import torchaudio.transforms as T

class MultiScaleBackbone(nn.Module):
    """CSPDarknet-inspired backbone with multi-scale for audio spectrograms."""
    def __init__(self, scales: List[int], channels: int = 64):
        super().__init__()
        self.scales = scales
        self.entry = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.pyramid = nn.ModuleList()
        for s in scales:
            layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=(1, s)),  # Pool time axis
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU()
            )
            self.pyramid.append(layer)
        self.fusion = nn.Conv2d(channels * 2 * len(scales), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)  # [B, C, Freq, Time]
        features = []
        for layer in self.pyramid:
            features.append(layer(x))
        # Upsample to match time dims (assume fixed input size or pad)
        max_t = max(f.shape[-1] for f in features)
        features = [nn.functional.interpolate(f, size=(f.shape[-2], max_t)) for f in features]
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)
