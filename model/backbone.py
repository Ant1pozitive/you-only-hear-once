import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """Standard Conv + BN + SiLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel Attention Module (SE-like)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BiPathBlock(nn.Module):
    """Bi-Path block: separate temporal and frequency processing"""
    def __init__(self, channels):
        super().__init__()
        # Temporal path (focus on time axis)
        self.temp_conv = nn.Sequential(
            Conv(channels, channels, k=(1, 5), p=(0, 2)),  # Wide in time, narrow in freq
            Conv(channels, channels, k=(1, 5), p=(0, 2))
        )
        # Frequency path (focus on mel-bins axis)
        self.freq_conv = nn.Sequential(
            Conv(channels, channels, k=(5, 1), p=(2, 0)),  # Wide in freq, narrow in time
            Conv(channels, channels, k=(5, 1), p=(2, 0))
        )
        self.fusion = Conv(channels * 2, channels, k=1)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        temp = self.temp_conv(x)
        freq = self.freq_conv(x)
        fused = torch.cat([temp, freq], dim=1)
        fused = self.fusion(fused)
        return self.ca(fused) + x  # residual


class BiPathBackbone(nn.Module):
    """Backbone with Bi-Path Fusion + Channel Attention"""
    def __init__(self, base_channels=32, scales=[4, 8, 16]):
        super().__init__()
        self.entry = nn.Sequential(
            Conv(1, base_channels, 3, 1),
            Conv(base_channels, base_channels * 2, 3, stride=(1, 2))  # down time
        )

        self.stages = nn.ModuleList()
        channels = base_channels * 2
        for scale in scales:
            stage = nn.Sequential(
                Conv(channels, channels * 2, 3, stride=(1, 2)),  # down time again
                *[BiPathBlock(channels * 2) for _ in range(3)],
                BiPathBlock(channels * 2)
            )
            self.stages.append(stage)
            channels *= 2

        # Simple FPN-like
        self.lateral_convs = nn.ModuleList([
            Conv(ch * 2, base_channels * 4, 1) for ch in [base_channels*2, base_channels*4, base_channels*8]
        ])

    def forward(self, x):
        # x: [B, 1, Mel, Time]
        x = self.entry(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # FPN top-down fusion
        outs = []
        p = features[-1]
        outs.append(self.lateral_convs[-1](p))

        for i in range(len(features)-2, -1, -1):
            p = F.interpolate(p, scale_factor=(1, 2), mode='nearest')
            p = torch.cat([p, features[i]], dim=1)
            p = Conv(p.shape[1], features[i].shape[1], 1)(p)  # 1x1 reduce
            outs.insert(0, self.lateral_convs[i](p))

        return outs  # multi-scale features
