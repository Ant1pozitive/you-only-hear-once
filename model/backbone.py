import torch
import torch.nn as nn

class Conv(nn.Module):
    """Standard Conv + BN + Act"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """CSP Bottleneck"""
    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, 1, 1)
        self.cv2 = Conv(c_, c, 3, 1)
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))

class CSPBlock(nn.Module):
    """CSP Block with n bottlenecks"""
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        c_ = out_ch // 2
        self.cv1 = Conv(in_ch, c_, 1, 1)
        self.cv2 = Conv(in_ch, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_ch, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class MultiScaleBackbone(nn.Module):
    """Multi-scale backbone for audio spectrograms with CSP blocks"""
    def __init__(self, scales=[4, 8, 16], base_channels=32):  # Smaller for audio
        super().__init__()
        self.entry = nn.Sequential(
            Conv(1, base_channels, 3, 1),
            Conv(base_channels, base_channels, 3, 2)  # Downsample time
        )

        self.stages = nn.ModuleList()
        channels = base_channels
        for scale in scales:
            stage = nn.Sequential(
                Conv(channels, channels * 2, 3, (1, 2)),  # Temporal downsample
                CSPBlock(channels * 2, channels * 2, n=3),
                CSPBlock(channels * 2, channels * 2, n=3)
            )
            self.stages.append(stage)
            channels *= 2

        # Simple FPN (top-down)
        self.fpn = nn.ModuleList()
        for i in range(len(scales)):
            self.fpn.append(Conv(channels // (2 ** i), base_channels * 4, 1))

    def forward(self, x):
        x = self.entry(x)  # [B, C, F, T]
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # FPN fusion: upsample and add
        p = features[-1]
        outs = [self.fpn[-1](p)]
        for i in range(len(features) - 2, -1, -1):
            p = nn.functional.interpolate(p, scale_factor=(1, 2), mode='nearest')  # Upsample time
            p = torch.cat((p, features[i]), dim=1)
            p = CSPBlock(p.shape[1], features[i].shape[1], n=1)(p)
            outs.insert(0, self.fpn[i](p))

        return outs  # [P3, P4, P5] multi-scale
