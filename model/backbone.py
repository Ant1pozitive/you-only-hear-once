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
            Conv(channels, channels, k=(1, 5), p=(0, 2)),
            Conv(channels, channels, k=(1, 5), p=(0, 2))
        )
        # Frequency path (focus on mel-bins axis)
        self.freq_conv = nn.Sequential(
            Conv(channels, channels, k=(5, 1), p=(2, 0)),
            Conv(channels, channels, k=(5, 1), p=(2, 0))
        )
        self.fusion = Conv(channels * 2, channels, k=1)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        temp = self.temp_conv(x)
        freq = self.freq_conv(x)
        fused = torch.cat([temp, freq], dim=1)
        fused = self.fusion(fused)
        return self.ca(fused) + x


class AudioConformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        
        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.SiLU(inplace=True),
            nn.Linear(channels * expansion, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_flat = x.view(B, C, -1).transpose(1, 2)
        nx = self.norm1(x_flat)
        attn_out, _ = self.mhsa(nx, nx, nx)
        x_flat = x_flat + attn_out
        
        x_conv = x_flat.transpose(1, 2).view(B, C, H, W)
        nx_conv = self.norm2(x_flat).transpose(1, 2).view(B, C, H, W)
        x_conv = x_conv + self.conv(nx_conv)
        
        x_conv_flat = x_conv.view(B, C, -1).transpose(1, 2)
        nx_ffn = self.norm3(x_conv_flat)
        x_out_flat = x_conv_flat + self.ffn(nx_ffn)
        
        return x_out_flat.transpose(1, 2).view(B, C, H, W)


class BiFPNBlock(nn.Module):
    """Fast Normalized Fusion BiFPN Block"""
    def __init__(self, feature_size, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        
        # Top-down convs
        self.p4_td_conv = Conv(feature_size, feature_size, k=3)
        self.p3_out_conv = Conv(feature_size, feature_size, k=3)
        
        # Bottom-up convs
        self.p4_out_conv = Conv(feature_size, feature_size, k=3)
        self.p5_out_conv = Conv(feature_size, feature_size, k=3)
        
        # Learnable weights for fast normalized fusion
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        p3, p4, p5 = inputs
        
        # Top-down pathway
        w1 = F.relu(self.w1)
        weight1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.p4_td_conv(weight1[0] * p4 + weight1[1] * p5_up)
        
        w2 = F.relu(self.w2)
        weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_out = self.p3_out_conv(weight2[0] * p3 + weight2[1] * p4_up)
        
        # Bottom-up pathway
        w3 = F.relu(self.w3)
        weight3 = w3 / (torch.sum(w3, dim=0) + self.epsilon)
        p3_down = F.max_pool2d(p3_out, kernel_size=2, stride=2)
        # Handle padding if sizes don't match exactly due to max_pool
        if p3_down.shape[-2:] != p4.shape[-2:]:
            p3_down = F.interpolate(p3_down, size=p4.shape[-2:], mode='nearest')
        p4_out = self.p4_out_conv(weight3[0] * p4 + weight3[1] * p4_td + weight2[2] * p3_down)
        
        w4 = F.relu(self.w4)
        weight4 = w4 / (torch.sum(w4, dim=0) + self.epsilon)
        p4_down = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        if p4_down.shape[-2:] != p5.shape[-2:]:
            p4_down = F.interpolate(p4_down, size=p5.shape[-2:], mode='nearest')
        p5_out = self.p5_out_conv(weight4[0] * p5 + weight4[1] * p4_down)
        
        return [p3_out, p4_out, p5_out]


class BiPathBackbone(nn.Module):
    def __init__(self, base_channels=32, scales=[4, 8, 16], bifpn_layers=2):
        super().__init__()
        self.entry = nn.Sequential(
            Conv(1, base_channels, 3, 1),
            Conv(base_channels, base_channels * 2, 3, stride=(1, 2))
        )

        self.stages = nn.ModuleList()
        channels = base_channels * 2
        out_channels_list = []
        
        for scale in scales:
            stage = nn.Sequential(
                Conv(channels, channels * 2, 3, stride=(1, 2)),
                *[BiPathBlock(channels * 2) for _ in range(3)],
                BiPathBlock(channels * 2),
                AudioConformerBlock(channels * 2, num_heads=4)
            )
            self.stages.append(stage)
            channels *= 2
            out_channels_list.append(channels)

        # Uniform feature dimension for BiFPN
        self.fpn_dim = base_channels * 4
        self.proj_convs = nn.ModuleList([
            Conv(ch, self.fpn_dim, 1) for ch in out_channels_list
        ])
        
        self.bifpn = nn.Sequential(
            *[BiFPNBlock(self.fpn_dim) for _ in range(bifpn_layers)]
        )

    def forward(self, x):
        # x: [B, 1, Mel, Time]
        x = self.entry(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # Project to same dimension and apply BiFPN
        projected = [proj(feat) for proj, feat in zip(self.proj_convs, features)]
        bifpn_outs = self.bifpn(projected)

        return bifpn_outs
