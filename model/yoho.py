import torch
import torch.nn as nn
import torchaudio.transforms as T
from .backbone import MultiScaleBackbone
from .heads import DetectionHead, SegmentationHead, compute_loss
from .memory_aug import AudioMemoryBank

class YOHO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spec_transform = T.MelSpectrogram(
            sample_rate=16000, n_mels=cfg.n_mels, hop_length=cfg.hop_length, f_max=cfg.f_max, normalized=cfg.normalized
        )
        self.backbone = MultiScaleBackbone()
        in_ch = 128  # From backbone fpn
        self.det_head = DetectionHead(in_ch, cfg.num_classes)
        self.seg_head = SegmentationHead(in_ch, cfg.num_classes) if cfg.seg_enabled else None
        self.memory = AudioMemoryBank(cfg.memory_slots, in_ch) if cfg.use_memory else None
        self.attention = nn.MultiheadAttention(in_ch, cfg.attention_heads, batch_first=True)
        self.num_classes = cfg.num_classes
        self.reg_max = 16

    def forward(self, audio, targets=None):
        spec = self.spec_transform(audio).unsqueeze(1)  # [B,1,F,T]
        feats = self.backbone(spec)
        
        # Attention: flatten freq-time
        b, c, f, t = feats[0].shape  # Use P3 for attn
        flat = feats[0].view(b, c, f*t).permute(0, 2, 1)  # [B, F*T, C]
        attn_out, _ = self.attention(flat, flat, flat)
        feats[0] = attn_out.permute(0, 2, 1).view(b, c, f, t)
        
        det = self.det_head(feats)
        seg = self.seg_head(feats) if self.seg_head else None
        
        # Memory
        if self.memory:
            query = feats[0].mean([2,3]).unsqueeze(1)  # [B,1,C]
            mem = self.memory.read(query)
            for i in range(len(feats)):
                feats[i] += mem.unsqueeze(2).unsqueeze(3)
            if self.training:
                self.memory.write(query, feats[0].mean([2,3]).unsqueeze(1))
                self.memory.synthesize()
        
        preds = {"det": det, "seg": seg}
        if self.training and targets:
            return compute_loss(preds, targets, self)
        return preds

    @torch.no_grad()
    def infer(self, audio, conf_thres=0.25, iou_thres=0.45):
        preds = self.forward(audio)
        cls_scores, bbox_dists = preds['det']
        # Post-process: decode + NMS per scale
        outputs = []
        for i in range(len(cls_scores)):
            scores = cls_scores[i].sigmoid()
            dists = bbox_dists[i].softmax(-1)
            boxes = self.dfl(dists)  # Decode to ltrb
            # NMS (torchvision.ops.nms or custom)
            # Assume simple: select max
            max_scores, labels = scores.max(-1)
            idx = max_scores > conf_thres
            outputs.append({"boxes": boxes[idx], "scores": max_scores[idx], "labels": labels[idx]})
        
        if 'seg' in preds:
            proto, coeffs = preds['seg']
            masks = (proto @ coeffs.view(coeffs.shape[0], self.num_classes, -1).T).sigmoid().view(-1, *proto.shape[2:])
            outputs[0]['masks'] = masks  # Attach to main
        
        return outputs[0]  # Merged
