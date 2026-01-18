import torch
import torch.nn as nn
import torch.nn.functional as F

class DFL(nn.Module):
    """Distribution Focal Loss module"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x):
        n, c, a = x.shape[:3]
        x = x.reshape(n, c, a, self.reg_max + 1).softmax(-1)
        return (x * torch.arange(self.reg_max + 1, device=x.device).float()).sum(-1)

class VarifocalLoss(nn.Module):
    """Varifocal loss for classification"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, alpha=0.75, gamma=2.0):
        iou = target  # Assume target has IoU-aware labels
        weight = alpha * pred.pow(gamma) * (1 - iou) + iou * target
        return F.binary_cross_entropy_with_logits(pred, iou * target, reduction='none') * weight.detach()

class CIoULoss(nn.Module):
    """Complete IoU Loss for regression"""
    def forward(self, pred, target):
        lt = torch.max(pred[:, :2] - target[:, :2], torch.zeros_like(pred[:, :2]))
        rb = torch.max(target[:, 2:] - pred[:, 2:], torch.zeros_like(pred[:, 2:]))
        wh = torch.clamp(rb + lt, min=0)
        overlap = wh[:, 0] * wh[:, 1]
        ap = (pred[:, 2:] - pred[:, :2]).prod(1)
        ag = (target[:, 2:] - target[:, :2]).prod(1)
        union = ap + ag - overlap + 1e-7
        iou = overlap / union
        enclose_wh = (torch.min(pred[:, 2:], target[:, 2:]) - torch.max(pred[:, :2], target[:, :2])).clamp(min=0)
        enclose_c2 = enclose_wh[:, 0].pow(2) + enclose_wh[:, 1].pow(2) + 1e-7
        rho2 = (pred[:, :2] + pred[:, 2:]) / 2 - (target[:, :2] + target[:, 2:]) / 2
        rho2 = rho2[:, 0].pow(2) + rho2[:, 1].pow(2)
        v = 4 / (torch.pi ** 2) * (torch.atan(target[:, 2] / target[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3])).pow(2)
        alpha = v / (1 - iou + v + 1e-7)
        return 1 - iou + rho2 / enclose_c2 + alpha * v

class DetectionHead(nn.Module):
    """Anchor-free YOLO detection head"""
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        stem_ch = in_channels // 4
        self.stem = nn.ModuleList([Conv(in_channels, stem_ch, 3, 1) for _ in range(3)])  # For 3 scales
        self.cls = nn.ModuleList([nn.Conv2d(stem_ch, num_classes, 1) for _ in range(3)])
        self.reg = nn.ModuleList([nn.Conv2d(stem_ch, 4 * (reg_max + 1), 1) for _ in range(3)])
        self.dfl = DFL(reg_max)

    def forward(self, feats):
        cls_scores = []
        bbox_dists = []
        for i, f in enumerate(feats):
            f = self.stem[i](f)
            cls_scores.append(self.cls[i](f))
            bbox_dists.append(self.reg[i](f))
        return cls_scores, bbox_dists

class SegmentationHead(nn.Module):
    """Prototype-based segmentation"""
    def __init__(self, in_channels, num_classes, proto_dim=32):
        super().__init__()
        self.proto_dim = proto_dim
        self.proto = Conv(in_channels, proto_dim, 1, 1, act=False)
        self.mask = Conv(in_channels, num_classes, 1, 1, act=False)

    def forward(self, feats):
        # Assume feats[0] for high-res mask (P3)
        proto = self.proto(feats[0]).relu()  # [B, 32, H, W]
        mask_coeffs = self.mask(feats[0])  # [B, classes, H, W]
        return proto, mask_coeffs.sigmoid()

def compute_loss(preds, targets, model):
    """Combined loss"""
    cls_scores, bbox_dists = preds['det']
    proto, mask_coeffs = preds['seg'] if 'seg' in preds else (None, None)
    
    # Targets: dict with 'boxes' [B,N,4], 'cls' [B,N], 'masks' [B,N,H,W]
    # For simplicity, assume batched and padded; in practice use matcher
    device = cls_scores[0].device
    batch_size = cls_scores[0].shape[0]
    
    # Detection loss (placeholder for full matcher; assume simple)
    vfl = VarifocalLoss()
    ciou = CIoULoss()
    dfl = model.module.dfl if hasattr(model, 'module') else model.dfl  # For DDP
    
    cls_loss = 0
    reg_loss = 0
    for scale in range(len(cls_scores)):
        pred_cls = cls_scores[scale].permute(0,2,3,1).reshape(batch_size, -1, model.num_classes).sigmoid()
        pred_dist = bbox_dists[scale].permute(0,2,3,1).reshape(batch_size, -1, 4*(model.reg_max+1))
        pred_box = dfl(pred_dist.view(batch_size, -1, 4, model.reg_max+1)).view(batch_size, -1, 4)
        
        # Assume targets per scale; simple: all on P3
        if scale == 0:
            tgt_cls = targets['cls'].float()  # One-hot or something
            tgt_box = targets['boxes']
            cls_loss += vfl(pred_cls, tgt_cls).mean()
            reg_loss += ciou(pred_box, tgt_box).mean()
    
    # Seg loss
    seg_loss = 0
    if proto is not None:
        tgt_masks = targets['masks']  # [B, N, H, W]
        masks = (proto.unsqueeze(1) * mask_coeffs.unsqueeze(2)).sum(2).sigmoid()  # Simple proto activation
        seg_loss = F.binary_cross_entropy(masks, tgt_masks)
    
    return cls_loss + 2.0 * reg_loss + seg_loss
