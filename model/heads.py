import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Conv(nn.Module):
    """Standard convolution + BN + activation"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """Distribution Focal Loss module for bounding box regression"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.arange(reg_max + 1).float())

    def forward(self, x):
        """x: [bs, 4*(reg_max+1), h*w] → [bs, 4, h*w]"""
        b, c, a = x.shape
        x = x.view(b, 4, self.reg_max + 1, a).permute(0, 3, 1, 2)
        x = F.softmax(x, dim=2)
        return torch.sum(x * self.project, dim=2)


class VarifocalLoss(nn.Module):
    """Varifocal Loss for classification (IoU-aware)"""
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        pos_mask = label >= 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_score.device)
        
        pred = pred_score[pos_mask]
        gt = gt_score[pos_mask]
        
        weight = alpha * pred.pow(gamma) * (1 - gt) + gt
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(
                pred.float(), gt.float(), reduction='none'
            ) * weight.detach()
        
        return loss.mean()


class AudioIoULoss(nn.Module):
    """Asymmetric IoU Loss specifically designed for Audio Spectrograms.
    Heavily penalizes time axis errors, lightly penalizes frequency axis errors."""
    def __init__(self, time_weight=2.0, freq_weight=0.5):
        super().__init__()
        self.time_weight = time_weight
        self.freq_weight = freq_weight

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes, target_boxes: [N, 4] in format (x1,y1,x2,y2) 
        where x is time, y is frequency.
        """
        # Intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        # Union
        area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_t = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_p + area_t - inter + 1e-7
        
        iou = inter / union
        
        # Enclosure box (smallest bounding box containing both)
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=1e-7)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=1e-7)
        
        # Center coordinates
        c_x_p = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        c_y_p = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        c_x_t = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        c_y_t = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        # Asymmetric distance penalties
        time_penalty = ((c_x_p - c_x_t) / enclose_w).pow(2)
        freq_penalty = ((c_y_p - c_y_t) / enclose_h).pow(2)
        
        # Combined asymmetric penalty
        penalty = (self.time_weight * time_penalty) + (self.freq_weight * freq_penalty)
        
        return (1 - iou + penalty).mean()


class AnchorFreeDetectionHead(nn.Module):
    """Modern anchor-free detection head (YOLOv8/v9 style)"""
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        c_ = in_channels // 4
        
        # Shared stem
        self.stem = nn.Sequential(
            Conv(in_channels, c_, 3, 1),
            Conv(c_, c_, 3, 1)
        )
        
        # Classification branch
        self.cls = nn.Sequential(
            Conv(c_, c_, 3, 1),
            nn.Conv2d(c_, num_classes, 1)
        )
        
        # Regression branch (DFL)
        self.reg = nn.Sequential(
            Conv(c_, c_, 3, 1),
            nn.Conv2d(c_, 4 * (reg_max + 1), 1)
        )
        
        self.dfl = DFL(reg_max)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores = []
        reg_preds = []
        
        for feat in feats:
            feat = self.stem(feat)
            cls = self.cls(feat)          # [bs, classes, h, w]
            reg = self.reg(feat)          # [bs, 4*(reg_max+1), h, w]
            cls_scores.append(cls)
            reg_preds.append(reg)
            
        return cls_scores, reg_preds


class SegmentationHead(nn.Module):
    """Prototype-based instance segmentation head"""
    def __init__(self, in_channels, num_classes, proto_channels=32):
        super().__init__()
        self.proto = Conv(in_channels, proto_channels, 1, 1, act=False)
        self.mask_coeff = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        high_res = feats[0]
        proto = self.proto(high_res).relu()
        coeffs = self.mask_coeff(high_res)
        return proto, coeffs


def decode_boxes(reg: torch.Tensor, stride: int, feat_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Decode DFL output to absolute boxes (x1,y1,x2,y2)
    reg: [bs, 4*(reg_max+1), h, w]
    """
    dfl = DFL(16)
    dist = dfl(reg)  # [bs, 4, h*w]
    dist = dist.permute(0, 2, 1)  # [bs, h*w, 4]
    
    h, w = reg.shape[2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=reg.device),
        torch.arange(w, device=reg.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).float().view(1, -1, 2)
    
    center = (grid + 0.5) * stride
    boxes = torch.cat((
        center - dist[..., :2],
        center + dist[..., 2:]
    ), dim=-1)  # [bs, h*w, 4]
    
    # Scale to spec coordinates (mel, time)
    boxes[..., [0, 2]] *= feat_shape[1]  # time axis
    boxes[..., [1, 3]] *= feat_shape[0]  # freq axis
    
    return boxes


def batched_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute batched IoU between [B, N, 4] and [B, M, 4]
    Returns: [B, N, M] tensor of IoUs
    """
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    
    union = area1[..., :, None] + area2[..., None, :] - inter
    return inter / (union + 1e-7)


def task_aligned_assigner(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    mask_gt: torch.Tensor = None,
    iou_thres: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully batched SimOTA-like task-aligned assigner.
    Args:
        pred_boxes: [B, A, 4]
        pred_scores: [B, A, C]
        gt_boxes: [B, M, 4]
        gt_labels: [B, M]
        mask_gt: [B, M] boolean mask indicating valid GT boxes (to ignore padding)
    Returns:
        assigned_gt_ids: [B, A] index of assigned gt or -1
        assigned_scores: [B, A] IoU-aware scores for positives
        pos_mask: [B, A] bool mask for positives
    """
    B, A, C = pred_scores.shape
    M = gt_boxes.shape[1]
    
    # Auto-infer mask if not provided (assume valid boxes have area > 0)
    if mask_gt is None:
        mask_gt = (gt_boxes.sum(dim=-1) > 0)

    # 1. Compute IoUs [B, A, M]
    ious = batched_box_iou(pred_boxes, gt_boxes)
    
    # 2. Gather class scores corresponding to GT labels
    # Expand labels to index into pred_scores: [B, A, C] -> [B, A, M]
    gt_labels_exp = gt_labels.unsqueeze(1).expand(B, A, M)
    scores_for_gt = pred_scores.gather(2, gt_labels_exp)
    
    # 3. Compute Alignment Metric (Cost = score * iou)
    align_metric = scores_for_gt * ious
    
    # Zero out invalid ground truths using the mask
    align_metric = align_metric * mask_gt.unsqueeze(1).float()
    
    # 4. Assign each anchor to the GT with the maximum alignment metric
    assigned_scores, assigned_gt_ids = align_metric.max(dim=-1)  # [B, A]
    
    # 5. Extract the corresponding IoUs for the assigned targets
    assigned_ious = ious.gather(2, assigned_gt_ids.unsqueeze(-1)).squeeze(-1)  # [B, A]
    
    # 6. Create positive mask
    pos_mask = (assigned_scores > 0) & (assigned_ious >= iou_thres)
    
    # 7. Nullify non-positive assignments
    assigned_gt_ids[~pos_mask] = -1
    assigned_scores[~pos_mask] = 0.0
    
    return assigned_gt_ids, assigned_scores, pos_mask
