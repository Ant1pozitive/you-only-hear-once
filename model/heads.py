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
        """
        pred_score: [bs, num_queries, num_classes]
        gt_score:   [bs, num_queries] IoU-aware target scores
        label:      [bs, num_queries] class labels (-1 for negative)
        """
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


class CIoULoss(nn.Module):
    """Complete IoU Loss with distance, aspect ratio and center penalty"""
    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes, target_boxes: [N, 4] in format (x1,y1,x2,y2)
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
        
        # Enclosure
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_diag = enclose_w.pow(2) + enclose_h.pow(2) + 1e-7
        
        # Center distance
        c_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        c_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        t_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        t_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (c_x - t_cx).pow(2) + (c_y - t_cy).pow(2)
        
        # Aspect ratio consistency
        v = 4 / (torch.pi**2) * (
            torch.atan((target_boxes[:, 2] - target_boxes[:, 0]) / 
                       (target_boxes[:, 3] - target_boxes[:, 1] + 1e-7)) -
            torch.atan((pred_boxes[:, 2] - pred_boxes[:, 0]) / 
                       (pred_boxes[:, 3] - pred_boxes[:, 1] + 1e-7))
        ).pow(2)
        
        alpha = v / ((1 - iou) + v + 1e-7)
        
        return (1 - iou + (center_dist / enclose_diag) + (alpha * v)).mean()


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
        """
        feats: list of multi-scale feature maps [P3, P4, P5]
        Returns:
            cls_scores: list of [bs, num_classes, h, w]
            reg_preds:  list of [bs, 4*(reg_max+1), h, w]
        """
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
        # Using highest resolution feature (P3)
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


def task_aligned_assigner(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thres: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SimOTA-like task-aligned assigner
    Returns:
        assigned_gt_ids: [num_anchors] index of assigned gt or -1
        assigned_scores: [num_anchors] IoU-aware scores for positives
        pos_mask: [num_anchors] bool mask for positives
    """
    # Simplified version for batch_size=1; extend for full batch
    # Compute IoU matrix [num_anchors, num_gt]
    iou = box_iou(pred_boxes, gt_boxes)  # implement box_iou below
    
    # Cost = - (score * iou)
    scores = pred_scores.gather(1, gt_labels.unsqueeze(0).repeat(pred_scores.shape[0], 1))
    cost = - (scores * iou)
    
    # Assign with min cost (Hungarian-like but approx)
    assigned_ids = cost.argmin(dim=1)
    
    # Filter low IoU
    max_iou = iou.max(dim=1)[0]
    pos_mask = max_iou >= iou_thres
    
    assigned_scores = max_iou[pos_mask]
    assigned_ids[~pos_mask] = -1
    
    return assigned_ids, assigned_scores, pos_mask


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes [N,4] and [M,4]"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)