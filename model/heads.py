import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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

class AudioIoULoss(nn.Module):
    def __init__(self, time_weight=2.0, freq_weight=0.5):
        super().__init__()
        self.time_weight = time_weight
        self.freq_weight = freq_weight

    def forward(self, pred_boxes, target_boxes):
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_t = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_p + area_t - inter + 1e-7
        iou = inter / union
        
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=1e-7)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=1e-7)
        
        c_x_p = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        c_y_p = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        c_x_t = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        c_y_t = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        time_penalty = ((c_x_p - c_x_t) / enclose_w).pow(2)
        freq_penalty = ((c_y_p - c_y_t) / enclose_h).pow(2)
        penalty = (self.time_weight * time_penalty) + (self.freq_weight * freq_penalty)
        
        return (1 - iou + penalty).mean()


class AnchorFreeDetectionHead(nn.Module):
    """Modern anchor-free detection head (YOLOv8/v9 style)"""
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        c_ = in_channels // 2
        
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

def decode_boxes(reg: torch.Tensor, stride: int, feat_shape: Tuple[int, int]) -> torch.Tensor:
    dfl = DFL(16)
    dist = dfl(reg).permute(0, 2, 1) 
    h, w = reg.shape[2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=reg.device),
        torch.arange(w, device=reg.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).float().view(1, -1, 2)
    center = (grid + 0.5) * stride
    boxes = torch.cat((center - dist[..., :2], center + dist[..., 2:]), dim=-1) 
    
    boxes[..., [0, 2]] *= feat_shape[1] 
    boxes[..., [1, 3]] *= feat_shape[0] 
    return boxes


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


@torch.no_grad()
def hungarian_matcher(
    pred_boxes: torch.Tensor,     # [N, 4]
    pred_scores: torch.Tensor,    # [N, C]
    gt_boxes: torch.Tensor,       # [M, 4]
    gt_labels: torch.Tensor,      # [M]
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DETR-style Bipartite Matching to eliminate NMS.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # We only match top-K predictions to speed up the Hungarian algorithm
    top_k = min(300, pred_boxes.shape[0])
    max_scores, _ = pred_scores.max(dim=1)
    _, top_idx = torch.topk(max_scores, top_k)
    
    pred_boxes_k = pred_boxes[top_idx]
    pred_scores_k = pred_scores[top_idx]

    # Classification Cost (Focal Loss-like)
    out_prob = pred_scores_k.sigmoid()
    cost_class_mat = -out_prob[:, gt_labels]

    # BBox Cost (L1)
    cost_bbox_mat = torch.cdist(pred_boxes_k, gt_boxes, p=1)

    # Generalized/IoU Cost
    iou_mat = box_iou(pred_boxes_k, gt_boxes)
    cost_giou_mat = -iou_mat

    # Total Cost Matrix
    C = cost_class * cost_class_mat + cost_bbox * cost_bbox_mat + cost_giou * cost_giou_mat
    C = C.cpu().numpy()

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(C)
    
    # Map back to original indices
    matched_pred_idx = top_idx[torch.as_tensor(row_ind, dtype=torch.long)]
    matched_gt_idx = torch.as_tensor(col_ind, dtype=torch.long)

    return matched_pred_idx, matched_gt_idx
