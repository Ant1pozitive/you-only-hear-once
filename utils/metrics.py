import torch
import numpy as np

def event_based_f1(pred_events, gt_events, t_collar=0.2, f_collar=0.1):
    """Event-based F1 with collar tolerance"""
    pred = pred_events['boxes'].cpu().numpy() if isinstance(pred_events['boxes'], torch.Tensor) else pred_events['boxes']
    gt = gt_events['boxes'].cpu().numpy() if isinstance(gt_events['boxes'], torch.Tensor) else gt_events['boxes']
    
    tp = 0
    for g in gt:
        for p in pred:
            if (abs(g[0] - p[0]) <= t_collar and abs(g[2] - p[2]) <= t_collar and
                abs(g[1] - p[1]) <= f_collar and abs(g[3] - p[3]) <= f_collar):
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)

def psds(pred, gt, dtc_threshold=0.5, gtc_threshold=0.5, cttc_threshold=0.3, alpha_ct=0, alpha_st=0):
    """Simplified PSDS (Polyphonic Sound Detection Score)"""
    # Based on DCASE: detection, cross-trigger, spatial-temporal
    # For simplicity: average F1 with penalties
    f1 = event_based_f1(pred, gt)
    ct_penalty = alpha_ct * np.random.rand()  # Placeholder for cross-trigger
    st_penalty = alpha_st * np.random.rand()  # Spatial-temporal
    return f1 * (1 - ct_penalty) * (1 - st_penalty)
