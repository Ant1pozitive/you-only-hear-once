"""
Evaluation metrics for SED: event-based F1 and PSDS.
Uses psds_eval for PSDS computation.
"""

import numpy as np
import pandas as pd
from psds_eval import PSDSEval

def event_based_f1(pred_events: dict, gt_events: dict, t_collar: float = 0.2, f_collar: float = 0.1) -> float:
    """
    Simple event-based F1-score with collar tolerance.
    
    Args:
        pred_events: dict with 'boxes' (N,4), 'labels' (N,)
        gt_events: dict with 'boxes' (M,4), 'labels' (M,)
        t_collar: time tolerance in seconds
        f_collar: frequency tolerance in Hz
    """
    pred_boxes = pred_events["boxes"].cpu().numpy()
    pred_labels = pred_events["labels"].cpu().numpy()
    gt_boxes = gt_events["boxes"].cpu().numpy()
    gt_labels = gt_events["labels"].cpu().numpy()

    tp = 0
    used = np.zeros(len(gt_boxes), dtype=bool)

    for p_box, p_label in zip(pred_boxes, pred_labels):
        for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if used[i]:
                continue
            t_overlap = min(p_box[2], g_box[2]) - max(p_box[0], g_box[0])
            f_overlap = min(p_box[3], g_box[3]) - max(p_box[1], g_box[1])
            if (t_overlap >= -t_collar and f_overlap >= -f_collar and p_label == g_label):
                tp += 1
                used[i] = True
                break

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def proper_psds(
    preds_list: list[dict],
    gts_list: list[dict],
    dtc_threshold: float = 0.5,
    gtc_threshold: float = 0.5,
    cttc_threshold: float = 0.3,
    alpha_ct: float = 0.0,
    alpha_st: float = 0.0
) -> dict:
    """
    Compute PSDS using psds_eval library.
    
    Args:
        preds_list: list of prediction dicts (each with 'boxes', 'labels')
        gts_list: list of ground truth dicts (each with 'boxes', 'labels')
        ... thresholds for PSDS scenarios
    
    Returns:
        dict with PSDS scores for different scenarios
    """
    # Prepare dataframes in psds_eval format
    pred_rows = []
    gt_rows = []

    for i, (pred, gt) in enumerate(zip(preds_list, gts_list)):
        filename = f"file_{i:04d}"

        # Predictions
        pred_boxes = pred["boxes"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        for box, label in zip(pred_boxes, pred_labels):
            pred_rows.append({
                "filename": filename,
                "onset": float(box[0]),
                "offset": float(box[2]),
                "event_label": str(label)  # convert to string
            })

        # Ground truth
        gt_boxes = gt["boxes"].cpu().numpy()
        gt_labels = gt["labels"].cpu().numpy()
        for box, label in zip(gt_boxes, gt_labels):
            gt_rows.append({
                "filename": filename,
                "onset": float(box[0]),
                "offset": float(box[2]),
                "event_label": str(label)
            })

    if not pred_rows or not gt_rows:
        return {"psds_scenario1": 0.0, "psds_scenario2": 0.0}

    pred_df = pd.DataFrame(pred_rows)
    gt_df = pd.DataFrame(gt_rows)

    # Get unique event labels
    class_names = sorted(gt_df["event_label"].unique())

    # Initialize evaluator
    psds_eval = PSDSEval(
        ground_truth=gt_df,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold
    )

    # Add operating point
    psds_eval.add_operating_point(pred_df)

    # Compute PSDS for different scenarios
    psds_scenario1 = psds_eval.compute()
    psds_scenario2 = psds_eval.compute(alpha_ct=alpha_ct, alpha_st=alpha_st)

    return {
        "psds_scenario1": psds_scenario1,
        "psds_scenario2": psds_scenario2,
        "class_names": class_names
    }