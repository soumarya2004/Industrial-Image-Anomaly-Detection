from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray)->dict:
    cm=confusion_matrix(y_true, y_pred, labels=[0, 1])
    return{
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }

def select_threshold_by_f1(y_true_val: np.ndarray, scores_val: np.ndarray)->float:
    precisions, recalls, thresholds = precision_recall_curve(y_true_val, scores_val)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1s=np.divide(
        2*precisions[:-1]*recalls[:-1],
        precisions[:-1]+recalls[:-1],
        out=np.zeros_like(precisions[:-1]),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    best_idx=int(np.argmax(f1s))
    return float(thresholds[best_idx])

def anomaly_detection_report_dict(y_true: np.ndarray, scores: np.ndarray, threshold: float)->dict:
    y_pred=(scores >= threshold).astype(int)
    cm=confusion_matrix(y_true, y_pred, labels=[0, 1])
    return{
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }

def pixel_level_metrics(gt_masks: np.ndarray, pred_heatmaps: np.ndarray)->dict:
    gt_flat=gt_masks.flatten()
    pred_flat=pred_heatmaps.flatten()
    pixel_roc_auc=(
        float(roc_auc_score(gt_flat, pred_flat)) if len(set(gt_flat.tolist())) > 1 else float("nan")
    )
    ious, precisions, recalls = [], [], []
    for gt_mask, heatmap in zip(gt_masks, pred_heatmaps):
        norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Otsu-style: threshold at the mean + 1 std of the normalized heatmap
        # as a simple, data-driven per-image binarization for IoU reporting.
        thresh = norm.mean() + norm.std()
        pred_mask=(norm>=thresh).astype(np.uint8)
        gt=gt_mask.astype(np.uint8)
        intersection=np.logical_and(pred_mask, gt).sum()
        union=np.logical_or(pred_mask, gt).sum()
        iou=intersection/union if union > 0 else 1.0 if gt.sum() == 0 else 0.0
        ious.append(iou)
        tp=intersection
        fp=pred_mask.sum()-tp
        fn=gt.sum()-tp
        precisions.append(tp/(tp+fp) if (tp+fp)>0 else 0.0)
        recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
    return{
        "pixel_roc_auc": pixel_roc_auc,
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "mean_pixel_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "mean_pixel_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "n_images": int(len(gt_masks)),
    }

def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y_true, scores)

def pr_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, scores)
