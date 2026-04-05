"""
inference.py
推理 & 阈值调优
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


@torch.no_grad()
def predict(model, loader, device="cpu"):
    """
    对 DataLoader 批量推理。

    Returns
    -------
    probs  : np.ndarray [N]   正类概率
    labels : np.ndarray [N]   真实标签
    """
    model.eval()
    model.to(device)
    all_probs, all_labels = [], []

    for batch in loader:
        event_ids, time_deltas, cont_values, is_same_pkg_ids, labels, padding_mask = batch
        event_ids       = event_ids.to(device)
        time_deltas     = time_deltas.to(device)
        cont_values     = cont_values.to(device)
        is_same_pkg_ids = is_same_pkg_ids.to(device)
        padding_mask    = padding_mask.to(device)

        logits = model(event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_best_threshold_by_ks(y_true, y_prob):
    """按 KS 最大化选阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    best_idx  = np.argmax(ks_values)
    return thresholds[best_idx], ks_values[best_idx]


def find_threshold_by_recall(y_true, y_prob, target_recall=0.8):
    """在 recall >= target_recall 下最大化 precision"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid = recalls[:-1] >= target_recall
    if not valid.any():
        raise ValueError(f"任意阈值均无法达到 recall >= {target_recall}")
    best_idx = np.argmax(precisions[:-1][valid])
    return (
        thresholds[valid][best_idx],
        precisions[:-1][valid][best_idx],
        recalls[:-1][valid][best_idx],
    )


def threshold_analysis(y_true, y_prob, thresholds=None):
    """打印各阈值下的风控核心指标"""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    print(f"{'Threshold':>10} │ {'Precision':>10} {'Recall(坏)':>10} "
          f"{'F1':>8} {'FPR(误杀)':>10}")
    print("─" * 58)

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)

        print(f"{t:>10.2f} │ {precision:>10.4f} {recall:>10.4f} {f1:>8.4f} {fpr:>10.4f}")
