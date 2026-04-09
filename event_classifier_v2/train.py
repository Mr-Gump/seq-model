"""
train.py
训练 & 评估主流程
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp
import numpy as np
from pathlib import Path


# ── KS 指标 ──────────────────────────────────────────────────────────────────
def ks_score(y_true, y_prob):
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    ks, _ = ks_2samp(pos, neg)
    return ks


def _unpack_batch(batch, device):
    """统一 batch 解包 & 搬运到 device"""
    event_ids, time_deltas, cont_values, is_same_pkg_ids, labels, padding_mask = batch
    return (
        event_ids.to(device),
        time_deltas.to(device),
        cont_values.to(device),
        is_same_pkg_ids.to(device),
        labels.to(device),
        padding_mask.to(device),
    )


# ── 单 Epoch 训练 ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        event_ids, time_deltas, cont_values, is_same_pkg_ids, labels, padding_mask = \
            _unpack_batch(batch, device)

        optimizer.zero_grad()
        logits = model(event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask)
        loss   = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    return {
        "loss": total_loss / len(all_labels),
        "auc":  roc_auc_score(all_labels, all_probs),
        "ks":   ks_score(all_labels, all_probs),
    }


# ── 评估 ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        event_ids, time_deltas, cont_values, is_same_pkg_ids, labels, padding_mask = \
            _unpack_batch(batch, device)

        logits = model(event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    return {
        "loss":   total_loss / len(all_labels),
        "auc":    roc_auc_score(all_labels, all_probs),
        "ks":     ks_score(all_labels, all_probs),
        "pr_auc": average_precision_score(all_labels, all_probs),
    }


# ── 主训练流程 ────────────────────────────────────────────────────────────────
def train(
    model,
    train_loader,
    val_loader,
    n_epochs:     int   = 20,
    lr:           float = 1e-3,
    weight_decay: float = 1e-2,
    pos_weight:   float = 3.0,
    save_dir:     str   = "./checkpoints",
    device:       str   = "auto",
    max_ks_gap:   float = 0.03,
    ks_gap_penalty: float = 0.5,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model  = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32)
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_selection_score = -float("inf")
    best_val_ks = 0.0
    best_ks_gap = float("inf")
    best_epoch = 0

    print(f"Device: {device}  |  pos_weight: {pos_weight:.2f}\n")
    print(f"{'Epoch':>6} │ {'TrLoss':>8} {'TrAUC':>7} {'TrKS':>7} │"
          f" {'ValLoss':>8} {'ValAUC':>7} {'ValKS':>7} {'ValPR':>7} {'KSGap':>7}")
    print("─" * 80)

    for epoch in range(1, n_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        vl = evaluate(model, val_loader, criterion, device)
        ks_gap = abs(tr["ks"] - vl["ks"])

        # 选择标准：优先高 Val KS，同时惩罚 Train/Val KS 差异过大
        penalty = max(0.0, ks_gap - max_ks_gap) * ks_gap_penalty
        selection_score = vl["ks"] - penalty

        flag = " ◀ best" if selection_score > best_selection_score else ""
        print(
            f"{epoch:>6} │ {tr['loss']:>8.4f} {tr['auc']:>7.4f} {tr['ks']:>7.4f} │"
            f" {vl['loss']:>8.4f} {vl['auc']:>7.4f} {vl['ks']:>7.4f} {vl['pr_auc']:>7.4f} {ks_gap:>7.4f}{flag}"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_ks = vl["ks"]
            best_ks_gap = ks_gap
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(
        f"\n最优 Epoch {best_epoch}，Val KS = {best_val_ks:.4f}，"
        f"Train/Val KS Gap = {best_ks_gap:.4f}，Selection Score = {best_selection_score:.4f}"
    )
    print(f"模型已保存至 {save_dir / 'best_model.pt'}")
    return model
