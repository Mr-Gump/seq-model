"""
main.py
直接使用 df_train / df_test / df_oot 训练 & 评估

DataFrame 列说明：
    sn             : 订单号
    verify_dt      : 进件时间
    target         : 标签 0/1
    event_seq      : List[int]
    time_delta_seq : List[float]
    value_seq      : List[Tuple(onway_amt, onway_cnt, borrow_amt, is_same_pkg)]
"""

import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import roc_auc_score

from .dataset   import build_dataloader
from .model     import EventTransformerClassifier
from .train     import train, evaluate
from .inference import predict, find_best_threshold_by_ks, threshold_analysis

# ── 可复现性 ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 配置 ──────────────────────────────────────────────────────────────────────
CFG = dict(
    max_seq_len = 128,
    batch_size  = 128,
    n_epochs    = 20,
    lr          = 1e-3,
    weight_decay= 1e-2,
    save_dir    = "./checkpoints",
    # 模型结构
    num_event_types = 13,
    d_model     = 48,
    d_event     = 32,
    d_time      = 32,
    d_cont      = 12,   # 连续特征：onway_amt, onway_cnt, borrow_amt
    d_pkg       = 4,   # is_same_pkg embedding
    n_heads     = 4,
    n_layers    = 2,
    d_ffn       = 72,
    dropout     = 0.1,
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main(df_train: pd.DataFrame, df_test: pd.DataFrame, df_oot: pd.DataFrame):
    device = get_device()

    # ── 1. 构建 DataLoader ────────────────────────────────────────────────────
    train_loader = build_dataloader(df_train, batch_size=CFG["batch_size"],
                                    shuffle=True,  max_seq_len=CFG["max_seq_len"])
    test_loader  = build_dataloader(df_test,  batch_size=CFG["batch_size"],
                                    shuffle=False, max_seq_len=CFG["max_seq_len"])
    oot_loader   = build_dataloader(df_oot,   batch_size=CFG["batch_size"],
                                    shuffle=False, max_seq_len=CFG["max_seq_len"])

    # ── 2. 计算 pos_weight（由训练集决定）────────────────────────────────────
    n_bad      = df_train["target"].sum()
    n_good     = len(df_train) - n_bad
    pos_weight = n_good / n_bad
    print(f"训练集 | 总量:{len(df_train)}  好:{n_good}  坏:{int(n_bad)}  "
          f"坏率:{n_bad/len(df_train):.2%}  pos_weight:{pos_weight:.2f}")
    print(f"测试集 | 总量:{len(df_test)}   坏率:{df_test['target'].mean():.2%}")
    print(f"OOT集  | 总量:{len(df_oot)}    坏率:{df_oot['target'].mean():.2%}\n")

    # ── 3. 初始化模型 ─────────────────────────────────────────────────────────
    model = EventTransformerClassifier(
        num_event_types = CFG["num_event_types"],
        d_model         = CFG["d_model"],
        d_event         = CFG["d_event"],
        d_time          = CFG["d_time"],
        d_cont          = CFG["d_cont"],
        d_pkg           = CFG["d_pkg"],
        n_heads         = CFG["n_heads"],
        n_layers        = CFG["n_layers"],
        d_ffn           = CFG["d_ffn"],
        dropout         = CFG["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：{total_params:,}\n")

    # ── 4. 训练 ──────────────────────────────────────────────────────────────
    train(
        model        = model,
        train_loader = train_loader,
        val_loader   = test_loader,     # 用 test 集做验证集（选最优 epoch）
        n_epochs     = CFG["n_epochs"],
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
        pos_weight   = pos_weight,
        save_dir     = CFG["save_dir"],
        device       = device,
    )

    # ── 5. 加载最优权重 ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(f"{CFG['save_dir']}/best_model.pt", map_location=device)
    )

    # ── 6. Test 集评估 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Test 集】阈值分析")
    test_probs, test_labels = predict(model, test_loader, device=device)
    test_auc = roc_auc_score(test_labels, test_probs)
    best_thr, best_ks = find_best_threshold_by_ks(test_labels, test_probs)
    print(f"AUC={test_auc:.4f}  KS={best_ks:.4f}  最优阈值={best_thr:.4f}")
    threshold_analysis(test_labels, test_probs)

    # ── 7. OOT 集评估 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【OOT 集】阈值分析")
    oot_probs, oot_labels = predict(model, oot_loader, device=device)
    oot_auc = roc_auc_score(oot_labels, oot_probs)
    _, oot_ks = find_best_threshold_by_ks(oot_labels, oot_probs)
    print(f"AUC={oot_auc:.4f}  KS={oot_ks:.4f}")
    threshold_analysis(oot_labels, oot_probs)

    # ── 8. 输出预测结果（带 sn） ──────────────────────────────────────────────
    df_test_result = df_test[["sn", "verify_time", "target"]].copy()
    df_test_result["prob"] = test_probs

    df_oot_result = df_oot[["sn", "verify_time", "target"]].copy()
    df_oot_result["prob"] = oot_probs

    return model, df_test_result, df_oot_result
