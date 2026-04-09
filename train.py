"""
项目级训练入口：
1) 自动加载与处理数据
2) 自动切分 train/val/test
3) 训练并保存模型
4) 导出 train/val/test 的 prob + score CSV
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from event_classifier_v2.dataset import build_dataloader
from event_classifier_v2.inference import predict
from event_classifier_v2.model import EventTransformerClassifier
from event_classifier_v2.pipeline import (
    build_feature_dataframe,
    build_prediction_frame,
    evaluate_probabilities,
    split_train_val_test,
)
from event_classifier_v2.train import train as fit_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_num_event_types(df) -> int:
    max_event_id = int(df["event_seq"].map(max).max())
    return max_event_id


def _default_model_cfg(num_event_types: int) -> Dict[str, float]:
    return {
        "num_event_types": num_event_types,
        "d_model": 32,
        "d_event": 32,
        "d_time": 32,
        "d_cont": 12,
        "d_pkg": 4,
        "n_heads": 4,
        "n_layers": 2,
        "d_ffn": 64,
        "dropout": 0.1,
        "max_seq_len": 32,
        "batch_size": 128,
        "n_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-2,
    }


def _log(stage: str, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{stage}] {message}")


def train(
    seq_data_path: str,
    sample_path: str,
    output_dir: str = "./artifacts",
    cfg: Optional[Dict[str, float]] = None,
    oot_ratio: float = 0.2,
    val_ratio_in_time: float = 0.25,
    random_state: int = 42,
) -> Dict[str, str]:
    """
    训练主方法（自动处理数据、切分、训练、保存和导出）。
    """
    _log("INIT", f"开始训练, random_state={random_state}")
    _set_seed(random_state)
    device = _get_device()
    _log("INIT", f"设备: {device}")

    output_path = Path(output_dir)
    model_dir = output_path / "models"
    pred_dir = output_path / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    _log("PATH", f"输出目录: {output_path.resolve()}")
    _log("PATH", f"模型目录: {model_dir.resolve()}")
    _log("PATH", f"预测目录: {pred_dir.resolve()}")

    _log("DATA", f"加载数据: seq={seq_data_path}, sample={sample_path}")
    df_all = build_feature_dataframe(seq_data_path=seq_data_path, sample_path=sample_path)
    _log("DATA", f"样本总量: {len(df_all)}")
    df_train, df_val, df_test = split_train_val_test(
        df_all,
        oot_ratio=oot_ratio,
        val_ratio_in_time=val_ratio_in_time,
        random_state=random_state,
    )

    num_event_types = _infer_num_event_types(df_all)
    final_cfg = _default_model_cfg(num_event_types=num_event_types)
    if cfg:
        final_cfg.update(cfg)
    _log(
        "CFG",
        (
            f"num_event_types={final_cfg['num_event_types']}, max_seq_len={final_cfg['max_seq_len']}, "
            f"batch_size={final_cfg['batch_size']}, epochs={final_cfg['n_epochs']}, "
            f"lr={final_cfg['lr']}, weight_decay={final_cfg['weight_decay']}"
        ),
    )

    train_loader_fit = build_dataloader(
        df_train,
        batch_size=final_cfg["batch_size"],
        shuffle=True,
        max_seq_len=final_cfg["max_seq_len"],
    )
    train_loader_eval = build_dataloader(
        df_train,
        batch_size=final_cfg["batch_size"],
        shuffle=False,
        max_seq_len=final_cfg["max_seq_len"],
    )
    val_loader = build_dataloader(
        df_val,
        batch_size=final_cfg["batch_size"],
        shuffle=False,
        max_seq_len=final_cfg["max_seq_len"],
    )
    test_loader = build_dataloader(
        df_test,
        batch_size=final_cfg["batch_size"],
        shuffle=False,
        max_seq_len=final_cfg["max_seq_len"],
    )

    n_bad = int(df_train["target"].sum())
    n_good = int(len(df_train) - n_bad)
    pos_weight = n_good / max(n_bad, 1)

    print("=" * 70)
    print("数据切分结果")
    print(
        f"train={len(df_train)} ({len(df_train)/len(df_all):.1%}), "
        f"bad_rate={df_train['target'].mean():.4f}"
    )
    print(
        f"val  ={len(df_val)} ({len(df_val)/len(df_all):.1%}), "
        f"bad_rate={df_val['target'].mean():.4f}"
    )
    print(
        f"test ={len(df_test)} ({len(df_test)/len(df_all):.1%}), "
        f"bad_rate={df_test['target'].mean():.4f}"
    )
    print(f"device={device}, pos_weight={pos_weight:.4f}")
    print(
        f"steps_per_epoch(train)={len(train_loader_fit)}, "
        f"steps_eval(train/val/test)=({len(train_loader_eval)}/{len(val_loader)}/{len(test_loader)})"
    )
    print("=" * 70)

    _log("MODEL", "初始化模型")
    model = EventTransformerClassifier(
        num_event_types=final_cfg["num_event_types"],
        d_model=final_cfg["d_model"],
        d_event=final_cfg["d_event"],
        d_time=final_cfg["d_time"],
        d_cont=final_cfg["d_cont"],
        d_pkg=final_cfg["d_pkg"],
        n_heads=final_cfg["n_heads"],
        n_layers=final_cfg["n_layers"],
        d_ffn=final_cfg["d_ffn"],
        dropout=final_cfg["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log("MODEL", f"可训练参数量: {total_params:,}")

    train_start = time.time()
    _log("TRAIN", "开始模型训练")
    fit_model(
        model=model,
        train_loader=train_loader_fit,
        val_loader=val_loader,
        n_epochs=final_cfg["n_epochs"],
        lr=final_cfg["lr"],
        weight_decay=final_cfg["weight_decay"],
        pos_weight=pos_weight,
        save_dir=str(model_dir),
        device=str(device),
    )
    _log("TRAIN", f"训练完成, 耗时: {time.time() - train_start:.1f}s")

    model_path = model_dir / "best_model.pt"
    _log("LOAD", f"加载最佳模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    predictions = {}
    metrics = {}

    for split, split_df, loader in [
        ("train", df_train, train_loader_eval),
        ("val", df_val, val_loader),
        ("test", df_test, test_loader),
    ]:
        _log("EVAL", f"开始评估并导出 {split} 集结果")
        probs, labels = predict(model, loader, device=device)
        metrics[split] = evaluate_probabilities(labels, probs)
        pred_df = build_prediction_frame(split_df, probs, split=split)
        split_csv = pred_dir / f"{split}_predictions.csv"
        pred_df.to_csv(split_csv, index=False)
        predictions[split] = str(split_csv)
        _log(
            "EVAL",
            f"{split} 完成: rows={len(split_df)}, auc={metrics[split]['auc']:.4f}, "
            f"pr_auc={metrics[split]['pr_auc']:.4f}, ks={metrics[split]['ks']:.4f}",
        )

    print("\n训练效果（按最佳模型）")
    print(f"{'split':>8} | {'AUC':>8} | {'PR-AUC':>8} | {'KS':>8}")
    print("-" * 45)
    for split in ("train", "val", "test"):
        m = metrics[split]
        print(f"{split:>8} | {m['auc']:>8.4f} | {m['pr_auc']:>8.4f} | {m['ks']:>8.4f}")

    metadata = {
        "model_path": str(model_path),
        "prediction_csv": predictions,
        "metrics": metrics,
        "split_rows": {
            "train": int(len(df_train)),
            "val": int(len(df_val)),
            "test": int(len(df_test)),
        },
        "cfg": final_cfg,
        "data_paths": {
            "seq_data_path": seq_data_path,
            "sample_path": sample_path,
        },
    }
    metadata_path = output_path / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n模型保存: {model_path}")
    print(f"元数据保存: {metadata_path}")
    print("预测明细:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {predictions[split]}")
    _log("DONE", "训练流程结束")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "train_csv": predictions["train"],
        "val_csv": predictions["val"],
        "test_csv": predictions["test"],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练事件序列模型并导出预测结果")
    parser.add_argument("--seq-data-path", default="./dataset/event_lst.csv")
    parser.add_argument("--sample-path", default="./dataset/tmp_0403.csv")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--oot-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio-in-time", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(
        seq_data_path=args.seq_data_path,
        sample_path=args.sample_path,
        output_dir=args.output_dir,
        oot_ratio=args.oot_ratio,
        val_ratio_in_time=args.val_ratio_in_time,
        random_state=args.random_state,
    )
