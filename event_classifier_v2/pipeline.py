"""
pipeline.py
数据准备、数据切分、评分与评估工具函数
"""

from __future__ import annotations

import ast
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def _parse_event_list(value):
    """将 event_lst 的字符串表示解析为 Python list。"""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


def _elapsed_hours_from_first_event(events):
    """
    使用事件绝对时间戳，构造“距首事件的小时数”序列。
    输入 events 中每个元素需包含 event_time（秒级时间戳）。
    """
    if len(events) == 0:
        return []
    if "event_time" not in events[0]:
        raise ValueError("event_lst 中缺少 event_time，无法构造首事件相对时间")

    first_ts = float(events[0]["event_time"])
    elapsed_hours = []
    for event in events:
        if "event_time" not in event:
            raise ValueError("event_lst 中存在缺少 event_time 的事件")
        ts = float(event["event_time"])
        elapsed_hours.append(max(ts - first_ts, 0.0) / 3600.0)
    return elapsed_hours


def build_feature_dataframe(seq_data_path: str, sample_path: str) -> pd.DataFrame:
    """
    从事件序列文件与样本标签文件构建训练用 DataFrame。
    产出列至少包含:
        sn, verify_time, target, event_seq, time_delta_seq, value_seq
    """
    df_seq = pd.read_csv(seq_data_path)
    df_sample = pd.read_csv(sample_path)

    if "event_lst" not in df_seq.columns:
        raise ValueError("事件序列文件缺少列: event_lst")
    if "sn" not in df_seq.columns or "sn" not in df_sample.columns:
        raise ValueError("输入文件缺少主键列: sn")
    if "target" not in df_sample.columns:
        raise ValueError("样本文件缺少标签列: target")
    if "verify_time" not in df_sample.columns:
        raise ValueError("样本文件缺少时间列: verify_time")

    df_seq = df_seq.drop_duplicates(subset=["sn"]).copy()
    df_sample = df_sample.drop_duplicates(subset=["sn"]).copy()

    df_seq["event_lst"] = df_seq["event_lst"].apply(_parse_event_list)
    df_seq["event_seq"] = df_seq["event_lst"].apply(
        lambda events: [event["event_id"] for event in events]
    )
    df_seq["time_delta_seq"] = df_seq["event_lst"].apply(_elapsed_hours_from_first_event)
    df_seq["value_seq"] = df_seq["event_lst"].apply(
        lambda events: [
            (
                event["onway_amt"],
                event["onway_cnt"],
                event["borrow_amt"],
                event["is_same_pkg"],
            )
            for event in events
        ]
    )
    df = pd.merge(
        df_sample,
        df_seq[["sn", "event_seq", "time_delta_seq", "value_seq"]],
        on="sn",
        how="inner",
    ).copy()

    df = df.dropna(subset=["event_seq", "time_delta_seq", "value_seq", "target"])
    df["target"] = df["target"].astype(int)
    return df


def split_train_val_test(
    df: pd.DataFrame,
    oot_ratio: float = 0.2,
    val_ratio_in_time: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    先按时间排序，后段作为 test（OOT），前段做分层随机划分 train/val。
    默认比例约为 60% / 20% / 20%。
    """
    if not 0 < oot_ratio < 1:
        raise ValueError("oot_ratio 必须在 (0,1) 内")
    if not 0 < val_ratio_in_time < 1:
        raise ValueError("val_ratio_in_time 必须在 (0,1) 内")

    df_sorted = df.sort_values("verify_time").reset_index(drop=True)
    oot_start = int(len(df_sorted) * (1 - oot_ratio))

    df_in_time = df_sorted.iloc[:oot_start].copy()
    df_test = df_sorted.iloc[oot_start:].copy()

    df_train, df_val = train_test_split(
        df_in_time,
        test_size=val_ratio_in_time,
        random_state=random_state,
        stratify=df_in_time["target"],
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def prob_to_score(prob: np.ndarray, base_odds: float = 0.284) -> np.ndarray:
    """
    将概率转换为评分（与 notebook 使用公式一致）。
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    factor = 20 / np.log(2)
    offset = 600 + factor * np.log(base_odds)
    odds = p / (1 - p)
    score = offset - factor * np.log(odds)
    return score


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    返回基础效果指标：AUC / PR-AUC / KS。
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": float(np.max(tpr - fpr)),
    }


def build_prediction_frame(df: pd.DataFrame, probs: np.ndarray, split: str) -> pd.DataFrame:
    """
    构造输出明细，包含 train/val/test 的 prob 与 score。
    """
    result = df[["sn", "verify_time", "target"]].copy()
    result["split"] = split
    result["prob"] = np.asarray(probs, dtype=np.float64)
    result["score"] = prob_to_score(result["prob"].to_numpy())
    return result
