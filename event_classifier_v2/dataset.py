"""
dataset.py
事件序列数据集 & DataLoader 构建
适配实际数据：df_train / df_test / df_oot
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── 常量 ─────────────────────────────────────────────────────────────────────
NUM_EVENT_TYPES = 14
MAX_SEQ_LEN     = 128


# ── 数据集 ────────────────────────────────────────────────────────────────────
class EventSequenceDataset(Dataset):
    """
    直接从 DataFrame 构建数据集。

    DataFrame 必须包含以下列：
        sn            : 订单号（仅保留用于追溯，不参与训练）
        target        : 标签 0/1
        event_seq     : List[int]，事件ID列表（1~14）
        time_delta_seq: List[float]，时间间隔列表（秒），首个为 0
        value_seq     : List[Tuple]，四元组列表
                        每个元组 = (onway_amt, onway_cnt, borrow_amt, is_same_pkg)

    Parameters
    ----------
    df          : pd.DataFrame
    max_seq_len : int  截断长度，保留最近 N 个事件
    """

    def __init__(self, df, max_seq_len: int = MAX_SEQ_LEN):
        self.df          = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        event_seq      = list(row["event_seq"])
        time_delta_seq = list(row["time_delta_seq"])
        value_seq      = list(row["value_seq"])
        label          = float(row["target"])

        # ── 截断：保留最近 max_seq_len 个事件 ────────────────────────────────
        if len(event_seq) > self.max_seq_len:
            event_seq      = event_seq[-self.max_seq_len:]
            time_delta_seq = time_delta_seq[-self.max_seq_len:]
            value_seq      = value_seq[-self.max_seq_len:]

        seq_len = len(event_seq)

        # ── event_id：LongTensor [L] ──────────────────────────────────────────
        event_ids = torch.tensor(event_seq, dtype=torch.long)   # 1-indexed，0 留给 PAD

        # ── time_delta：log1p 压缩 → [L, 1] ──────────────────────────────────
        time_deltas = torch.tensor(
            [np.log1p(t / 3600 ) for t in time_delta_seq],
            dtype=torch.float32
        ).unsqueeze(-1)   # [L, 1]

        # ── value：拆分为连续特征 + 二值特征 ──────────────────────────────────
        # value_seq 中每个元组：(onway_amt, onway_cnt, borrow_amt, is_same_pkg)
        cont_values    = []   # [onway_amt, onway_cnt, borrow_amt]，做 log1p
        is_same_pkg_ids = []  # 0/1，走 Embedding

        for v in value_seq:
            onway_amt, onway_cnt, borrow_amt, is_same_pkg = v
            cont_values.append([
                np.log1p(float(onway_amt / 1000)),
                np.log1p(float(onway_cnt)),
                np.log1p(float(borrow_amt / 1000)),
            ])
            is_same_pkg_ids.append(int(is_same_pkg))

        cont_values     = torch.tensor(cont_values,     dtype=torch.float32)  # [L, 3]
        is_same_pkg_ids = torch.tensor(is_same_pkg_ids, dtype=torch.long)     # [L]

        label = torch.tensor(label, dtype=torch.float32)

        return event_ids, time_deltas, cont_values, is_same_pkg_ids, label, seq_len


# ── Collate（动态 padding） ───────────────────────────────────────────────────
def collate_fn(batch):
    """
    将变长序列 padding 到 batch 内最大长度。

    Returns
    -------
    event_ids       [B, L]      LongTensor
    time_deltas     [B, L, 1]   FloatTensor
    cont_values     [B, L, 3]   FloatTensor
    is_same_pkg_ids [B, L]      LongTensor   （0/1）
    labels          [B]         FloatTensor
    padding_mask    [B, L]      BoolTensor   True = PAD 位置
    """
    event_ids_list, time_deltas_list, cont_values_list, \
        is_same_pkg_list, labels_list, lens = zip(*batch)

    B       = len(batch)
    max_len = max(lens)

    event_ids       = torch.zeros(B, max_len, dtype=torch.long)
    time_deltas     = torch.zeros(B, max_len, 1)
    cont_values     = torch.zeros(B, max_len, 3)
    is_same_pkg_ids = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask    = torch.ones(B, max_len, dtype=torch.bool)    # True = PAD

    for i, (eid, td, cv, pkg, _, l) in enumerate(batch):
        event_ids[i, :l]       = eid
        time_deltas[i, :l]     = td
        cont_values[i, :l]     = cv
        is_same_pkg_ids[i, :l] = pkg
        padding_mask[i, :l]    = False   # 有效位置

    labels = torch.stack(labels_list).float()   # [B]

    return event_ids, time_deltas, cont_values, is_same_pkg_ids, labels, padding_mask


def build_dataloader(df, batch_size=256, shuffle=True, max_seq_len=MAX_SEQ_LEN):
    dataset = EventSequenceDataset(df, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
    )
