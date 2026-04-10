"""
model.py
基于 LSTM Encoder 的事件序列二分类模型
- 保留特征融合部分（one-hot + 连续特征 + 二值特征 + 时间特征）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsolutePositionalEncoding(nn.Module):
    """标准绝对位置编码（sin/cos）。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class EventEmbedding(nn.Module):
    """
    将单个 Event 的特征编码并融合为 d_model 维向量。

    本实验编码方式：
        1) event_id -> one-hot 事件类型特征
        2) 拼接 value 特征（cont_values + is_same_pkg）与时间差（小时）特征
        3) 两层网络: Linear -> Norm -> ReLU -> Linear 映射到 d_model
    """

    def __init__(
        self,
        num_event_types: int   = 14,
        d_model:         int   = 128,
        d_event:         int   = 32,
        d_time:          int   = 24,
        d_cont:          int   = 56,   # 连续 value 维度
        d_pkg:           int   = 16,   # is_same_pkg embedding 维度
        dropout:         float = 0.1,
    ):
        super().__init__()
        del d_event, d_time, d_cont, d_pkg

        self.onehot_dim = num_event_types + 1  # 含 PAD(0)
        input_dim = self.onehot_dim + 3 + 1 + 1  # onehot + cont_values(3) + is_same_pkg + time_hours
        hidden_dim = max(d_model, input_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_ids, time_deltas, cont_values, is_same_pkg_ids):
        """
        event_ids       : [B, L]      LongTensor
        time_deltas     : [B, L, 1]   FloatTensor（已 log1p）
        cont_values     : [B, L, 3]   FloatTensor（已 log1p）
        is_same_pkg_ids : [B, L]      LongTensor  0 or 1

        Returns : [B, L, d_model]
        """
        event_onehot = F.one_hot(event_ids, num_classes=self.onehot_dim).float()  # [B, L, onehot_dim]
        pkg = is_same_pkg_ids.unsqueeze(-1).float()                                # [B, L, 1]
        time_hours = time_deltas                                                   # [B, L, 1]
        x_in = torch.cat([event_onehot, cont_values, pkg, time_hours], dim=-1)
        x = self.feature_mlp(x_in)
        x = self.dropout(x)
        return x


class MeanPooling(nn.Module):
    """对有效位置取均值，忽略 PAD"""

    def forward(self, x, padding_mask):
        """
        x            : [B, L, d_model]
        padding_mask : [B, L]  True = PAD

        Returns : [B, d_model]
        """
        mask = (~padding_mask).float().unsqueeze(-1)         # [B, L, 1]
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class EventTransformerClassifier(nn.Module):
    """
    事件序列二分类模型

    Parameters
    ----------
    num_event_types : int    事件类型数（不含 PAD），默认 14
    d_model         : int    隐层维度，默认 128
    d_event         : int    event_id embedding 维度，默认 32
    d_time          : int    time_delta 投影维度，默认 24
    d_cont          : int    连续 value 投影维度，默认 56
    d_pkg           : int    is_same_pkg embedding 维度，默认 16
    n_heads         : int    兼容旧配置，LSTM 版本中不使用
    n_layers        : int    LSTM 层数，默认 3
    d_ffn           : int    兼容旧配置，LSTM 版本中不使用
    dropout         : float  dropout 率，默认 0.1
    """

    def __init__(
        self,
        num_event_types: int   = 14,
        d_model:         int   = 128,
        d_event:         int   = 32,
        d_time:          int   = 24,
        d_cont:          int   = 56,
        d_pkg:           int   = 16,
        n_heads:         int   = 4,
        n_layers:        int   = 3,
        d_ffn:           int   = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()
        del n_heads, d_ffn

        self.embedding = EventEmbedding(
            num_event_types=num_event_types,
            d_model=d_model,
            d_event=d_event,
            d_time=d_time,
            d_cont=d_cont,
            d_pkg=d_pkg,
            dropout=dropout,
        )
        # LSTM 按序建模，不再依赖显式位置编码。
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.pooling = MeanPooling()

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask):
        """
        Parameters
        ----------
        event_ids       : [B, L]      LongTensor
        time_deltas     : [B, L, 1]   FloatTensor
        cont_values     : [B, L, 3]   FloatTensor
        is_same_pkg_ids : [B, L]      LongTensor
        padding_mask    : [B, L]      BoolTensor  True=PAD

        Returns
        -------
        logits : [B]   未经 sigmoid（训练用 BCEWithLogitsLoss）
        """
        x = self.embedding(event_ids, time_deltas, cont_values, is_same_pkg_ids)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=padding_mask.size(1)
        )
        x = self.pooling(x, padding_mask)
        logits = self.classifier(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask):
        """推理接口，返回正类概率 [B]"""
        self.eval()
        logits = self.forward(event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask)
        return torch.sigmoid(logits)
