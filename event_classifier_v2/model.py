"""
model.py
基于 Transformer Encoder 的事件序列二分类模型
- is_same_pkg 改用 nn.Embedding(2, d) 处理
- value 拆分为：连续特征(3维) + 二值Embedding(1维)
"""

import math

import torch
import torch.nn as nn


class tAPE(nn.Module):
    """time Absolute Position Encoding（tAPE, ConvTran）。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024, scale_factor: float = 1.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term * (d_model / max_len))
        pe[:, 1::2] = torch.cos(position * div_term * (d_model / max_len))
        self.register_buffer("pe", scale_factor * pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class EventEmbedding(nn.Module):
    """
    将单个 Event 的特征编码并融合为 d_model 维向量。

    本实验使用：
        event_id + value 维度信息（cont_values + is_same_pkg）
    拼接后映射到 d_model。
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
        del d_time

        self.event_emb = nn.Embedding(num_event_types + 1, d_event, padding_idx=0)
        self.cont_proj = nn.Linear(3, d_cont)
        self.pkg_emb = nn.Embedding(2, d_pkg)
        self.fusion_proj = nn.Linear(d_event + d_cont + d_pkg, d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_ids, time_deltas, cont_values, is_same_pkg_ids):
        """
        event_ids       : [B, L]      LongTensor
        time_deltas     : [B, L, 1]   FloatTensor（已 log1p）
        cont_values     : [B, L, 3]   FloatTensor（已 log1p）
        is_same_pkg_ids : [B, L]      LongTensor  0 or 1

        Returns : [B, L, d_model]
        """
        del time_deltas
        e = self.event_emb(event_ids)              # [B, L, d_event]
        c = self.cont_proj(cont_values)            # [B, L, d_cont]
        pkg = self.pkg_emb(is_same_pkg_ids)        # [B, L, d_pkg]
        x = self.fusion_proj(torch.cat([e, c, pkg], dim=-1))
        x = self.norm(x)
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
    n_heads         : int    注意力头数，默认 4
    n_layers        : int    Encoder 层数，默认 3
    d_ffn           : int    FFN 内层维度，默认 256
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

        self.embedding = EventEmbedding(
            num_event_types=num_event_types,
            d_model=d_model,
            d_event=d_event,
            d_time=d_time,
            d_cont=d_cont,
            d_pkg=d_pkg,
            dropout=dropout,
        )
        self.positional_encoding = tAPE(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,    # 输入格式 [B, L, d_model]
            norm_first=True,     # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.pooling(x, padding_mask)
        logits = self.classifier(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask):
        """推理接口，返回正类概率 [B]"""
        self.eval()
        logits = self.forward(event_ids, time_deltas, cont_values, is_same_pkg_ids, padding_mask)
        return torch.sigmoid(logits)
