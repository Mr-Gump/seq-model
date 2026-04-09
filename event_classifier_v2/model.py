"""
model.py
基于 Transformer Encoder 的事件序列二分类模型
- is_same_pkg 改用 nn.Embedding(2, d) 处理
- value 拆分为：连续特征(3维) + 二值Embedding(1维)
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """RoPE: 对输入向量按位置进行旋转编码。"""

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("RoPE 要求 d_model 为偶数")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)  # [L]
        freqs = torch.einsum("l,d->ld", positions, self.inv_freq)  # [L, D/2]
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=x.dtype).unsqueeze(0)  # [1, L, D]
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=x.dtype).unsqueeze(0)  # [1, L, D]
        return (x * cos) + (self._rotate_half(x) * sin)


class EventEmbedding(nn.Module):
    """
    将单个 Event 的特征编码并融合为 d_model 维向量。

    本实验编码方式：
        1) event_id -> 8维可学习 embedding
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

        self.event_emb = nn.Embedding(num_event_types + 1, 8, padding_idx=0)
        input_dim = 8 + 3 + 1 + 1  # event_emb(8) + cont_values(3) + is_same_pkg + time_hours
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
        event_vec = self.event_emb(event_ids)                                      # [B, L, 8]
        pkg = is_same_pkg_ids.unsqueeze(-1).float()                                # [B, L, 1]
        time_hours = time_deltas                                                   # [B, L, 1]
        x_in = torch.cat([event_vec, cont_values, pkg, time_hours], dim=-1)
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
    n_heads         : int    注意力头数，默认 4
    n_layers        : int    Encoder 层数，默认 3
    d_ffn           : int    FFN 内层维度，默认 256
    dropout         : float  dropout 率，默认 0.1
    rope_base       : float  RoPE 频率基数，默认 10000.0
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
        rope_base:       float = 10000.0,
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
        self.rope = RotaryPositionEmbedding(d_model=d_model, base=rope_base)

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
        x = self.rope(x)
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
