"""
model.py
基于 Transformer Encoder 的事件序列二分类模型
- is_same_pkg 改用 nn.Embedding(2, d) 处理
- value 拆分为：连续特征(3维) + 二值Embedding(1维)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCPE(nn.Module):
    """
    Feature-based Cycle-aware Time Positional Encoding.
    输出为语义嵌入 + 时间编码。
    """

    def __init__(self, d_model: int, num_event_types: int, padding_idx: int = 0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model 必须为偶数")
        self.d_model = d_model
        self.num_freqs = d_model // 2
        self.K = num_event_types
        self.padding_idx = padding_idx

        init_freqs = torch.tensor(
            [2 * math.pi * k / self.num_freqs for k in range(self.num_freqs)], dtype=torch.float32
        )
        self.w = nn.Parameter(init_freqs)
        self.W_mu = nn.Parameter(torch.randn(self.num_freqs, num_event_types) * 0.1)
        self.event_embed = nn.Embedding(num_event_types, d_model, padding_idx=padding_idx)

    def forward(self, t: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        """
        t  : [B, L]   处理后的连续时间
        ki : [B, L]   事件类型 id（含 PAD=0）
        """
        ki_onehot = F.one_hot(ki, num_classes=self.K).float()  # [B, L, K]
        mu = torch.einsum("fk,blk->blf", self.W_mu, ki_onehot)
        mu = F.softplus(mu)

        theta = t.unsqueeze(-1) * self.w.view(1, 1, -1)
        cos_enc = mu * torch.cos(theta)
        sin_enc = mu * torch.sin(theta)
        time_encoding = torch.stack([cos_enc, sin_enc], dim=-1).view(t.size(0), t.size(1), self.d_model)

        semantic_embed = self.event_embed(ki)
        out = semantic_embed + time_encoding

        if self.padding_idx is not None:
            valid_mask = (ki != self.padding_idx).unsqueeze(-1).float()
            out = out * valid_mask
        return out


class EventEmbedding(nn.Module):
    """
    将单个 Event 的特征编码并融合为 d_model 维向量。

    本实验使用：
        FCPE(事件语义 + 时间编码) + value 维度信息（cont_values + is_same_pkg）
    融合后映射到 d_model。
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
        del d_event, d_time
        self.fcpe = FCPE(d_model=d_model, num_event_types=num_event_types + 1, padding_idx=0)
        self.cont_proj = nn.Linear(3, d_cont)
        self.pkg_emb = nn.Embedding(2, d_pkg)
        self.value_proj = nn.Linear(d_cont + d_pkg, d_model)

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
        t = time_deltas.squeeze(-1)
        e = self.fcpe(t, event_ids)                # [B, L, d_model]
        c = self.cont_proj(cont_values)            # [B, L, d_cont]
        pkg = self.pkg_emb(is_same_pkg_ids)        # [B, L, d_pkg]
        v = self.value_proj(torch.cat([c, pkg], dim=-1))
        x = e + v
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
