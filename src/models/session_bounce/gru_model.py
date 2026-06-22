"""Session Bounce GRU 모델 (계획서 §3.5).

입력:
  x_num (B, L, 6)  숫자 채널 [is_view,is_cart,is_remove,is_purchase,gap_log,price_log]
  x_cat (B, L)     카테고리 인덱스(0=padding)
출력:
  logit (B,)       sigmoid 전 단일 로짓 → churn30 확률
"""
import torch
import torch.nn as nn


class SessionBounceGRU(nn.Module):
    def __init__(
        self,
        n_categories,
        n_num=6,
        cat_emb=32,
        num_proj=16,
        hidden=64,
        layers=1,
        dropout=0.2,
    ):
        super().__init__()
        # category padding index 0 → 임베딩 0 고정(계획서 공통규칙: padding 명시)
        self.cat_emb = nn.Embedding(n_categories, cat_emb, padding_idx=0)
        self.num_proj = nn.Linear(n_num, num_proj)
        self.gru = nn.GRU(
            input_size=cat_emb + num_proj,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_num, x_cat):
        e = self.cat_emb(x_cat)                 # (B,L,cat_emb)
        p = self.num_proj(x_num)                # (B,L,num_proj)
        h, _ = self.gru(torch.cat([e, p], dim=-1))
        return self.head(h[:, -1, :]).squeeze(-1)  # 마지막 스텝 = 현재 이벤트
