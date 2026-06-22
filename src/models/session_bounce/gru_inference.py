"""Session Bounce GRU 실시간 추론 (계획서 §2.2/§3.6).

저장된 artifact(model.pt + model_config.json + feature_schema.json +
category_index_map.json)를 로드해, 최근 이벤트 시퀀스로 session_bounce_probability를 낸다.

이벤트 dict 예시(최근→과거 무관, 시간순 입력):
  {"event_type": "view"|"cart"|"remove_from_cart"|"purchase",
   "category_id": 1487580005134238553, "price": 2.62, "timestamp": <epoch_sec>}
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.models.session_bounce.gru_model import SessionBounceGRU

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "session_bounce" / "gru"
EVENT_TYPES = {"view": 0, "cart": 1, "remove_from_cart": 2, "purchase": 3}


class SessionBouncePredictor:
    def __init__(self, model_dir=MODEL_DIR, device="cpu"):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        self.schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
        cmap = json.loads((model_dir / "category_index_map.json").read_text(encoding="utf-8"))
        self.cat_index = {int(k): int(v) for k, v in cmap["map"].items()}
        self.seq_len = self.schema["seq_len"]
        self.device = torch.device(device)
        self.model = SessionBounceGRU(
            n_categories=self.cfg["n_categories_incl_pad"], n_num=self.cfg["n_num"],
            cat_emb=self.cfg["cat_emb"], num_proj=self.cfg["num_proj"],
            hidden=self.cfg["hidden"], layers=self.cfg["layers"], dropout=self.cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval()

    def _encode(self, events):
        """시간순 이벤트 리스트 -> (1,L,6) x_num, (1,L) x_cat (좌측 padding)."""
        ev = events[-self.seq_len:]
        L = len(ev)
        x_num = np.zeros((self.seq_len, 6), dtype=np.float32)
        x_cat = np.zeros(self.seq_len, dtype=np.int64)
        prev_t = None
        for k, e in enumerate(ev):
            row = self.seq_len - L + k
            et = EVENT_TYPES.get(e.get("event_type"), 0)
            x_num[row, et] = 1.0
            t = e.get("timestamp")
            gap = 0.0 if (prev_t is None or t is None) else max(0.0, float(t) - float(prev_t))
            x_num[row, 4] = np.log1p(gap)
            x_num[row, 5] = np.log1p(max(0.0, float(e.get("price", 0.0))))
            cid = e.get("category_id")
            x_cat[row] = self.cat_index.get(int(cid), 0) if cid is not None else 0
            prev_t = t
        return x_num[None], x_cat[None]

    def predict(self, events):
        x_num, x_cat = self._encode(events)
        with torch.no_grad():
            logit = self.model(
                torch.from_numpy(x_num).to(self.device),
                torch.from_numpy(x_cat).to(self.device),
            )
            prob = torch.sigmoid(logit).item()
        return {"session_bounce_probability": round(float(prob), 4)}


if __name__ == "__main__":
    p = SessionBouncePredictor()
    demo = [
        {"event_type": "view", "category_id": 1487580005134238553, "price": 5.24, "timestamp": 0},
        {"event_type": "cart", "category_id": 1487580005134238553, "price": 5.24, "timestamp": 40},
    ]
    print(p.predict(demo))
