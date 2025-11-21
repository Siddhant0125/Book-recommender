import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from two_tower_model import TwoTowerConfig, TwoTowerModel

# -----------------------------
# Helpers to parse list-like cells from CSV
# -----------------------------

def _ensure_list_cell(cell: Any) -> List[Any]:
    if cell is None:
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, (tuple, set)):
        return list(cell)
    s = str(cell).strip()
    if not s:
        return []
    # Try JSON
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("(", "[").replace(")", "]"))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    # Fallback split by common separators
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _parse_int_list(cell: Any) -> List[int]:
    vals = _ensure_list_cell(cell)
    out: List[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            # If string has numbers, extract digits
            import re
            nums = re.findall(r"-?\d+", str(v))
            out.extend([int(n) for n in nums])
    return out


def _parse_float_list(cell: Any) -> List[float]:
    vals = _ensure_list_cell(cell)
    out: List[float] = []
    import re
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            # Extract any numeric substrings
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
            out.extend([float(n) for n in nums])
    return out


# -----------------------------
# Loading artifacts
# -----------------------------

def load_vocabs(vocabs_path: str) -> Dict[str, Any]:
    with open(vocabs_path, "r", encoding="utf-8") as f:
        voc = json.load(f)
    return voc


def load_book_features(pre_dir: str) -> pd.DataFrame:
    # Prefer parquet if present
    pq = os.path.join(pre_dir, "book_features.parquet")
    csv = os.path.join(pre_dir, "book_features.csv")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"No book features found under {pre_dir}")
    return df


def load_ratings(data_dir: str) -> pd.DataFrame:
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}")
    return pd.read_csv(ratings_path)


# -----------------------------
# Build user->positives and in-memory feature dict
# -----------------------------

def build_user_pos(ratings: pd.DataFrame, min_positive: int = 4) -> Dict[int, List[int]]:
    pos = ratings[ratings.get("rating", pd.Series([0]*len(ratings))) >= min_positive]
    user_pos: Dict[int, List[int]] = defaultdict(list)
    for row in pos.itertuples(index=False):
        try:
            uid = int(getattr(row, "user_id"))
            bid = int(getattr(row, "book_id"))
        except Exception:
            # Skip malformed
            continue
        user_pos[uid].append(bid)
    # Deduplicate while preserving order
    for u in list(user_pos.keys()):
        seen = set()
        seq = []
        for b in user_pos[u]:
            if b not in seen:
                seen.add(b)
                seq.append(b)
        user_pos[u] = seq
    return user_pos


def build_book_feature_map(book_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    feat: Dict[int, Dict[str, Any]] = {}
    for row in book_df.itertuples(index=False):
        try:
            bid = int(getattr(row, "book_id"))
        except Exception:
            continue
        t_ids = _parse_int_list(getattr(row, "book_text_ids", []))
        m_ids = _parse_int_list(getattr(row, "book_mood_ids", []))
        m_sc = _parse_float_list(getattr(row, "book_mood_scores", []))
        g_ids = _parse_int_list(getattr(row, "book_genre_ids", []))
        # Align lengths
        K = max(len(m_ids), len(m_sc))
        if K == 0:
            m_ids, m_sc = [0], [0.0]
        else:
            if len(m_ids) < K:
                m_ids = m_ids + [0] * (K - len(m_ids))
            if len(m_sc) < K:
                m_sc = m_sc + [0.0] * (K - len(m_sc))
            # Normalize mood scores to sum=1 over non-pad ids
            ssum = sum(s for i, s in zip(m_ids, m_sc) if i != 0)
            if ssum > 0:
                m_sc = [ (s/ssum if i != 0 else 0.0) for i, s in zip(m_ids, m_sc) ]
        feat[bid] = {
            "text_ids": t_ids,
            "mood_ids": m_ids,
            "mood_scores": m_sc,
            "genre_ids": g_ids,
        }
    return feat


# -----------------------------
# User aggregation from positive books (exclude target when needed)
# -----------------------------

def aggregate_user_features(book_ids: List[int], book_feat: Dict[int, Dict[str, Any]]) -> Tuple[List[int], List[float], List[int]]:
    mood_weight: Counter = Counter()
    genre_count: Counter = Counter()
    for bid in book_ids:
        bf = book_feat.get(bid)
        if not bf:
            continue
        mids = bf.get("mood_ids", [])
        msc = bf.get("mood_scores", [])
        for i, s in zip(mids, msc):
            if i != 0:
                mood_weight[i] += float(s)
        for gid in bf.get("genre_ids", []):
            if gid != 0:
                genre_count[gid] += 1.0
    # Moods: normalized weights
    if mood_weight:
        mids, wts = zip(*mood_weight.most_common())
        mids = list(mids)
        wts = list(wts)
        ssum = float(sum(wts))
        wts = [w/ssum for w in wts] if ssum > 0 else [1.0/len(wts)] * len(wts)
    else:
        mids, wts = [0], [0.0]
    # Genres: take all seen (order by count desc)
    gids = [g for g, _ in genre_count.most_common()] if genre_count else [0]
    return list(mids), list(wts), list(gids)


# -----------------------------
# Dataset and collate
# -----------------------------

class PairDataset(Dataset):
    def __init__(self, user_pos: Dict[int, List[int]], book_feat: Dict[int, Dict[str, Any]], exclude_target: bool = True) -> None:
        self.user_pos = {u: list(bs) for u, bs in user_pos.items() if len(bs) > 0}
        self.users = sorted(self.user_pos.keys())
        self.book_feat = book_feat
        self.exclude_target = exclude_target
        # Build (u, b) pairs
        self.pairs: List[Tuple[int, int]] = []
        for u, bs in self.user_pos.items():
            for b in bs:
                self.pairs.append((u, b))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        u, pos_b = self.pairs[idx]
        src_books = [b for b in self.user_pos[u] if (not self.exclude_target) or (b != pos_b)]
        if not src_books:
            src_books = [pos_b]
        um_ids, um_sc, ug_ids = aggregate_user_features(src_books, self.book_feat)
        bf = self.book_feat[pos_b]
        return {
            "user_mood_ids": um_ids,
            "user_mood_scores": um_sc,
            "user_genre_ids": ug_ids,
            "text_ids": bf["text_ids"],
            "book_mood_ids": bf["mood_ids"],
            "book_mood_scores": bf["mood_scores"],
            "book_genre_ids": bf["genre_ids"],
        }


def _pad_2d_long(seqs: List[List[int]], pad_id: int = 0) -> torch.LongTensor:
    maxlen = max((len(s) for s in seqs), default=1)
    return torch.tensor([s + [pad_id] * (maxlen - len(s)) for s in seqs], dtype=torch.long)


def _pad_2d_float(seqs: List[List[float]], pad_val: float = 0.0) -> torch.FloatTensor:
    maxlen = max((len(s) for s in seqs), default=1)
    return torch.tensor([s + [pad_val] * (maxlen - len(s)) for s in seqs], dtype=torch.float)


def pair_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "user_mood_ids": _pad_2d_long([b["user_mood_ids"] for b in batch]),
        "user_mood_scores": _pad_2d_float([b["user_mood_scores"] for b in batch]),
        "user_genre_ids": _pad_2d_long([b["user_genre_ids"] for b in batch]),
        "text_ids": _pad_2d_long([b["text_ids"] for b in batch]),
        "book_mood_ids": _pad_2d_long([b["book_mood_ids"] for b in batch]),
        "book_mood_scores": _pad_2d_float([b["book_mood_scores"] for b in batch]),
        "book_genre_ids": _pad_2d_long([b["book_genre_ids"] for b in batch]),
    }


# -----------------------------
# Training and evaluation
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: TwoTowerModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, temperature: float, use_amp: bool = False, scaler: torch.cuda.amp.GradScaler | None = None) -> float:
    model.train()
    total, steps = 0.0, 0
    amp_enabled = (device.type == "cuda" and use_amp)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                u_vec, b_vec = model(
                    user_mood_ids=batch["user_mood_ids"],
                    user_mood_scores=batch["user_mood_scores"],
                    user_genre_ids=batch["user_genre_ids"],
                    text_ids=batch["text_ids"],
                    book_mood_ids=batch["book_mood_ids"],
                    book_mood_scores=batch["book_mood_scores"],
                    book_genre_ids=batch["book_genre_ids"],
                )
                loss = TwoTowerModel.contrastive_loss(u_vec, b_vec, temperature=temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            u_vec, b_vec = model(
                user_mood_ids=batch["user_mood_ids"],
                user_mood_scores=batch["user_mood_scores"],
                user_genre_ids=batch["user_genre_ids"],
                text_ids=batch["text_ids"],
                book_mood_ids=batch["book_mood_ids"],
                book_mood_scores=batch["book_mood_scores"],
                book_genre_ids=batch["book_genre_ids"],
            )
            loss = TwoTowerModel.contrastive_loss(u_vec, b_vec, temperature=temperature)
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def evaluate_loo(
    model: TwoTowerModel,
    user_pos: Dict[int, List[int]],
    book_feat: Dict[int, Dict[str, Any]],
    device: torch.device,
    k: int = 10,
    negatives: int = 100,
) -> float:
    model.eval()
    users = [u for u, bs in user_pos.items() if len(bs) >= 2]
    if not users:
        return 0.0
    all_books = list(book_feat.keys())
    hits, total = 0, 0
    with torch.no_grad():
        for u in users:
            pos = user_pos[u]
            test_b = pos[-1]
            train_bs = pos[:-1]
            if not train_bs:
                continue
            um_ids, um_sc, ug_ids = aggregate_user_features(train_bs, book_feat)
            # Build candidate set: test + sampled negatives
            exclude = set(pos)
            pool = [b for b in all_books if b not in exclude]
            if len(pool) <= (k - 1):
                negs = pool
            else:
                negs = random.sample(pool, min(negatives, len(pool)))
            candidates = [test_b] + negs
            # Build tensors
            U_m = _pad_2d_long([um_ids] * len(candidates)).to(device)
            U_ms = _pad_2d_float([um_sc] * len(candidates)).to(device)
            U_g = _pad_2d_long([ug_ids] * len(candidates)).to(device)
            B_t = _pad_2d_long([book_feat[b]["text_ids"] for b in candidates]).to(device)
            B_m = _pad_2d_long([book_feat[b]["mood_ids"] for b in candidates]).to(device)
            B_ms = _pad_2d_float([book_feat[b]["mood_scores"] for b in candidates]).to(device)
            B_g = _pad_2d_long([book_feat[b]["genre_ids"] for b in candidates]).to(device)
            u_vecs, b_vecs = model(
                user_mood_ids=U_m,
                user_mood_scores=U_ms,
                user_genre_ids=U_g,
                text_ids=B_t,
                book_mood_ids=B_m,
                book_mood_scores=B_ms,
                book_genre_ids=B_g,
            )
            scores = torch.sum(u_vecs * b_vecs, dim=-1)  # cosine since normalized
            rank = torch.argsort(scores, descending=True)
            topk = rank[:k].tolist()
            if 0 in topk:
                hits += 1
            total += 1
    return hits / max(total, 1)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Tower recommender (mood-based)")
    parser.add_argument("--data-dir", default=os.path.join("dataset"))
    parser.add_argument("--preprocessed-dir", default=os.path.join("preprocessed"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=os.path.join("artifacts"))
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (set 2-4 on Colab)")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision (AMP) when on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.set_defaults(amp=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If amp not specified, default to True on CUDA
    use_amp = args.amp if args.amp is not None else (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))
    print(f"Device: {device.type} | AMP: {use_amp}")

    # Load artifacts
    vocabs_path = os.path.join(args.preprocessed_dir, "vocabs.json")
    vocabs = load_vocabs(vocabs_path)
    book_df = load_book_features(args.preprocessed_dir)
    ratings = load_ratings(args.data_dir)

    # Build in-memory structures
    user_pos = build_user_pos(ratings, min_positive=4)
    book_feat = build_book_feature_map(book_df)

    # Filter users to those with at least 1 positive for training
    train_user_pos = {u: bs for u, bs in user_pos.items() if len(bs) >= 1}
    if not train_user_pos:
        raise RuntimeError("No users with positives found (rating >= 4)")

    # Model config from vocabs
    text_vocab_size = len(vocabs.get("text_stoi", {}))
    mood_vocab_size = len(vocabs.get("mood_stoi", {}))
    genre_vocab_size = len(vocabs.get("genre_stoi", {}))
    if min(text_vocab_size, mood_vocab_size, genre_vocab_size) <= 0:
        raise RuntimeError("Vocab sizes must be > 0. Check preprocessed/vocabs.json")

    cfg = TwoTowerConfig(
        text_vocab_size=text_vocab_size,
        mood_vocab_size=mood_vocab_size,
        genre_vocab_size=genre_vocab_size,
        pad_id=0,
        unk_id=1,
        # You can tune these
        text_emb_dim=64,
        mood_emb_dim=32,
        genre_emb_dim=32,
        text_lstm_hidden=128,
        text_lstm_layers=1,
        text_bidirectional=True,
        tower_dim=64,
        tower_hidden=128,
        text_dropout=0.1,
        tower_dropout=0.1,
    )

    model = TwoTowerModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DataLoader (exclude target positive to avoid leakage when forming user features)
    train_ds = PairDataset(train_user_pos, book_feat, exclude_target=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pair_collate,
    )

    # Train
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, device, args.temperature, use_amp=use_amp, scaler=scaler)
        print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}")

    # Evaluate leave-one-out HR@10
    hr10 = evaluate_loo(model, user_pos, book_feat, device, k=10, negatives=100)
    print(f"Leave-One-Out HitRate@10: {hr10:.4f}")

    # Save model and vocabs
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "two_tower.pt")
    # Save CPU-mapped state_dict for portability
    state_dict_cpu = {k: v.to("cpu") for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": state_dict_cpu,
        # Save as *_vocab (stoi) for convenience
        "text_vocab": vocabs.get("text_stoi", {}),
        "mood_vocab": vocabs.get("mood_stoi", {}),
        "genre_vocab": vocabs.get("genre_stoi", {}),
        "config": cfg.__dict__,
    }, save_path)
    print(f"Saved checkpoint: {save_path}")

    # Example: load for inference later
    ckpt = torch.load(save_path, map_location="cpu")
    re_cfg_dict = ckpt["config"]
    re_cfg = TwoTowerConfig(**re_cfg_dict)
    reload_model = TwoTowerModel(re_cfg)
    reload_model.load_state_dict(ckpt["model_state_dict"])  # fixed: removed stray parentheses
    reload_model.eval()
    print("Reloaded model ready for inference.")


if __name__ == "__main__":
    main()
