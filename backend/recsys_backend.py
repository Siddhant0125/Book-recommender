"""
Backend utilities for the two-tower mood-based book recommender.

Functions:
 1) load_checkpoint
 2) build_model
 3) load_book_features_and_embeddings
 4) predict_moods
 5) build_user_features_from_sentence
 6) explain_recommendation

Note: Default paths are relative to this file's directory.
"""
from __future__ import annotations

import os
import json
import ast
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path for imports like `models.*`
import sys as _sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import torch
import pandas as pd

# Import model classes from the repo
from models.two_tower_model import TwoTowerModel, TwoTowerConfig

# Cached MoodTagger instance (loaded once)
_MOOD_TAGGER: Any | None = None
try:
    from mood_classifier_tagger.mood_tagger import MoodTagger  # repo path
except Exception:  # pragma: no cover
    MoodTagger = None  # type: ignore


# -----------------------------
# Helper parsing utilities
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
    # Try JSON or Python literal
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("(", "[").replace(")", "]"))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    # Delimiter fallback
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
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
            out.extend([float(n) for n in nums])
    return out


def _pad_2d_long(seqs: List[List[int]], pad_id: int = 0) -> torch.Tensor:
    maxlen = max((len(s) for s in seqs), default=1)
    return torch.tensor([s + [pad_id] * (maxlen - len(s)) for s in seqs], dtype=torch.long)


def _pad_2d_float(seqs: List[List[float]], pad_val: float = 0.0) -> torch.Tensor:
    maxlen = max((len(s) for s in seqs), default=1)
    return torch.tensor([s + [pad_val] * (maxlen - len(s)) for s in seqs], dtype=torch.float)


# -----------------------------
# 1) load_checkpoint
# -----------------------------

def load_checkpoint(
    checkpoint_path: str = "../models/two_tower.pt",
    device: str = "cpu",
) -> Tuple[dict, dict, dict, dict, dict]:
    """Load the checkpoint (relative to this file by default).

    Returns: (state_dict, text_vocab, mood_vocab, genre_vocab, config)
    """
    base_dir = os.path.dirname(__file__)
    ckpt_path = checkpoint_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(base_dir, ckpt_path))
    map_loc = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=map_loc)
    if not isinstance(ckpt, dict):
        return ckpt, {}, {}, {}, {}
    state_dict = ckpt.get("model_state_dict", ckpt)
    text_vocab = ckpt.get("text_vocab", {})
    mood_vocab = ckpt.get("mood_vocab", {})
    genre_vocab = ckpt.get("genre_vocab", {})
    config = ckpt.get("config", {})
    return state_dict, text_vocab, mood_vocab, genre_vocab, config


# -----------------------------
# 2) build_model
# -----------------------------

def build_model(
    text_vocab: Dict[str, int],
    mood_vocab: Dict[str, int],
    genre_vocab: Dict[str, int],
    config: Dict[str, Any],
    device: str = "cpu",
) -> TwoTowerModel:
    """Construct TwoTowerModel using vocab sizes and config fields."""
    def cfg_get(*names: str, default: Any = None):
        for n in names:
            if n in config:
                return config[n]
        return default

    cfg = TwoTowerConfig(
        text_vocab_size=len(text_vocab) if text_vocab is not None else int(cfg_get("text_vocab_size", default=0) or 0),
        mood_vocab_size=len(mood_vocab) if mood_vocab is not None else int(cfg_get("mood_vocab_size", default=0) or 0),
        genre_vocab_size=len(genre_vocab) if genre_vocab is not None else int(cfg_get("genre_vocab_size", default=0) or 0),
        pad_id=int(cfg_get("pad_id", default=0) or 0),
        unk_id=int(cfg_get("unk_id", default=1) or 1),
        text_emb_dim=int(cfg_get("text_emb_dim", default=128) or 128),
        mood_emb_dim=int(cfg_get("mood_emb_dim", default=64) or 64),
        genre_emb_dim=int(cfg_get("genre_emb_dim", default=64) or 64),
        text_lstm_hidden=int(cfg_get("text_lstm_hidden", "lstm_hidden_dim", default=128) or 128),
        text_lstm_layers=int(cfg_get("text_lstm_layers", default=1) or 1),
        text_bidirectional=bool(cfg_get("text_bidirectional", default=True)),
        text_dropout=float(cfg_get("text_dropout", default=0.1) or 0.1),
        tower_dim=int(cfg_get("tower_dim", "embedding_dim", default=128) or 128),
        tower_hidden=int(cfg_get("tower_hidden", default=256) or 256),
        tower_dropout=float(cfg_get("tower_dropout", default=0.1) or 0.1),
        emb_init_std=float(cfg_get("emb_init_std", default=0.02) or 0.02),
    )
    model = TwoTowerModel(cfg).to(torch.device(device))
    return model


# -----------------------------
# 3) load_book_features_and_embeddings
# -----------------------------

def load_book_features_and_embeddings(
    model: TwoTowerModel,
    books_path: str = "../preprocessed/book_features.parquet",
    device: str = "cpu",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Load book features and compute book embeddings in batches.

    Returns:
      books_df: DataFrame in same order as embeddings
      book_vecs: Tensor [N_books, D] on CPU
    """
    base_dir = os.path.dirname(__file__)
    path = books_path
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(base_dir, path))

    # Load DataFrame (CSV or Parquet)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Accept 'text_ids' or legacy 'book_text_ids'
    text_col = "text_ids" if "text_ids" in df.columns else ("book_text_ids" if "book_text_ids" in df.columns else None)
    if text_col is None:
        raise ValueError("Features must include 'text_ids' or 'book_text_ids'.")

    required = [text_col, "book_mood_ids", "book_mood_scores", "book_genre_ids"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    # Parse to lists
    df = df.copy()
    df[text_col] = df[text_col].apply(_parse_int_list)
    df["book_mood_ids"] = df["book_mood_ids"].apply(_parse_int_list)
    df["book_mood_scores"] = df["book_mood_scores"].apply(_parse_float_list)
    df["book_genre_ids"] = df["book_genre_ids"].apply(_parse_int_list)

    # Align mood ids/scores
    def _align_row(row: pd.Series) -> pd.Series:
        mids = row["book_mood_ids"]
        msc = row["book_mood_scores"]
        k = min(len(mids), len(msc))
        row["book_mood_ids"] = mids[:k]
        row["book_mood_scores"] = msc[:k]
        return row

    df = df.apply(_align_row, axis=1)

    # Try to merge description and other metadata from original dataset
    books_enriched_path = os.path.normpath(os.path.join(base_dir, "..", "dataset", "books_enriched.csv"))
    if os.path.exists(books_enriched_path):
        try:
            books_enriched = pd.read_csv(books_enriched_path)
            # Merge on book_id to get description and other metadata
            merge_cols = ["book_id"]
            if "description" in books_enriched.columns:
                merge_cols.append("description")
            if "authors" in books_enriched.columns:
                merge_cols.append("authors")
            if "average_rating" in books_enriched.columns:
                merge_cols.append("average_rating")

            books_enriched_subset = books_enriched[merge_cols].copy()
            df = df.merge(books_enriched_subset, on="book_id", how="left")
        except Exception as e:
            # If merge fails, just proceed with what we have
            pass

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    books_df_parsed = df.copy()

    batch_size = 512
    vecs: List[torch.Tensor] = []
    N = len(books_df_parsed)
    for i in range(0, N, batch_size):
        b = books_df_parsed.iloc[i:i+batch_size]
        T = _pad_2d_long(b[text_col].tolist(), pad_id=model.cfg.pad_id).to(dev)
        M = _pad_2d_long(b["book_mood_ids"].tolist(), pad_id=model.cfg.pad_id).to(dev)
        MS = _pad_2d_float(b["book_mood_scores"].tolist(), pad_val=0.0).to(dev)
        G = _pad_2d_long(b["book_genre_ids"].tolist(), pad_id=model.cfg.pad_id).to(dev)
        with torch.no_grad():
            be = model.book(
                text_ids=T,
                book_mood_ids=M,
                book_mood_scores=MS,
                book_genre_ids=G,
            )
        vecs.append(be.detach().cpu())

    book_vecs = torch.cat(vecs, dim=0) if vecs else torch.zeros((0, model.cfg.tower_dim))
    return books_df_parsed, book_vecs


# -----------------------------
# 4) predict_moods (cached tagger)
# -----------------------------

def predict_moods(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Use a cached/global MoodTagger instance to get top-k moods."""
    global _MOOD_TAGGER
    if MoodTagger is None:
        return []
    if _MOOD_TAGGER is None:
        _MOOD_TAGGER = MoodTagger()
    moods, scores = _MOOD_TAGGER.tag(text or "", top_k=top_k)
    return list(zip(moods, scores))


# -----------------------------
# 5) build_user_features_from_sentence
# -----------------------------

def build_user_features_from_sentence(
    sentence: str,
    mood_vocab: Dict[str, int],
    max_moods: int,
    max_genres: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (1, K)/(1, K)/(1, G) user tensors from free-text sentence."""
    preds = predict_moods(sentence, top_k=max_moods)
    mids: List[int] = []
    mscs: List[float] = []
    for m, s in preds:
        mid = int(mood_vocab.get(m, 0))
        if mid != 0:
            mids.append(mid)
            mscs.append(float(s))
    # Pad/truncate
    mids = (mids + [0] * max_moods)[:max_moods]
    mscs = (mscs + [0.0] * max_moods)[:max_moods]

    dev = torch.device(device)
    user_mood_ids = torch.tensor([mids], dtype=torch.long, device=dev)      # [1, K]
    user_mood_scores = torch.tensor([mscs], dtype=torch.float, device=dev)  # [1, K]
    user_genre_ids = torch.zeros((1, max_genres), dtype=torch.long, device=dev)  # [1, G]
    return user_mood_ids, user_mood_scores, user_genre_ids


# -----------------------------
# 6) explain_recommendation
# -----------------------------

def explain_recommendation(
    user_mood_ids: torch.Tensor,
    user_mood_scores: torch.Tensor,
    user_genre_ids: torch.Tensor,
    book_mood_ids: torch.Tensor,
    book_mood_scores: torch.Tensor,
    book_genre_ids: torch.Tensor,
    id2mood: Dict[int, str],
    id2genre: Dict[int, str],
    top_moods: int = 3,
) -> str:
    """Create a short reason string using overlaps; fallback to book tags if none."""
    def to_ids(x: torch.Tensor) -> List[int]:
        arr = x.detach().cpu().view(-1).tolist()
        return [int(v) for v in arr if int(v) != 0]

    u_m = to_ids(user_mood_ids)
    u_g = to_ids(user_genre_ids)
    b_m = to_ids(book_mood_ids)
    b_g = to_ids(book_genre_ids)

    # Overlaps (preserve book order)
    mood_overlap: List[int] = []
    seen = set()
    for mid in b_m:
        if mid in u_m and mid not in seen:
            seen.add(mid)
            mood_overlap.append(mid)
        if len(mood_overlap) >= top_moods:
            break

    genre_overlap: List[int] = []
    seen_g = set()
    for gid in b_g:
        if gid in u_g and gid not in seen_g:
            seen_g.add(gid)
            genre_overlap.append(gid)

    # Map to labels
    def ids_to_names(ids: List[int], mapping: Dict[int, str]) -> List[str]:
        return [mapping.get(i, str(i)) for i in ids]

    mood_names = ids_to_names(mood_overlap, id2mood)
    genre_names = ids_to_names(genre_overlap, id2genre)

    parts: List[str] = []
    if mood_names:
        if len(mood_names) == 1:
            parts.append(f"it matches your mood: {mood_names[0]}")
        elif len(mood_names) == 2:
            parts.append(f"it matches your moods: {mood_names[0]} and {mood_names[1]}")
        else:
            parts.append("it matches your moods: " + ", ".join(mood_names[:-1]) + f", and {mood_names[-1]}")
    if genre_names:
        if len(genre_names) == 1:
            parts.append(f"it’s in genre: {genre_names[0]}")
        elif len(genre_names) > 1:
            parts.append("it’s in genres: " + ", ".join(genre_names[:-1]) + f", and {genre_names[-1]}")

    # Fallback: no overlaps -> use book's top moods/genres
    if not parts:
        book_mood_names = ids_to_names(b_m[:top_moods], id2mood)
        book_genre_names = ids_to_names(b_g[:max(1, top_moods)], id2genre)
        subparts: List[str] = []
        if book_mood_names:
            if len(book_mood_names) == 1:
                subparts.append(f"for moods like {book_mood_names[0]}")
            else:
                subparts.append("for moods like " + ", ".join(book_mood_names[:-1]) + f", and {book_mood_names[-1]}")
        if book_genre_names:
            if len(book_genre_names) == 1:
                subparts.append(f"in {book_genre_names[0]}")
            else:
                subparts.append("in genres " + ", ".join(book_genre_names[:-1]) + f", and {book_genre_names[-1]}")
        if subparts:
            return "Recommended " + " and ".join(subparts) + "."
        return "Recommended based on overall similarity to your moods and genres."

    return "Recommended because " + ", and ".join(parts) + "."
