import os
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any

import pandas as pd
import numpy as np

# -----------------------------
# Config dataclass
# -----------------------------

@dataclass
class PreprocConfig:
    books_path: str
    ratings_path: str
    # text vocab
    max_vocab_size: int = 50000
    min_token_freq: int = 1
    max_text_len: int = 256
    # per-book caps
    max_moods_per_book: int = 3
    max_genres_per_book: int = 8
    # per-user caps
    min_positive_rating: int = 4
    max_user_moods: int = 16
    max_user_genres: int = 16
    # column names
    col_book_id: str = "book_id"
    col_title: str = "title"
    col_authors: str = "authors"
    col_desc: str = "description"
    col_moods: str = "moods_top3"
    col_mood_scores: str = "moods_scores"
    col_genres: str = "genres"
    col_user_id: str = "user_id"
    col_rating: str = "rating"


# -----------------------------
# Utility helpers
# -----------------------------

def _ensure_list_cell(cell: Any) -> List[Any]:
    """Best-effort parse of list-like strings into Python lists.
    Accepts: actual lists, JSON-like strings, Python repr lists, or pipe/comma separated strings.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, (tuple, set)):
        return list(cell)

    s = str(cell).strip()
    if not s:
        return []

    # Try JSON first
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

    # Fallback: split by common separators
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return parts
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    # Single token
    return [s]


def _tokenize_simple(text: str) -> List[str]:
    """Very simple whitespace tokenizer with lowercasing. Keeps punctuation as tokens if attached.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    return str(text).lower().split()


def _build_vocab_from_texts(texts: Iterable[str], min_freq: int = 1, max_size: Optional[int] = None,
                            specials: Optional[List[str]] = None) -> Tuple[Dict[str, int], List[str]]:
    counter = Counter()
    for t in texts:
        tokens = _tokenize_simple(t)
        counter.update(tokens)

    # Sort by freq desc, then lexicographically for stability
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    vocab_tokens = []
    if specials:
        vocab_tokens.extend(specials)
    for tok, freq in items:
        if freq < min_freq:
            continue
        if max_size is not None and len(vocab_tokens) >= max_size:
            break
        if specials and tok in specials:
            continue
        vocab_tokens.append(tok)

    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    return stoi, vocab_tokens


def _build_vocab_from_lists(seqs: Iterable[List[str]], specials: Optional[List[str]] = None) -> Tuple[Dict[str, int], List[str]]:
    counter = Counter()
    for lst in seqs:
        counter.update([str(x).strip().lower() for x in lst if str(x).strip()])
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    vocab_tokens = []
    if specials:
        vocab_tokens.extend(specials)
    for tok, _ in items:
        if specials and tok in specials:
            continue
        vocab_tokens.append(tok)
    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    return stoi, vocab_tokens


def _ids_for_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int,
                    pad_id: int = 0, unk_id: int = 1) -> List[int]:
    ids = [vocab.get(t, unk_id) for t in tokens]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def _ids_for_list(values: List[str], vocab: Dict[str, int], max_len: int,
                  pad_id: int = 0, unk_id: int = 1) -> List[int]:
    ids = [vocab.get(str(v).strip().lower(), unk_id) for v in values if str(v).strip()]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def _pad_or_truncate_scores(scores: List[float], max_len: int) -> List[float]:
    if len(scores) >= max_len:
        return scores[:max_len]
    return scores + [0.0] * (max_len - len(scores))


def _parse_float_list(cell: Any) -> List[float]:
    """Parse a cell into a list of floats robustly.
    - Accepts lists/tuples of numbers/strings
    - Accepts strings containing numbers, including wrappers like 'np.float64(0.12)'
      or messy brackets; extracts all numeric substrings.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    # If it's already a list/tuple, coerce elementwise
    if isinstance(cell, (list, tuple)):
        out: List[float] = []
        for x in cell:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            try:
                # Handle numpy scalar
                if isinstance(x, (int, float, np.floating)):
                    out.append(float(x))
                else:
                    out.append(float(str(x).strip()))
            except Exception:
                # As a fallback, extract any numeric substrings from representation
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
                out.extend([float(n) for n in nums])
        return out
    # If it's a string, extract numbers via regex
    s = str(cell)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return [float(n) for n in nums]


# -----------------------------
# Core preprocessing
# -----------------------------

def load_dataframes(cfg: PreprocConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(cfg.books_path)
    ratings = pd.read_csv(cfg.ratings_path)
    # Normalize column names to expected
    books = books.rename(columns={
        cfg.col_book_id: "book_id",
        cfg.col_title: "title",
        cfg.col_authors: "authors",
        cfg.col_desc: "description",
        cfg.col_moods: "moods_top3",
        cfg.col_mood_scores: "moods_scores",
        cfg.col_genres: "genres",
    })
    ratings = ratings.rename(columns={
        cfg.col_user_id: "user_id",
        cfg.col_book_id: "book_id",
        cfg.col_rating: "rating",
    })
    return books, ratings


def build_vocabularies(books: pd.DataFrame, cfg: PreprocConfig) -> Dict[str, Any]:
    # Prepare fused text: [TITLE] t [AUTHOR] a [DESC] d
    title = books.get("title", pd.Series([""] * len(books)))
    authors = books.get("authors", pd.Series([""] * len(books)))
    desc = books.get("description", pd.Series([""] * len(books)))
    fused_texts = [f"[TITLE] {t or ''} [AUTHOR] {a or ''} [DESC] {d or ''}" for t, a, d in zip(title, authors, desc)]

    # specials for text vocab
    text_specials = ["[PAD]", "[UNK]", "[TITLE]", "[AUTHOR]", "[DESC]"]
    text_stoi, text_vocab = _build_vocab_from_texts(
        fused_texts, min_freq=cfg.min_token_freq, max_size=cfg.max_vocab_size, specials=text_specials
    )

    # moods/genres
    mood_lists = [
        _ensure_list_cell(v) for v in books.get("moods_top3", pd.Series([[]] * len(books)))
    ]
    genre_lists = [
        _ensure_list_cell(v) for v in books.get("genres", pd.Series([[]] * len(books)))
    ]

    mood_specials = ["[PAD]", "[UNK]"]
    genre_specials = ["[PAD]", "[UNK]"]
    mood_stoi, mood_vocab = _build_vocab_from_lists(mood_lists, specials=mood_specials)
    genre_stoi, genre_vocab = _build_vocab_from_lists(genre_lists, specials=genre_specials)

    vocabs = {
        "text_stoi": text_stoi,
        "text_itos": text_vocab,
        "mood_stoi": mood_stoi,
        "mood_itos": mood_vocab,
        "genre_stoi": genre_stoi,
        "genre_itos": genre_vocab,
        "pad_id": 0,
        "unk_id": 1,
    }
    return vocabs


def encode_books(books: pd.DataFrame, vocabs: Dict[str, Any], cfg: PreprocConfig) -> pd.DataFrame:
    pad_id, unk_id = vocabs["pad_id"], vocabs["unk_id"]
    text_stoi = vocabs["text_stoi"]
    mood_stoi = vocabs["mood_stoi"]
    genre_stoi = vocabs["genre_stoi"]

    title = books.get("title", pd.Series([""] * len(books)))
    authors = books.get("authors", pd.Series([""] * len(books)))
    desc = books.get("description", pd.Series([""] * len(books)))

    fused_texts = [f"[TITLE] {t or ''} [AUTHOR] {a or ''} [DESC] {d or ''}" for t, a, d in zip(title, authors, desc)]

    # Encode text
    book_text_ids = [
        _ids_for_tokens(_tokenize_simple(ft), text_stoi, cfg.max_text_len, pad_id, unk_id)
        for ft in fused_texts
    ]

    # Encode moods (ids and scores)
    raw_moods = [
        [str(x).strip().lower() for x in _ensure_list_cell(v)]
        for v in books.get("moods_top3", pd.Series([[]] * len(books)))
    ]
    raw_scores = [
        _parse_float_list(v)
        for v in books.get("moods_scores", pd.Series([[]] * len(books)))
    ]

    book_mood_ids: List[List[int]] = []
    book_mood_scores: List[List[float]] = []
    K = cfg.max_moods_per_book
    for moods, scores in zip(raw_moods, raw_scores):
        # Normalize input length/quality
        if not moods:
            moods = []
        if not scores or len(scores) != len(moods):
            # fallback to uniform scores
            scores = [1.0] * len(moods)
        # Normalize per-book scores to sum=1 (if possible)
        ssum = float(sum(scores))
        if ssum > 0:
            scores = [s / ssum for s in scores]
        # Truncate/pad
        m_ids = _ids_for_list(moods[:K], mood_stoi, K, pad_id, unk_id)
        m_s = _pad_or_truncate_scores(scores[:K], K)
        book_mood_ids.append(m_ids)
        book_mood_scores.append(m_s)

    # Encode genres
    raw_genres = [
        [str(x).strip().lower() for x in _ensure_list_cell(v)]
        for v in books.get("genres", pd.Series([[]] * len(books)))
    ]
    G = cfg.max_genres_per_book
    book_genre_ids = [
        _ids_for_list(g[:G], genre_stoi, G, pad_id, unk_id) for g in raw_genres
    ]

    out = pd.DataFrame({
        "book_id": books.get("book_id"),
        "book_text_ids": book_text_ids,
        "book_mood_ids": book_mood_ids,
        "book_mood_scores": book_mood_scores,
        "book_genre_ids": book_genre_ids,
    })

    # Keep optional reference columns for debugging
    if "title" in books.columns:
        out["title"] = books["title"]
    return out


def aggregate_users(ratings: pd.DataFrame, books: pd.DataFrame, vocabs: Dict[str, Any], cfg: PreprocConfig) -> pd.DataFrame:
    pad_id, unk_id = vocabs["pad_id"], vocabs["unk_id"]
    mood_stoi = vocabs["mood_stoi"]
    genre_stoi = vocabs["genre_stoi"]

    # Only positives
    pos = ratings[ratings["rating"] >= cfg.min_positive_rating].copy()
    if pos.empty:
        # Return empty frame with expected columns
        return pd.DataFrame(columns=["user_id", "user_mood_ids", "user_mood_scores", "user_genre_ids"]).astype({
            "user_id": ratings["user_id"].dtype if "user_id" in ratings.columns else np.int64
        })

    # Join with needed book fields
    join_cols = ["book_id", "moods_top3", "moods_scores", "genres"]
    bcols = [c for c in join_cols if c in books.columns]
    # De-duplicate and ensure key is present
    bsel = list(dict.fromkeys(bcols + ["book_id"]))
    joined = pos.merge(books[bsel], on="book_id", how="left")

    # Aggregate per user
    user_mood_weight: Dict[Any, Counter] = defaultdict(Counter)
    user_genre_count: Dict[Any, Counter] = defaultdict(Counter)

    for row in joined.itertuples(index=False):
        uid = getattr(row, "user_id")
        moods = [str(x).strip().lower() for x in _ensure_list_cell(getattr(row, "moods_top3", []))]
        scores = _parse_float_list(getattr(row, "moods_scores", []))
        if len(scores) != len(moods):
            scores = [1.0] * len(moods)
        ssum = float(sum(scores))
        if ssum > 0:
            scores = [s / ssum for s in scores]
        for m, s in zip(moods, scores):
            if m:
                user_mood_weight[uid][m] += float(s)
        genres = [str(x).strip().lower() for x in _ensure_list_cell(getattr(row, "genres", []))]
        for g in genres:
            if g:
                user_genre_count[uid][g] += 1.0

    # Build rows
    rows = []
    for uid in sorted(set(joined["user_id"].tolist())):
        # Moods: take top-K by weight, normalize to sum=1
        mood_ctr = user_mood_weight.get(uid, Counter())
        if mood_ctr:
            mood_items = mood_ctr.most_common(cfg.max_user_moods)
            moods, weights = zip(*mood_items)
            weights = list(weights)
            wsum = float(sum(weights))
            weights = [w / wsum for w in weights] if wsum > 0 else [1.0 / len(weights)] * len(weights)
        else:
            moods, weights = ["[UNK]"], [1.0]
        mood_ids = _ids_for_list(list(moods), mood_stoi, cfg.max_user_moods, pad_id, unk_id)
        mood_scores = _pad_or_truncate_scores(list(weights), cfg.max_user_moods)

        # Genres: take top-K by count
        genre_ctr = user_genre_count.get(uid, Counter())
        if genre_ctr:
            g_items = [k for k, _ in genre_ctr.most_common(cfg.max_user_genres)]
        else:
            g_items = ["[UNK]"]
        genre_ids = _ids_for_list(list(g_items), genre_stoi, cfg.max_user_genres, pad_id, unk_id)

        rows.append({
            "user_id": uid,
            "user_mood_ids": mood_ids,
            "user_mood_scores": mood_scores,
            "user_genre_ids": genre_ids,
        })

    return pd.DataFrame(rows)


def preprocess(cfg: PreprocConfig) -> Dict[str, Any]:
    books, ratings = load_dataframes(cfg)
    vocabs = build_vocabularies(books, cfg)
    book_features = encode_books(books, vocabs, cfg)
    user_features = aggregate_users(ratings, books, vocabs, cfg)
    return {
        "book_features": book_features,
        "user_features": user_features,
        "vocabs": vocabs,
    }


# -----------------------------
# CLI (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for two-tower recommender")
    parser.add_argument("--data-dir", default=os.path.join("../dataset"))
    parser.add_argument("--books", default="books_with_moods.csv")
    parser.add_argument("--ratings", default="ratings.csv")
    parser.add_argument("--out-dir", default=os.path.join("../preprocessed"))
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--max-moods-per-book", type=int, default=3)
    parser.add_argument("--max-genres-per-book", type=int, default=8)
    parser.add_argument("--min-positive-rating", type=int, default=4)
    parser.add_argument("--max-user-moods", type=int, default=16)
    parser.add_argument("--max-user-genres", type=int, default=16)

    args = parser.parse_args()

    cfg = PreprocConfig(
        books_path=os.path.join(args.data_dir, args.books),
        ratings_path=os.path.join(args.data_dir, args.ratings),
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
        max_text_len=args.max_text_len,
        max_moods_per_book=args.max_moods_per_book,
        max_genres_per_book=args.max_genres_per_book,
        min_positive_rating=args.min_positive_rating,
        max_user_moods=args.max_user_moods,
        max_user_genres=args.max_user_genres,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    artifacts = preprocess(cfg)
    bf = artifacts["book_features"]
    uf = artifacts["user_features"]
    voc = artifacts["vocabs"]

    # Save as parquet if available, else CSV
    try:
        bf.to_parquet(os.path.join(args.out_dir, "book_features.parquet"), index=False)
        uf.to_parquet(os.path.join(args.out_dir, "user_features.parquet"), index=False)
    except Exception:
        bf.to_csv(os.path.join(args.out_dir, "book_features.csv"), index=False)
        uf.to_csv(os.path.join(args.out_dir, "user_features.csv"), index=False)

    with open(os.path.join(args.out_dir, "vocabs.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in voc.items()}, f)

    print("Saved book_features rows:", len(bf))
    print("Saved user_features rows:", len(uf))
    print("Vocab sizes → text:", len(voc["text_stoi"]), "moods:", len(voc["mood_stoi"]), "genres:", len(voc["genre_stoi"]))
