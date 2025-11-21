# Mood-based Book Recommender — Process Flow

Date: 2025-11-13

This document explains the end-to-end pipeline for the mood-based book recommender in this repository: from mood tagging on enriched books to preprocessing, training the two-tower model, and serving interactive recommendations with Streamlit.

---

## 1) Architecture at a glance

- Objective: Recommend books based on a user’s mood sentence by matching a user embedding to precomputed book embeddings.
- Model: Two-Tower architecture (PyTorch)
  - BookTower(text, book moods + scores, book genres) → L2-normalized vector
  - UserTower(user moods + scores, user genres) → L2-normalized vector
  - Similarity: Dot product (cosine since vectors are normalized)
- Moods: Detected via a hybrid NLP approach (Sentence-BERT + Zero-shot BART-MNLI) or a fallback heuristic.

Key components
- `mood_classifier_tagger/` — Detects moods for books (offline) or for sentences (online) if installed.
- `preprocessing/two_tower_preprocessing.py` — Builds vocabularies and encodes books/users into fixed-size ID lists.
- `models/two_tower_model.py` — Model definitions for BookTower, UserTower, TwoTowerModel.
- `models/train_two_tower.py` — Training loop with InfoNCE contrastive loss.
- `test_sentence.py` — Streamlit app: loads checkpoint, precomputes book vectors, recommends top-k for a mood sentence.

---

## 2) End-to-end pipeline

### 0) Environment setup

- Python dependencies are listed in `requirements.txt`. PyTorch is intentionally omitted; install a CPU or GPU wheel from the official index as appropriate for your machine.
- The first mood tagging run will download transformer models (BART MNLI and SBERT), which may take time and disk space.

Windows (cmd.exe) quick start (example)
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 1) Mood tagging: books_enriched → books_with_moods

- Input: `dataset/books_enriched.csv`
  - Expected columns: at least `book_id`, `title`, `description`; `genres` if available.
- Output: `dataset/books_with_moods.csv`
  - Adds columns (example): `mood_primary`, `mood_primary_score`, `moods_top3`, `moods_scores`
- How: `mood_classifier_tagger/mood_tagger.py` uses `MoodClassifier` (hybrid: SBERT + zero-shot). If `transformers`/`sentence-transformers` are not installed, consider installing them for best results.

Run (Windows cmd.exe)
```
python .\mood_classifier_tagger\mood_tagger.py --csv dataset\books_enriched.csv --out dataset\books_with_moods.csv --top-k 3
```

Tips
- You can set `--limit N` to tag only the first N rows for a quick trial.
- `MOOD_TEMPLATES` (in `mood_tagger.py`) can be tuned to your domain.

### 2) Preprocessing: build vocabularies and encode features

- Script: `preprocessing/two_tower_preprocessing.py`
- Produces (under `preprocessed/`):
  - `vocabs.json` — text/mood/genre vocabularies and special IDs {pad_id, unk_id}
  - `book_features.csv` (or parquet) — fixed-length features per book
  - `user_features.csv` (optional) — aggregated user features from positive ratings

Key default caps (can be adjusted in `PreprocConfig`)
- `max_text_len = 256`
- `max_moods_per_book = 3`
- `max_genres_per_book = 8`
- `max_user_moods = 16`
- `max_user_genres = 16`

Outputs — `book_features.csv` schema
- `book_id: int`
- `book_text_ids: list[int]` (length = `max_text_len`; padded with `pad_id`)
- `book_mood_ids: list[int]` (length = `max_moods_per_book`; mood vocab IDs)
- `book_mood_scores: list[float]` (same length; normalized to sum 1 over non-pad)
- `book_genre_ids: list[int]` (length = `max_genres_per_book`; genre vocab IDs)
- Optional: `title`, may also include `description` depending on your pipeline

Notes
- Robust list parsing handles JSON-like strings, Python repr lists, and comma/pipe-separated values.
- Score vectors are normalized per book and padded to fixed length.

### 3) Training: Two-Tower model

- Script: `models/train_two_tower.py`
- Inputs: `preprocessed/vocabs.json`, `preprocessed/book_features.*`, `dataset/ratings.csv`
- Workflow:
  1. Build user→positive books from `ratings.csv` (rating ≥ threshold)
  2. Aggregate user moods (weighted by scores) and genres (count) from positives
  3. Pad sequences to fixed lengths; build batches
  4. Train Two-Tower with InfoNCE contrastive loss; optional AMP on CUDA
  5. Save a checkpoint with weights + vocabs + config

Run (Windows cmd.exe)
```
python .\models\train_two_tower.py --data-dir dataset --preprocessed-dir preprocessed
```

Expected checkpoint contents (`models/two_tower.pt`)
- `model_state_dict`: learned weights
- `text_vocab`, `mood_vocab`, `genre_vocab`: dicts mapping token→id strings
- `config`: hyperparameters (e.g., `text_emb_dim`, `text_lstm_hidden`, `mood_emb_dim`, `genre_emb_dim`, `tower_dim`, `tower_hidden`, `pad_id`, `unk_id`)

### 4) Inference & interactive app: Streamlit

- Script: `test_sentence.py`
- What it does:
  1. Loads checkpoint: rebuilds `TwoTowerModel` from `config` and vocab sizes; loads weights
  2. Loads `preprocessed/book_features.*`; merges `title`/`description` from `dataset/books_enriched.csv` if missing
  3. Converts list-like columns to tensors; runs `BookTower` in batches to precompute all book vectors
  4. Takes a user mood sentence, predicts moods (hybrid classifier if installed, else fallback heuristic)
  5. Builds user features: pad/truncate moods to `max_user_moods`, zero-pad genres to `max_user_genres`
  6. Computes user vector via `UserTower`, scores all books by dot product, selects top‑k
  7. Displays recommendations with a short “Why this book?” explanation from overlapping mood/genre IDs

Run (Windows cmd.exe)
```
streamlit run test_sentence.py
```

Tips
- The sidebar lets you change paths to the checkpoint and preprocessed data, and adjust the embedding batch size.
- The app tolerates minor state_dict mismatches and will warn if there are missing/unexpected keys.

---

## 3) Data contracts (inputs/outputs)

Books (enriched)
- `dataset/books_enriched.csv` should include: `book_id`, `title`, `description` (and preferably `genres`).

Books (with moods)
- `dataset/books_with_moods.csv` (from tagging) typically adds:
  - `mood_primary: str`
  - `mood_primary_score: float`
  - `moods_top3: list[str]`
  - `moods_scores: list[float]` (same length as `moods_top3`)

Preprocessed vocabs
- `preprocessed/vocabs.json` contains:
  - `text_stoi`, `text_itos`
  - `mood_stoi`, `mood_itos`
  - `genre_stoi`, `genre_itos`
  - `pad_id`, `unk_id`

Preprocessed book features
- `preprocessed/book_features.csv` (or `.parquet`) contains fixed-length, numeric features (see schema above).

Checkpoint
- `models/two_tower.pt` bundles: `model_state_dict`, vocabs, and `config` used to reconstruct the model.

---

## 4) Hyperparameters mapping

The app and train script are tolerant to key names in `config` using aliases. Common keys:
- Embeddings: `text_emb_dim`, `mood_emb_dim`, `genre_emb_dim`
- Text encoder: `text_lstm_hidden` (alias `lstm_hidden_dim`), `text_lstm_layers`, `text_bidirectional`, `text_dropout`
- Tower: `tower_dim` (aliases `embedding_dim`, `projection_dim`), `tower_hidden`, `tower_dropout`
- Special IDs: `pad_id`, `unk_id`
- User caps (for inference): `max_user_moods`, `max_user_genres` (used by the app to pad sentences)

---

## 5) Troubleshooting

- Missing dependencies
  - Install `transformers` and `sentence-transformers` for high-quality mood detection.
  - Install a compatible PyTorch wheel for your platform (CPU/GPU).
- Large files / memory
  - Use the Streamlit sidebar to reduce the embedding batch size if you hit memory limits.
  - Prefer Parquet for faster I/O if your environment supports it.
- Missing columns
  - The app can handle `book_text_ids` vs. `text_ids` and similar variants. Ensure `book_mood_ids`, `book_mood_scores`, and `book_genre_ids` exist.
- Vocab mismatches
  - The checkpoint must include `text_vocab`, `mood_vocab`, `genre_vocab` that match the IDs embedded in `book_features`. Rebuild/preprocess if misaligned.
- State dict warnings
  - Some missing/unexpected keys can be benign (e.g., changed dropout); the app warns but proceeds. If embeddings or LSTM shapes differ, retrain or rebuild correctly.

---

## 6) Quick run recipes (Windows cmd.exe)

Install
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Mood tagging
```
python .\mood_classifier_tagger\mood_tagger.py --csv dataset\books_enriched.csv --out dataset\books_with_moods.csv --top-k 3
```

Training (optional)
```
python .\models\train_two_tower.py --data-dir dataset --preprocessed-dir preprocessed
```

Streamlit app
```
streamlit run test_sentence.py
```

---

## 7) Glossary

- Tower: Encoder that produces an embedding for a modality (user or item).
- PAD/UNK: Special tokens used to pad sequences and handle unknown tokens.
- InfoNCE: Contrastive loss used to bring matching user–book pairs closer than non‑matching ones.
- SBERT: Sentence-BERT; used for semantic similarity in mood detection.
- Zero-shot: Classification without fine-tuning via a natural language inference model (BART-MNLI).

---

This document should enable you to reproduce the full pipeline and understand how artifacts connect across steps. If you customize the schema (e.g., number of moods/genres), ensure the checkpoint’s vocabs align with the preprocessed features and the Streamlit app configuration.

