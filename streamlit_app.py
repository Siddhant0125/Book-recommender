#!/usr/bin/env python3
"""
Streamlit app to test a two-tower mood-based book recommender.

Usage:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
from typing import Dict, Tuple, List

import torch
import pandas as pd
import streamlit as st

from backend.recsys_backend import (
    load_checkpoint,
    build_model,
    load_book_features_and_embeddings,
    build_user_features_from_sentence,
    explain_recommendation,
)

# ------------------------------------------------------------------
# Config: default artifact paths (for display only)
# ------------------------------------------------------------------
HERE = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.normpath(os.path.join(HERE, "models", "two_tower.pt"))
BOOKS_FEATURES_PATH = os.path.normpath(os.path.join(HERE, "preprocessed", "book_features.parquet"))


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def _load_model_and_data(device: str) -> Tuple[torch.nn.Module, Dict[str, int], Dict[str, int], Dict[str, int], Dict, pd.DataFrame, torch.Tensor, Dict[int, str], Dict[int, str]]:
    # Load checkpoint (backend resolves path relative to backend/ by default)
    state_dict, text_vocab, mood_vocab, genre_vocab, config = load_checkpoint(device=device)
    # Build model and load weights
    model = build_model(text_vocab, mood_vocab, genre_vocab, config, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load books + embeddings (backend default path)
    books_df, book_vecs = load_book_features_and_embeddings(model, device=device)
    # Ensure embeddings are on CPU for similarity calc
    if book_vecs.device.type != "cpu":
        book_vecs = book_vecs.cpu()

    # id->label maps
    id2mood = {v: k for k, v in mood_vocab.items()}
    id2genre = {v: k for k, v in genre_vocab.items()}

    return model, text_vocab, mood_vocab, genre_vocab, config, books_df, book_vecs, id2mood, id2genre


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------

def _display_metadata(books_df: pd.DataFrame, mood_vocab: Dict[str, int], genre_vocab: Dict[str, int]) -> None:
    # Add CSS to increase metric font size significantly
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        font-size: 36px !important;
    }
    [data-testid="metric-container"] > div {
        font-size: 32px !important;
    }
    [data-testid="metric-container"] > div:last-child {
        font-size: 36px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 28px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Books", f"{len(books_df):,}")
    c2.metric("Moods", f"{len(mood_vocab):,}")
    c3.metric("Genres", f"{len(genre_vocab):,}")

    # Show a few example labels (filter out [PAD] and [UNK])
    st.markdown("### Examples")
    mood_samples = [m for m in list(mood_vocab.keys())[:15] if m not in ["[PAD]", "[UNK]"]][:10]
    genre_samples = [g for g in list(genre_vocab.keys())[:15] if g not in ["[PAD]", "[UNK]"]][:10]
    st.markdown(f"<p style='font-size:20px;'>• Moods: {', '.join(mood_samples)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'>• Genres: {', '.join(genre_samples)}</p>", unsafe_allow_html=True)


def _user_mood_summary(user_mood_ids: torch.Tensor, user_mood_scores: torch.Tensor, id2mood: Dict[int, str]) -> List[str]:
    # user tensors are [1, K]
    mids = user_mood_ids[0].detach().cpu().tolist()
    mscs = user_mood_scores[0].detach().cpu().tolist()
    out: List[str] = []
    for i, s in zip(mids, mscs):
        if int(i) != 0 and float(s) > 0:
            mood_name = id2mood.get(int(i), str(i))
            # Filter out [PAD] and [UNK] tokens
            if mood_name not in ["[PAD]", "[UNK]"]:
                out.append(mood_name)
    return out


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Mood Based Book Recommender", layout="wide")
    st.markdown("# 📚 Mood-Based Book Recommender")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Device: {device}")

    # Load model and data (cached)
    try:
        model, text_vocab, mood_vocab, genre_vocab, config, books_df, book_vecs, id2mood, id2genre = _load_model_and_data(device)
    except FileNotFoundError as e:
        st.error(f"Artifact not found: {e}")
        return
    except Exception as e:
        st.exception(e)
        return

    # Metadata
    with st.expander("Dataset and model info", expanded=True):
        _display_metadata(books_df, mood_vocab, genre_vocab)
        # st.caption(f"Checkpoint: {CHECKPOINT_PATH}")
        # st.caption(f"Book features: {BOOKS_FEATURES_PATH}")

    # UI inputs
    st.markdown("## Describe what you're in the mood to read")

    st.markdown("<p style='font-size:20px; font-weight: bold;'>Number of recommendations</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        top_k = st.number_input("", min_value=1, max_value=5, value=2, step=1, label_visibility="collapsed")
    with col2:
        st.empty()

    st.markdown("<p style='font-size:20px; font-weight: bold;'>How do you feel, and what themes do you want?</p>", unsafe_allow_html=True)

    # Add CSS to increase text area font size
    st.markdown("""
    <style>
    textarea {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    sentence = st.text_area("", height=120, placeholder="e.g., I want a hopeful, cozy fantasy adventure with a touch of romance.", label_visibility="collapsed")

    recommend = st.button("Recommend", use_container_width=True)

    if recommend:
        if not sentence.strip():
            st.warning("Please enter a short sentence about your mood or what you'd like to read.")
            return

        with st.spinner("🔍 Analyzing your mood and finding perfect books..."):
            # Build user features from sentence
            max_moods = int(config.get("max_moods", 8))
            max_genres = int(config.get("max_genres", 8))
            user_mood_ids, user_mood_scores, user_genre_ids = build_user_features_from_sentence(
                sentence=sentence,
                mood_vocab=mood_vocab,
                max_moods=max_moods,
                max_genres=max_genres,
                device=device,
            )

            # Compute user vector (on selected device)
            with torch.no_grad():
                user_vec = model.user(
                    user_mood_ids=user_mood_ids,
                    user_mood_scores=user_mood_scores,
                    user_genre_ids=user_genre_ids,
                )  # [1, D]
            # Move to CPU to match book_vecs
            user_vec_cpu = user_vec.detach().cpu()
            scores = (user_vec_cpu @ book_vecs.T).squeeze(0)  # [N]
            topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))

        # Show detected moods
        detected = _user_mood_summary(user_mood_ids, user_mood_scores, id2mood)
        if detected:
            st.markdown("### Detected moods:")
            st.markdown(f"**{', '.join(detected)}**")
        else:
            st.warning("No moods detected (zero scores). The mood tagger may not be available or the text was too vague.")

        st.markdown("## Recommendations")
        for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), start=1):
            row = books_df.iloc[idx]
            title = row.get("title", f"Book {row.get('book_id', idx)}")
            description = row.get("description", "") or "(no description)"
            # Prepare book tensors for explanation
            bm_ids = torch.tensor(row["book_mood_ids"], dtype=torch.long)
            bm_scores = torch.tensor(row["book_mood_scores"], dtype=torch.float)
            bg_ids = torch.tensor(row["book_genre_ids"], dtype=torch.long)
            why = explain_recommendation(
                user_mood_ids=user_mood_ids[0],
                user_mood_scores=user_mood_scores[0],
                user_genre_ids=user_genre_ids[0],
                book_mood_ids=bm_ids,
                book_mood_scores=bm_scores,
                book_genre_ids=bg_ids,
                id2mood=id2mood,
                id2genre=id2genre,
            )
            with st.container():
                st.markdown(f"### {rank}. {title}")
                st.markdown(f"**Similarity: {score:.4f}**")
                st.markdown(f"<p style='font-size:16px;'>{description if len(description) < 800 else description[:800] + ' …'}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px; font-style: italic;'>**Why this book?** {why}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
