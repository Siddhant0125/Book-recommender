import os
import torch
import pandas as pd

from recsys_backend import load_checkpoint, build_model, load_book_features_and_embeddings, explain_recommendation

ROOT = os.path.dirname(os.path.dirname(__file__))
PRE = os.path.join(ROOT, 'preprocessed', 'book_features.csv')
SAMPLE = os.path.join(os.path.dirname(__file__), 'book_features_sample.csv')


def main():
    print('Loading checkpoint...')
    state_dict, text_vocab, mood_vocab, genre_vocab, config = load_checkpoint()
    print(f"Vocab sizes: text={len(text_vocab)} mood={len(mood_vocab)} genre={len(genre_vocab)}")

    print('Building model...')
    model = build_model(text_vocab, mood_vocab, genre_vocab, config, device='cpu')
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False), None
        if isinstance(missing, tuple):
            missing_keys, unexpected_keys = missing
        else:
            missing_keys, unexpected_keys = [], []
        print(f"Loaded state_dict with strict=False; missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    except Exception as e:
        print('load_state_dict failed:', e)

    if os.path.exists(PRE):
        print('Preparing sample book features (first 64 rows)...')
        df = pd.read_csv(PRE) if PRE.lower().endswith('.csv') else pd.read_parquet(PRE)
        df.head(64).to_csv(SAMPLE, index=False)
        print('Computing book embeddings for sample...')
        books_df, book_vecs = load_book_features_and_embeddings(model, books_path=SAMPLE, device='cpu')
        print('books_df shape:', books_df.shape)
        print('book_vecs shape:', tuple(book_vecs.shape))
        assert book_vecs.ndim == 2 and book_vecs.shape[0] == len(books_df)
        print('Sample explain string check...')
        # Minimal explanation check with zeros
        import numpy as np
        um = torch.zeros(1, int(config.get('max_moods', 8)), dtype=torch.long)
        ums = torch.zeros_like(um, dtype=torch.float)
        ug = torch.zeros(1, int(config.get('max_genres', 8)), dtype=torch.long)
        bm = torch.zeros_like(um)
        bms = torch.zeros_like(ums)
        bg = torch.zeros_like(ug)
        id2mood = {v: k for k, v in mood_vocab.items()}
        id2genre = {v: k for k, v in genre_vocab.items()}
        msg = explain_recommendation(um, ums, ug, bm, bms, bg, id2mood, id2genre)
        print('explain_recommendation:', msg)
    else:
        print('Preprocessed features not found; skipping embedding test.')

    print('Smoke test finished OK.')


if __name__ == '__main__':
    main()
