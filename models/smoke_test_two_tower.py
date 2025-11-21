import torch
from two_tower_model import TwoTowerConfig, TwoTowerModel

def main():
    # Dummy vocab sizes
    cfg = TwoTowerConfig(
        text_vocab_size=5000,
        mood_vocab_size=100,
        genre_vocab_size=200,
        pad_id=0,
        unk_id=1,
        text_emb_dim=32,
        mood_emb_dim=16,
        genre_emb_dim=16,
        text_lstm_hidden=32,
        tower_dim=64,
        tower_hidden=128,
    )

    model = TwoTowerModel(cfg)
    model.eval()

    B = 8
    T = 50
    Km_book = 3
    Kg_book = 8
    Km_user = 16
    Kg_user = 16

    # Random inputs (with some pads = 0)
    g = torch.Generator().manual_seed(42)

    text_ids = torch.randint(low=1, high=cfg.text_vocab_size, size=(B, T), generator=g)
    text_ids[:, -5:] = 0  # pad tail

    book_mood_ids = torch.randint(1, cfg.mood_vocab_size, (B, Km_book), generator=g)
    book_mood_ids[:, -1] = 0
    book_mood_scores = torch.rand(B, Km_book, generator=g)
    book_mood_scores[:, -1] = 0.0

    book_genre_ids = torch.randint(1, cfg.genre_vocab_size, (B, Kg_book), generator=g)
    book_genre_ids[:, -2:] = 0

    user_mood_ids = torch.randint(1, cfg.mood_vocab_size, (B, Km_user), generator=g)
    user_mood_ids[:, -4:] = 0
    user_mood_scores = torch.rand(B, Km_user, generator=g)
    user_mood_scores[:, -4:] = 0.0

    user_genre_ids = torch.randint(1, cfg.genre_vocab_size, (B, Kg_user), generator=g)
    user_genre_ids[:, -3:] = 0

    with torch.no_grad():
        u, b = model(
            user_mood_ids=user_mood_ids,
            user_mood_scores=user_mood_scores,
            user_genre_ids=user_genre_ids,
            text_ids=text_ids,
            book_mood_ids=book_mood_ids,
            book_mood_scores=book_mood_scores,
            book_genre_ids=book_genre_ids,
        )
        print('user_vec shape:', tuple(u.shape))
        print('book_vec shape:', tuple(b.shape))
        print('user_vec norm (first 3):', torch.linalg.vector_norm(u, dim=-1)[:3])
        print('book_vec norm (first 3):', torch.linalg.vector_norm(b, dim=-1)[:3])

    loss = TwoTowerModel.contrastive_loss(u, b, temperature=0.07)
    print('InfoNCE loss:', float(loss.item()))

if __name__ == '__main__':
    main()

