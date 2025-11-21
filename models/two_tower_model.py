from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    # Vocab sizes and special ids
    text_vocab_size: int
    mood_vocab_size: int
    genre_vocab_size: int
    pad_id: int = 0
    unk_id: int = 1

    # Embedding sizes
    text_emb_dim: int = 128
    mood_emb_dim: int = 64
    genre_emb_dim: int = 64

    # Text encoder
    text_lstm_hidden: int = 128
    text_lstm_layers: int = 1
    text_bidirectional: bool = True
    text_dropout: float = 0.1

    # Fusion + tower projection
    tower_dim: int = 128
    tower_hidden: int = 256
    tower_dropout: float = 0.1

    # Initialization
    emb_init_std: float = 0.02


class MoodEncoder(nn.Module):
    """Weighted pooling over mood embeddings.
    Inputs:
      mood_ids: LongTensor [B, K]
      mood_scores: FloatTensor [B, K] (weights, typically normalized but not required)
    Output:
      mood_vec: Tensor [B, D]
    """

    def __init__(self, vocab_size: int, emb_dim: int, pad_id: int = 0, init_std: float = 0.02):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        nn.init.normal_(self.emb.weight, std=init_std)
        if pad_id is not None:
            with torch.no_grad():
                self.emb.weight[pad_id].zero_()

    def forward(self, mood_ids: torch.LongTensor, mood_scores: torch.FloatTensor) -> torch.Tensor:
        mask = (mood_ids != self.pad_id).float()  # [B, K]
        scores = torch.clamp(mood_scores, min=0.0) * mask  # zero out pads and negatives
        # Normalize per sample
        denom = scores.sum(dim=1, keepdim=True) + 1e-8
        weights = scores / denom
        emb = self.emb(mood_ids)  # [B, K, D]
        pooled = (emb * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return pooled


class MultiIDEncoder(nn.Module):
    """Mean-pooled embedding for a set of IDs (e.g., genres)."""

    def __init__(self, vocab_size: int, emb_dim: int, pad_id: int = 0, init_std: float = 0.02):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        nn.init.normal_(self.emb.weight, std=init_std)
        if pad_id is not None:
            with torch.no_grad():
                self.emb.weight[pad_id].zero_()

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        mask = (ids != self.pad_id).float()  # [B, K]
        emb = self.emb(ids)  # [B, K, D]
        summed = (emb * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        count = mask.sum(dim=1, keepdim=True)  # [B, 1]
        pooled = summed / (count + 1e-8)
        return pooled


class BookTextEncoder(nn.Module):
    """Embedding + (Bi)LSTM text encoder with mean pooling over valid tokens."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.1, pad_id: int = 0, init_std: float = 0.02):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        nn.init.normal_(self.emb.weight, std=init_std)
        if pad_id is not None:
            with torch.no_grad():
                self.emb.weight[pad_id].zero_()

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, text_ids: torch.LongTensor) -> torch.Tensor:
        # lengths from mask
        mask = (text_ids != self.pad_id).long()  # [B, T]
        lengths = mask.sum(dim=1).clamp(min=1).cpu()

        x = self.emb(text_ids)  # [B, T, E]
        # pack padded sequence for LSTM efficiency
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H*D]

        # Recompute mask to match out shape (T may be reduced)
        T = out.size(1)
        mask = (torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)).float()  # [B, T]

        # masked mean pooling
        summed = (out * mask.unsqueeze(-1)).sum(dim=1)  # [B, H*D]
        count = mask.sum(dim=1, keepdim=True)  # [B, 1]
        pooled = summed / (count + 1e-8)
        return self.dropout(pooled)


class BookTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.text_enc = BookTextEncoder(
            vocab_size=cfg.text_vocab_size,
            emb_dim=cfg.text_emb_dim,
            hidden=cfg.text_lstm_hidden,
            num_layers=cfg.text_lstm_layers,
            bidirectional=cfg.text_bidirectional,
            dropout=cfg.text_dropout,
            pad_id=cfg.pad_id,
            init_std=cfg.emb_init_std,
        )
        self.mood_enc = MoodEncoder(cfg.mood_vocab_size, cfg.mood_emb_dim, pad_id=cfg.pad_id, init_std=cfg.emb_init_std)
        self.genre_enc = MultiIDEncoder(cfg.genre_vocab_size, cfg.genre_emb_dim, pad_id=cfg.pad_id, init_std=cfg.emb_init_std)

        # Projections to a common fusion space
        self.proj_text = nn.Linear(self.text_enc.out_dim, cfg.tower_dim)
        self.proj_mood = nn.Linear(cfg.mood_emb_dim, cfg.tower_dim)
        self.proj_genre = nn.Linear(cfg.genre_emb_dim, cfg.tower_dim)

        fusion_in = cfg.tower_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.tower_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.tower_dropout),
            nn.Linear(cfg.tower_hidden, cfg.tower_dim),
        )

    def forward(
        self,
        text_ids: torch.LongTensor,
        book_mood_ids: torch.LongTensor,
        book_mood_scores: torch.FloatTensor,
        book_genre_ids: torch.LongTensor,
    ) -> torch.Tensor:
        t = self.text_enc(text_ids)  # [B, Ht]
        m = self.mood_enc(book_mood_ids, book_mood_scores)  # [B, Hm]
        g = self.genre_enc(book_genre_ids)  # [B, Hg]

        t = self.proj_text(t)
        m = self.proj_mood(m)
        g = self.proj_genre(g)
        z = torch.cat([t, m, g], dim=-1)
        z = self.mlp(z)
        z = F.normalize(z, dim=-1)
        return z


class UserTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.mood_enc = MoodEncoder(cfg.mood_vocab_size, cfg.mood_emb_dim, pad_id=cfg.pad_id, init_std=cfg.emb_init_std)
        self.genre_enc = MultiIDEncoder(cfg.genre_vocab_size, cfg.genre_emb_dim, pad_id=cfg.pad_id, init_std=cfg.emb_init_std)

        self.proj_mood = nn.Linear(cfg.mood_emb_dim, cfg.tower_dim)
        self.proj_genre = nn.Linear(cfg.genre_emb_dim, cfg.tower_dim)

        fusion_in = cfg.tower_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.tower_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.tower_dropout),
            nn.Linear(cfg.tower_hidden, cfg.tower_dim),
        )

    def forward(
        self,
        user_mood_ids: torch.LongTensor,
        user_mood_scores: torch.FloatTensor,
        user_genre_ids: torch.LongTensor,
    ) -> torch.Tensor:
        m = self.mood_enc(user_mood_ids, user_mood_scores)  # [B, Hm]
        g = self.genre_enc(user_genre_ids)  # [B, Hg]
        m = self.proj_mood(m)
        g = self.proj_genre(g)
        z = torch.cat([m, g], dim=-1)
        z = self.mlp(z)
        z = F.normalize(z, dim=-1)
        return z


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.user = UserTower(cfg)
        self.book = BookTower(cfg)

    def forward(
        self,
        user_mood_ids: torch.LongTensor,
        user_mood_scores: torch.FloatTensor,
        user_genre_ids: torch.LongTensor,
        text_ids: torch.LongTensor,
        book_mood_ids: torch.LongTensor,
        book_mood_scores: torch.FloatTensor,
        book_genre_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.user(user_mood_ids, user_mood_scores, user_genre_ids)
        b = self.book(text_ids, book_mood_ids, book_mood_scores, book_genre_ids)
        return u, b

    @staticmethod
    def contrastive_loss(user_vec: torch.Tensor, book_vec: torch.Tensor, temperature: float = 0.07, symmetric: bool = True) -> torch.Tensor:
        # user_vec, book_vec: [B, D], assumed L2-normalized
        logits = user_vec @ book_vec.t()  # [B, B]
        logits = logits / temperature
        targets = torch.arange(user_vec.size(0), device=user_vec.device)
        loss_u2i = F.cross_entropy(logits, targets)
        if symmetric:
            loss_i2u = F.cross_entropy(logits.t(), targets)
            return 0.5 * (loss_u2i + loss_i2u)
        return loss_u2i
