import torch
import torch.nn as nn
import torch.nn.functional as F


class FMInteraction(nn.Module):
    """Element-wise second-order FM interaction: 0.5 * (sum(v)^2 - sum(v^2))"""

    def forward(self, fields):
        # fields: (B, n_fields, emb_size)
        sum_emb = fields.sum(dim=1)           # (B, emb_size)
        sum_sq  = (fields ** 2).sum(dim=1)    # (B, emb_size)
        return 0.5 * (sum_emb ** 2 - sum_sq)  # (B, emb_size)


class FMUserTower(nn.Module):
    def __init__(self, num_users, emb_size, output_size):
        super().__init__()
        self.user_emb       = nn.Embedding(num_users, emb_size)
        self.gender_emb     = nn.Embedding(2, emb_size)
        self.age_emb        = nn.Embedding(7, emb_size)
        self.occupation_emb = nn.Embedding(21, emb_size)
        self.fm   = FMInteraction()
        self.proj = nn.Linear(emb_size, output_size)

    def forward(self, user_ids, genders, ages, occupations):
        fields = torch.stack([
            self.user_emb(user_ids),
            self.gender_emb(genders),
            self.age_emb(ages),
            self.occupation_emb(occupations),
        ], dim=1)                                      # (B, 4, emb_size)
        return F.normalize(self.proj(self.fm(fields)), dim=-1)


class FMItemTower(nn.Module):
    def __init__(self, num_items, num_genres, emb_size, output_size):
        super().__init__()
        self.item_emb   = nn.Embedding(num_items, emb_size)
        self.genre_proj = nn.Linear(num_genres, emb_size, bias=False)
        self.fm   = FMInteraction()
        self.proj = nn.Linear(emb_size, output_size)

    def forward(self, item_ids, genres):
        fields = torch.stack([
            self.item_emb(item_ids),
            self.genre_proj(genres),
        ], dim=1)                                      # (B, 2, emb_size)
        return F.normalize(self.proj(self.fm(fields)), dim=-1)


class FMModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_size=64, output_size=64):
        super().__init__()
        self.user_tower = FMUserTower(num_users, emb_size, output_size)
        self.item_tower = FMItemTower(num_items, num_genres, emb_size, output_size)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, user_ids, genders, ages, occupations, item_ids, genres):
        user_emb = self.user_tower(user_ids, genders, ages, occupations)
        item_emb = self.item_tower(item_ids, genres)
        return user_emb, item_emb

    def in_batch_softmax_loss(self, user_emb, item_emb):
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature.clamp(min=0.01)
        labels = torch.arange(len(user_emb), device=user_emb.device)
        return F.cross_entropy(logits, labels)
