import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users, emb_size, output_size):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.gender_emb = nn.Embedding(2, 8)
        self.age_emb = nn.Embedding(7, 8)
        self.occupation_emb = nn.Embedding(21, 16)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size + 8 + 8 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

    def forward(self, user_ids, genders, ages, occupations):
        x = torch.cat([
            self.user_emb(user_ids),
            self.gender_emb(genders),
            self.age_emb(ages),
            self.occupation_emb(occupations)
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres, emb_size, output_size):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.genre_proj = nn.Linear(num_genres, 32, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

    def forward(self, item_ids, genres):
        x = torch.cat([
            self.item_emb(item_ids),
            self.genre_proj(genres)
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_size=64, output_size=64):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_size, output_size)
        self.item_tower = ItemTower(num_items, num_genres, emb_size, output_size)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, user_ids, genders, ages, occupations, item_ids, genres):
        user_emb = self.user_tower(user_ids, genders, ages, occupations)
        item_emb = self.item_tower(item_ids, genres)
        return user_emb, item_emb

    def in_batch_softmax_loss(self, user_emb, item_emb):
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature.clamp(min=0.01)
        labels = torch.arange(len(user_emb), device=user_emb.device)
        return F.cross_entropy(logits, labels)
