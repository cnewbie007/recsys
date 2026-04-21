import wandb
import torch
import argparse
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from shared.base import BaseRecommender
from two_tower.model import TwoTowerModel
from two_tower.preprocessing import TwoTowerData, NUM_GENRES


class TwoTowerRecommender(BaseRecommender):
    def __init__(self, emb_size=64, output_size=64, batch_size=512, num_epochs=50, lr=1e-3):
        self.emb_size = emb_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data = None
        self.item_index = None

    def train(self, data: TwoTowerData, use_wandb=False):
        self.data = data
        self.model = TwoTowerModel(
            data.num_users, data.num_items, NUM_GENRES,
            self.emb_size, self.output_size
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        user_ids, item_ids = data.get_train_pairs()
        loader = DataLoader(TensorDataset(user_ids, item_ids), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for user_batch, item_batch in loader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)

                user_emb, item_emb = self.model(
                    user_batch,
                    data.user_gender[user_batch].to(self.device),
                    data.user_age[user_batch].to(self.device),
                    data.user_occupation[user_batch].to(self.device),
                    item_batch,
                    data.item_genres[item_batch].to(self.device)
                )
                loss = self.model.in_batch_softmax_loss(user_emb, item_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch+1:02d}/{self.num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

            log = {'epoch': epoch + 1, 'train_loss': avg_loss, 'lr': scheduler.get_last_lr()[0]}

            if (epoch + 1) % 10 == 0:
                self.build_index()
                metrics = self.evaluate(data)
                log['recall@10'] = metrics['recall@10']

            if use_wandb:
                wandb.log(log)

        self.build_index()

    def build_index(self):
        self.model.eval()
        all_item_ids = torch.arange(self.data.num_items).to(self.device)
        all_genres = self.data.item_genres.to(self.device)

        with torch.no_grad():
            self.item_index = self.model.item_tower(all_item_ids, all_genres).cpu().numpy()

    def recommend(self, user_id, topk=10):
        self.model.eval()
        user_tensor = torch.tensor([user_id]).to(self.device)

        with torch.no_grad():
            user_emb = self.model.user_tower(
                user_tensor,
                self.data.user_gender[user_tensor].to(self.device),
                self.data.user_age[user_tensor].to(self.device),
                self.data.user_occupation[user_tensor].to(self.device)
            ).cpu().numpy()

        scores = user_emb @ self.item_index.T
        return np.argsort(-scores[0])[:topk].tolist()

    def evaluate(self, test_data: TwoTowerData):
        self.model.eval()
        user_ids, true_items = test_data.get_test_pairs()

        all_users = user_ids.to(self.device)
        with torch.no_grad():
            user_embs = self.model.user_tower(
                all_users,
                test_data.user_gender[all_users].to(self.device),
                test_data.user_age[all_users].to(self.device),
                test_data.user_occupation[all_users].to(self.device)
            ).cpu().numpy()

        scores = user_embs @ self.item_index.T
        top10 = np.argsort(-scores, axis=1)[:, :10]

        hits = sum(true_item in top10[i] for i, true_item in enumerate(true_items.numpy()))
        recall_at_10 = hits / len(user_ids)
        print(f'Recall@10: {recall_at_10:.4f}')
        return {'recall@10': recall_at_10}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / 'model.pt')
        with open(path / 'item_index.pkl', 'wb') as f:
            pickle.dump(self.item_index, f)

    def load(self, path):
        path = Path(path)
        self.model.load_state_dict(torch.load(path / 'model.pt', map_location=self.device))
        with open(path / 'item_index.pkl', 'rb') as f:
            self.item_index = pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    wandb.init(
        project='recsys-two-tower',
        name=f'emb{args.emb_size}-ep{args.epochs}',
        config=vars(args)
    )

    data = TwoTowerData()
    recommender = TwoTowerRecommender(
        emb_size=args.emb_size,
        output_size=args.emb_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )
    recommender.train(data, use_wandb=True)
    metrics = recommender.evaluate(data)
    wandb.log({'final_recall@10': metrics['recall@10']})
    wandb.finish()
    recommender.save(f'checkpoints/two_tower_emb{args.emb_size}')
