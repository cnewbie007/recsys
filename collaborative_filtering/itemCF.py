import wandb
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from shared.base import BaseRecommender
from collaborative_filtering.preprocessing import CFData


class ItemCFRecommender(BaseRecommender):
    def __init__(self):
        self.data            = None
        self.item_similarity = None

    def train(self, data: CFData):
        self.data = data
        self.build_index()

    def build_index(self):
        self.item_similarity = cosine_similarity(
            self.data.train_matrix.T, dense_output=False
        )

    def recommend(self, user_id, topk=10):
        user_row = self.data.train_matrix[user_id]
        scores   = np.asarray(user_row.dot(self.item_similarity).todense()).flatten()
        return np.argsort(-scores)[:topk].tolist()

    def evaluate(self, data: CFData):
        test_users, true_items = data.get_test_pairs()
        test_users = test_users.numpy()
        true_items = true_items.numpy()

        test_rows = self.data.train_matrix[test_users]                          # (n, num_items)
        scores    = np.asarray(test_rows.dot(self.item_similarity).todense())   # (n, num_items)
        top10     = np.argsort(-scores, axis=1)[:, :10]

        hits         = sum(true_items[i] in top10[i] for i in range(len(true_items)))
        recall_at_10 = hits / len(true_items)
        print(f'Recall@10: {recall_at_10:.4f}')
        return {'recall@10': recall_at_10}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / 'item_similarity.npz', self.item_similarity)

    def load(self, path):
        path = Path(path)
        self.item_similarity = sp.load_npz(path / 'item_similarity.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(project='recsys-cf', name='itemcf')

    data        = CFData()
    recommender = ItemCFRecommender()
    recommender.train(data)
    metrics = recommender.evaluate(data)
    recommender.save('checkpoints/itemcf')

    if not args.no_wandb:
        wandb.log(metrics)
        wandb.finish()
