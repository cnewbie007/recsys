import wandb
import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from shared.base import BaseRecommender
from collaborative_filtering.preprocessing import CFData


class UserCFRecommender(BaseRecommender):
    def __init__(self):
        self.data            = None
        self.user_similarity = None

    def train(self, data: CFData):
        self.data = data
        self.build_index()

    def build_index(self):
        self.user_similarity = cosine_similarity(
            self.data.train_matrix, dense_output=False
        )

    def recommend(self, user_id, topk=10):
        user_sim_row = self.user_similarity[user_id]
        scores       = np.asarray(user_sim_row.dot(self.data.train_matrix).todense()).flatten()
        return np.argsort(-scores)[:topk].tolist()

    def evaluate(self, data: CFData):
        test_users, true_items = data.get_test_pairs()
        test_users = test_users.numpy()
        true_items = true_items.numpy()

        test_sims = self.user_similarity[test_users]                                    # (n, num_users)
        scores    = np.asarray(test_sims.dot(self.data.train_matrix).todense())         # (n, num_items)
        top10     = np.argsort(-scores, axis=1)[:, :10]

        hits         = sum(true_items[i] in top10[i] for i in range(len(true_items)))
        recall_at_10 = hits / len(true_items)
        print(f'Recall@10: {recall_at_10:.4f}')
        return {'recall@10': recall_at_10}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / 'user_similarity.npz', self.user_similarity)

    def load(self, path):
        path = Path(path)
        self.user_similarity = sp.load_npz(path / 'user_similarity.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(project='recsys-cf', name='usercf')

    data        = CFData()
    recommender = UserCFRecommender()
    recommender.train(data)
    metrics = recommender.evaluate(data)
    recommender.save('checkpoints/usercf')

    if not args.no_wandb:
        wandb.log(metrics)
        wandb.finish()
