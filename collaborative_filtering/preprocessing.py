import numpy as np
import torch
import scipy.sparse as sp
from data.ml1m import load_ml1m


class CFData:
    def __init__(self):
        self.train, self.test, self.users, self.movies = load_ml1m()

        self.num_users = self.users['user_id'].max() + 1
        self.num_items = self.movies['item_id'].max() + 1

        rows = self.train['user_id'].values
        cols = self.train['item_id'].values
        self.train_matrix = sp.csr_matrix(
            (np.ones(len(self.train)), (rows, cols)),
            shape=(self.num_users, self.num_items)
        )

    def get_test_pairs(self):
        users = torch.tensor(self.test['user_id'].values, dtype=torch.long)
        items = torch.tensor(self.test['item_id'].values, dtype=torch.long)
        return users, items
