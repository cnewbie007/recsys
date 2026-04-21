import numpy as np
import torch
from data.ml1m import load_ml1m


GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES = len(GENRES)


class TwoTowerData:
    def __init__(self):
        self.train, self.test, self.users, self.movies = load_ml1m()

        self.num_users = self.users['user_id'].max() + 1
        self.num_items = self.movies['item_id'].max() + 1

        users_sorted = self.users.sort_values('user_id')
        self.user_gender = torch.tensor(users_sorted['gender'].values, dtype=torch.long)
        self.user_age = torch.tensor(users_sorted['age'].values, dtype=torch.long)
        self.user_occupation = torch.tensor(users_sorted['occupation'].values, dtype=torch.long)

        genre_matrix = np.zeros((self.num_items, NUM_GENRES), dtype=np.float32)
        for _, row in self.movies.iterrows():
            for g in row['genres'].split('|'):
                if g in GENRE_TO_IDX:
                    genre_matrix[row['item_id'], GENRE_TO_IDX[g]] = 1.0
        self.item_genres = torch.tensor(genre_matrix, dtype=torch.float32)

    def get_train_pairs(self):
        users = torch.tensor(self.train['user_id'].values, dtype=torch.long)
        items = torch.tensor(self.train['item_id'].values, dtype=torch.long)
        return users, items

    def get_test_pairs(self):
        users = torch.tensor(self.test['user_id'].values, dtype=torch.long)
        items = torch.tensor(self.test['item_id'].values, dtype=torch.long)
        return users, items
