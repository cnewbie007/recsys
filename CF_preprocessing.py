import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity


def construct_user_item_matrix():
    # load dataset
    dataset = pd.read_csv(
        '/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_train.csv'
    )[['user_id', 'item_id']]
    dataset['interaction'] = 1

    # initial the item user matrix
    user_ids = dataset['user_id']
    item_ids = dataset['item_id']
    interaction = dataset['interaction']

    # create sparse matrix for both user-item and item-user
    num_users = max(dataset['user_id']) + 1
    num_items = max(dataset['item_id']) + 1
    num_pairs = len(dataset)

    user_item_matrix = sp.csr_matrix(
        (
            interaction,
            (user_ids, item_ids)
        ),
        shape=(num_users, num_items)
    )
    
    # save the user-item matrix
    save_npz(
        '/home/keyu/keyu/recommendation/data/amazon/user_item_matrix.npz', 
        user_item_matrix
    )


def load_user_item_matrix():
    # check if the file exists 
    path = '/home/keyu/keyu/recommendation/data/amazon/user_item_matrix.npz'
    if not os.path.exists(path):
        construct_user_item_matrix()
    
    # load the matrix
    user_item_matrix = load_npz(path)
    return user_item_matrix


if __name__ == '__main__':
    matrix = load_user_item_matrix()