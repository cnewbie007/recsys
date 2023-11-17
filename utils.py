import json
import random
import pandas as pd
import collections
from sklearn.preprocessing import LabelEncoder


def user_item_index():
    
    # read the file
    path = './amazon/reviews_Electronics_5.json'

    # get the pair of the user and item
    user_ids = []
    item_ids = []
    with open(path, 'r', encoding='UTF-8') as file:
        for pair in file:
            pair = json.loads(pair)
            user_ids.append(pair['reviewerID'])
            item_ids.append(pair['asin'])

    # define the index encoder    
    index_encoder = LabelEncoder()

    # get user and item index (raw id -> user_index / item_index)
    user_index = index_encoder.fit_transform(user_ids)
    item_index = index_encoder.fit_transform(item_ids)

    return user_index, item_index


def split_train_test(user_indices, item_indices):

    # get the freqs of the user and item, make sure they appear in both train and test
    user_freqs = collections.Counter(user_indices)
    item_freqs = collections.Counter(item_indices)

    # get test_indices by making sure some rare pairs in both train and test
    test_indices = []
    for i in range(len(user_indices)):
        if random.random() < 0.1 and user_freqs[user_indices[i]] > 3 and item_freqs[item_indices[i]] > 2:
            test_indices.append(i)
            user_freqs[user_indices[i]] -= 1
            item_freqs[item_indices[i]] -= 1
    
    # define the dataframe for all data
    dataframe = pd.DataFrame(
        {
            'idx_in_org': [x for x in range(len(user_indices))],
            'user_id': user_indices,
            'item_id': item_indices
        },
        columns=[
            'idx_in_org',
            'user_id',
            'item_id'
        ]
    )

    # split the train and test by test indices
    test_df = dataframe.iloc[test_indices]
    train_df = dataframe.drop(test_indices)

    # save train and test pair
    train_df.to_csv('./amazon/encoded_pairs_train.csv', index=False)
    test_df.to_csv('./amazon/encoded_pairs_test.csv', index=False)

    print(len(test_indices) / len(user_indices))


# here is the adding comment
users, items = user_item_index()
split_train_test(users, items)

