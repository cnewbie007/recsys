import json
import random
import pandas as pd
import collections
from sklearn.preprocessing import LabelEncoder


def get_rawdata():

    # read the file:
    path = '/home/keyu/keyu/recommendation/data/amazon/reviews_Electronics_5.json'

    # get the pair of the user and item
    user_ids = []
    item_ids = []
    review_scores = []
    timestamp = []
    with open(path, 'r', encoding='UTF-8') as file:
        for pair in file:
            pair = json.loads(pair)
            user_ids.append(pair['reviewerID'])
            item_ids.append(pair['asin'])
            review_scores.append(pair['overall'])
            timestamp.append(pair['unixReviewTime'])

    # define the index encoder    
    index_encoder = LabelEncoder()

    # get user and item index (raw id -> user_index / item_index)
    user_index = index_encoder.fit_transform(user_ids)
    item_index = index_encoder.fit_transform(item_ids)

    # user interacted items
    size = len(user_index)
    user_interact_dict = collections.defaultdict(set)
    for i in range(size):
        user = user_index[i]
        item = item_index[i]
        user_interact_dict[user].add(item)
    
    # randomly generate 1/3 of none interated pairs out of total interactions for each user
    item_set = list(set(item_index))
    fake_users = []
    fake_items = []
    for user in user_interact_dict:
        n = len(user_interact_dict[user]) // 3 * 2 + 1
        while n > 0:
            item_id = random.choice(item_set)
            if item_id not in user_interact_dict[user]:
                fake_users.append(user)
                fake_items.append(item_id)
                n -= 1
        
    # generate fake test indices, 1/2
    fake_test_indices = random.sample([x for x in range(len(fake_users))], len(fake_users) // 2)
    # save the fakes
    dataframe = pd.DataFrame(
        {
            'fake_user': fake_users,
            'fake_item': fake_items
        },
        columns=[
            'fake_user',
            'fake_item'
        ]
    )

    train_fake = dataframe.iloc[fake_test_indices]
    test_fake = dataframe.drop(fake_test_indices)

    train_fake.to_csv('/home/keyu/keyu/recommendation/data/amazon/fake_interactions_train.csv', index=False)
    test_fake.to_csv('/home/keyu/keyu/recommendation/data/amazon/fake_interactions_test.csv', index=False)

    return user_ids, item_ids, review_scores, timestamp, user_index, item_index    
    # read the file
    path = '/home/keyu/keyu/recommendation/data/amazon/reviews_Electronics_5.json'

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


def get_test_idices(user_indices, item_indices):

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
    
    return test_indices


def preprocessing_main():
    # get the data list
    user_ids, item_ids, review_scores, timestamp, user_enc_ids, item_enc_ids = get_rawdata()

    # get the test indices
    test_indices = get_test_idices(user_enc_ids, item_enc_ids)

    # define the dataframe for all data
    dataframe = pd.DataFrame(
        {
            'org_user_id': user_ids,
            'org_item_id': item_ids,
            'user_id': user_enc_ids,
            'item_id': item_enc_ids,
            'review_score': review_scores,
            'timestamp': timestamp
        },
        columns=[
            'org_user_id',
            'org_item_id',
            'user_id',
            'item_id',
            'review_score',
            'timestamp'
        ]
    )

    # split the train and test by test indices
    test_df = dataframe.iloc[test_indices]
    train_df = dataframe.drop(test_indices)

    # save train and test pair
    train_df.to_csv('/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_train.csv', index=False)
    test_df.to_csv('/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_test.csv', index=False)


if __name__ == '__main__':
    preprocessing_main()