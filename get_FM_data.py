import json
from typing import Any
import torch
import pandas as pd
import collections
from torch.utils.data import Dataset, DataLoader


class IDEmbDataset(Dataset):
    def __init__(self, mode):
        # load datasets
        dataset = pd.read_csv(
            '/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_{}.csv'.format(mode)
        )[['user_id', 'item_id']]
        fake_data = pd.read_csv(
            '/home/keyu/keyu/recommendation/data/amazon/fake_interactions_{}.csv'.format(mode)
        )

        # fake data rename the headers
        fake_data.rename(
            columns={
                'fake_user':'user_id',
                'fake_item':'item_id'
            }, 
            inplace=True
        )

        # assign the labels
        dataset['label'] = 1
        fake_data['label'] = -1

        # concatenate the datasets
        self.data = pd.concat([dataset, fake_data])

        # convert to torch tensors
        self.user_ids = torch.tensor(self.data['user_id'].values)
        self.item_ids = torch.tensor(self.data['item_id'].values)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.user_ids[index], self.item_ids[index], self.labels[index]


def get_loader(mode, batchsz):
    mapping = {
        'train': True,
        'test': False
    }

    # define dataset
    data = IDEmbDataset(mode)
    # get loader
    loader = DataLoader(data, batch_size=batchsz, shuffle=mapping[mode])

    return loader