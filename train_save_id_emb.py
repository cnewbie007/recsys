import torch
import argparse
import pandas as pd
import torch.nn as nn
from get_ID_data import *


class Config:
    def __init__(self):
        df = pd.read_csv('/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_train.csv')
        self.num_users = max(df['user_id'])
        self.num_items = max(df['item_id'])
        self.id_emb_size = args().id_emb_size


class IDEmbeddingModel(nn.Module):
    def __init__(self):
        super(IDEmbeddingModel).__init__()

        # get the configurations
        self.config = Config()

        # define the user embeddings and item embeddings
        self.user_embeddings = nn.Embedding(self.config.num_users, self.config.id_emb_size)
        self.item_embeddings = nn.Embedding(self.config.num_items, self.config.id_emb_size)

    def forward(self, user_item_pairs):
        # extract the indices of the user and item 
        batch_user_indices = user_item_pairs[:, 0]
        batch_item_indices = user_item_pairs[:, 1]

        # sample the user and item embeddings
        batch_user_embeddings = self.user_embeddings(batch_user_indices)
        batch_item_embeddings = self.item_embeddings(batch_item_indices)

        return batch_user_embeddings, batch_item_embeddings


class TrainIDEmbeddings:
    def __init__(self):
        # define the model
        self.model = IDEmbeddingModel()

        # define optimizers and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args().learning_rate
        )
        self.cos_emb = nn.CosineEmbeddingLoss(margin=0.5)
        self.cos_sim = nn.CosineSimilarity()

        dataset = get_loader()


# Arguments
def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--learning_rate", type=str, default='roberta',
                                  help=" backbone network, roberta/bert ")
    train_arg_parser.add_argument("--id_emb_size", type=int, default=32,
                                  help=" embedding size for ID embedding trainning ")
    train_arg_parser.add_argument("--batchsz", type=int, default=32,
                                  help="batch size")
    train_arg_parser.add_argument("--num_epochs", type=int, default=10,
                                  help="batch size")
    return train_arg_parser.parse_args()