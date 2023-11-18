import torch
import wandb
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from get_ID_data import *
from torch.autograd import Variable
from sklearn.metrics import *


class Config:
    def __init__(self):
        df = pd.read_csv('/home/keyu/keyu/recommendation/data/amazon/encoded_pairs_train.csv')
        self.num_users = max(df['user_id']) + 1
        self.num_items = max(df['item_id']) + 1
        self.id_emb_size = args().id_emb_size


class IDEmbeddingModel(nn.Module):
    def __init__(self):
        super(IDEmbeddingModel, self).__init__()

        # get the configurations
        self.config = Config()

        # define the user embeddings and item embeddings
        self.user_embeddings = nn.Embedding(self.config.num_users, self.config.id_emb_size)
        self.item_embeddings = nn.Embedding(self.config.num_items, self.config.id_emb_size)

        # Initialize embeddings with random values
        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.1)

    def forward(self, user_ids, item_ids):

        # sample the user and item embeddings
        batch_user_embeddings = self.user_embeddings(user_ids)
        batch_item_embeddings = self.item_embeddings(item_ids)

        return batch_user_embeddings, batch_item_embeddings


class TrainIDEmbeddings:
    def __init__(self):
        # define the model
        self.model = IDEmbeddingModel().to(device)

        # define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args().learning_rate
        )

        # define loss function
        self.cos_emb = nn.CosineEmbeddingLoss(margin=0.5)
        self.cos_sim = nn.CosineSimilarity()
        self.loss_fn = nn.MSELoss()

        # get test_loader
        self.train_loader = get_loader('train', args().batchsz)
        self.test_loader = get_loader('test', args().batchsz)

        print(
            ' #########################################', '\n',
            '######## Data Initialization Done #######', '\n',
            '#########################################',
        )

    def get_batch(self):
        temp_user, temp_item = set(), set()
        k = args().batchsz
        selected_indices = []
        while k > 0:
            idx = random.randint(0, self.num_train)
            curr_user = self.train_user[idx]
            curr_item = self.train_item[idx]
            if curr_user not in temp_user and curr_item not in temp_item:
                selected_indices.append(idx)
                temp_user.add(curr_user)
                temp_item.add(curr_item)
                k -= 1
        selected_indices = torch.tensor(selected_indices)
        return selected_indices

    def evaluation(self):
        y_true = np.array([])
        y_pred = np.array([])

        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                # random sample non repeating pairs
                user_id, item_id, label = batch

                # put to the device
                user_id = Variable(user_id).to(device)
                item_id = Variable(item_id).to(device)

                # get the predictions
                user_emb, item_emb = self.model(user_id, item_id)

                # compute similarity
                similarity = self.cos_sim(user_emb, item_emb)
                batch_pred = torch.where(similarity > 0, torch.tensor(1), torch.tensor(-1))

                # append the result
                y_pred = np.append(y_pred, batch_pred.data.cpu().numpy())
                y_true = np.append(y_true, label)

        auc_score = roc_auc_score(y_true, y_pred)
        return auc_score

    def train(self):
        
        batch_loss = 0
        for _, batch in enumerate(self.train_loader):
            # random sample non repeating pairs
            user_id, item_id, label = batch

            # put to the device
            user_id = Variable(user_id).to(device)
            item_id = Variable(item_id).to(device) 
            label = Variable(label).to(device)
            
            # get the embeddings
            user_emb, item_emb = self.model(user_id, item_id)

            # compute the loss
            # loss = self.cos_emb(user_emb, item_emb, label)
            similarities = self.cos_sim(user_emb, item_emb)
            loss = self.loss_fn(similarities, label)
            batch_loss += loss.item()

            # update the embeddings
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return batch_loss / len(self.train_loader) 

    def workflow(self):
        
        for epoch in range(args().num_epochs):

            # train one epoch
            train_loss = self.train()

            # evaluate the AUC
            auc_score = self.evaluation()

            print()
            print(
                '*Epoch: {:02d}/{:02d}'.format(epoch + 1, args().num_epochs), '\n',
                'Train Loss: {:.5f}'.format(train_loss), '\n',
                'AUC score: {:.5f}'.format(auc_score)
            )

            if args().wandb:
                wandb.log(
                {
                    'Train Loss': train_loss,
                    'AUC': auc_score
                }
            )


# Arguments
def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--learning_rate", type=float, default=0.0005,
                                  help=" backbone network, roberta/bert ")
    train_arg_parser.add_argument("--id_emb_size", type=int, default=32,
                                  help=" embedding size for ID embedding trainning ")
    train_arg_parser.add_argument("--batchsz", type=int, default=32,
                                  help="batch size")
    train_arg_parser.add_argument("--num_epochs", type=int, default=30,
                                  help="batch size")
    train_arg_parser.add_argument("--wandb", type=int, default=0,
                                  help="batch size")
    return train_arg_parser.parse_args()


if __name__ == '__main__':

    device = torch.device('cuda:{}'.format(args().gpu) if torch.cuda.is_available() else 'cpu')

    if args().wandb:
        wandb.init(
            project='RecSys-Amazon', 
            name='ID-Embedding'
        )

    print(
        ' #########################################', '\n',
        '############ HyperParameters: ###########', '\n',
        '#########################################',
    )

    for key in args().__dict__.keys():
        print(key + ':', args().__dict__[key])
    print()

    TrainIDEmbeddings().workflow()
