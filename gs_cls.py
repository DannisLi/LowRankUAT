# coding: utf8
'''
grid search depth and width of classical feedforward network
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler

from sklearn.model_selection import GridSearchCV

from utils import setup_seed, get_iris_data, get_rice_data, get_bank_data, get_adult_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iris")
    args = parser.parse_args()
    return args


class Net(nn.Module):
    def __init__(self, in_dim:int, out_dim:int=1, depth:int=4, width:int=100, low_rank=False):
        super().__init__()
        self.width = width
        if low_rank:
            layers = [nn.Linear(in_dim, width), nn.SiLU()]
            for _ in range(depth-1):
                layers.append(nn.Linear(width, int(width/3), bias=False))
                layers.append(nn.Linear(int(width/3), width))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(width, out_dim))
            self.net = nn.Sequential(*layers)
            del layers
        else:
            layers = [nn.Linear(in_dim, width), nn.SiLU()]
            for _ in range(depth-1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(width, out_dim))
            self.net = nn.Sequential(*layers)
            del layers

    def forward(self, X):
        X = self.net(X)
        return X



if __name__ == "__main__":
    # set random seed
    setup_seed(521)

    #
    args = parse()

    # get data
    if args.dataset == "iris":
        X, y = get_iris_data()
    elif args.dataset == "rice":
        X, y = get_rice_data()
    elif args.dataset == "bank":
        X, y = get_bank_data()
    elif args.dataset == "adult":
        X, y = get_adult_data()
    

    # 
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = np.max(y) + 1
    n_splits = 10
    n_jobs = 5

    print ("features:", X.shape, "\t", "targets:", y.shape)

    # lr scheduler
    warmup_epoch, total_epoch = 10, 50
    lr_lambda = lambda e: (e+1)/warmup_epoch if e+1<=warmup_epoch else 0.5*(1+np.cos(np.pi*(e+1-warmup_epoch)/(total_epoch-warmup_epoch)))

    net = NeuralNetClassifier(
        Net,
        module__in_dim = n_features,
        module__out_dim = n_classes,
        module__low_rank = False,
        max_epochs = total_epoch,
        batch_size = min(int(n_samples/10), 64),
        criterion = nn.CrossEntropyLoss(),
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        iterator_train__shuffle = True,
        callbacks = [LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda)],
        train_split = None,
        verbose = 0,
        device="cuda"
    )


    # 
    params = {
        'module__depth': [2, 3, 4],
        'module__width': [n*n_features for n in [4, 5, 6]],
    }
    gs = GridSearchCV(net, params, refit=False, cv=n_splits, scoring='accuracy', n_jobs=n_jobs, verbose=3)
    gs.fit(X, y)
    print (gs.best_params_, gs.best_score_)

