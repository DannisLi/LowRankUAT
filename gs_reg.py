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

from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler

from sklearn.model_selection import GridSearchCV

from utils import setup_seed, Net, get_realestate_data, get_wine_data, get_abalone_data, get_bike_data


os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="realestate")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # set random seed
    setup_seed(47)

    #
    args = parse()

    # get data
    if args.dataset == "realestate":
        X, y = get_realestate_data()
    elif args.dataset == "abalone":
        X, y = get_abalone_data()
    elif args.dataset == "wine":
        X, y = get_wine_data()
    elif args.dataset == "bike":
        X, y = get_bike_data()
    

    # 
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_splits = 10
    n_jobs = 5

    print ("features:", X.shape, "\t", "targets:", y.shape)

    # lr scheduler
    warmup_epoch, total_epoch = 10, 50
    lr_lambda = lambda e: (e+1)/warmup_epoch if e+1<=warmup_epoch else 0.5*(1+np.cos(np.pi*(e+1-warmup_epoch)/(total_epoch-warmup_epoch)))

    net = NeuralNetRegressor(
        Net,
        module__in_dim = n_features,
        module__out_dim = 1,
        module__low_rank = True,
        max_epochs = total_epoch,
        batch_size = min(int(n_samples/10), 64),
        criterion = nn.MSELoss(),
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        iterator_train__shuffle = True,
        callbacks = [LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda)],
        train_split = None,
        verbose = 0,
        device = "cuda"
    )


    # 
    params = {
        'module__depth': [2, 3, 4],
        'module__width': [n*n_features for n in [4, 5, 6]],
    }
    gs = GridSearchCV(net, params, refit=True, cv=n_splits, scoring='neg_root_mean_squared_error', n_jobs=n_jobs, verbose=3)

    gs.fit(X, y)

    print (gs.cv_results_)
    print (gs.best_params_, gs.best_score_)
    

