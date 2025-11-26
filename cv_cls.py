# coding: utf8

import os
import sys
import argparse
from tabnanny import verbose
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler

from sklearn.model_selection import GridSearchCV

from scipy.stats import ttest_rel

from utils import setup_seed, Net, get_iris_data, get_rice_data, get_bank_data, get_adult_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iris")
    parser.add_argument('--depth', type=int)
    parser.add_argument('--width', type=int)
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    # set random seed
    setup_seed()

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
        module__depth = args.depth,
        module__width = args.width,
        max_epochs = total_epoch,
        batch_size = min(int(n_samples/10), 64),
        criterion = nn.CrossEntropyLoss(),
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        iterator_train__shuffle = True,
        # callbacks = [EpochScoring(scoring='accuracy', lower_is_better=False), LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda)],
        callbacks = [LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda)],
        train_split = None,
        verbose = 0,
        device="cuda"
    )


    # 
    params = {
        'module__low_rank': [False, True],
    }
    gs = GridSearchCV(net, params, refit=False, cv=n_splits, scoring='accuracy', n_jobs=n_jobs, verbose=3)

    gs.fit(X, y)
    result = gs.cv_results_
    print (result)

    accs_classical = []
    accs_low = []
    for i in range(n_splits):
        score = result[f"split{i}_test_score"]
        accs_classical.append(score[0])
        accs_low.append(score[1])
    
    print (ttest_rel(accs_classical, accs_low, alternative="greater"))

