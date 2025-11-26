
import os
import random
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_data(X, y, task="classification"):
    X = pd.get_dummies(X)
    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = y.astype(np.int64)
    elif task == "regression":
        y = y.to_numpy().reshape(-1,1)
        y = minmax_scaler.fit_transform(y)
        y = y.astype(np.float32)
    X = X.astype(np.float32)
    return X, y


# classification datasets
def get_iris_data():
    data = fetch_ucirepo(id=53)
    df = data.data.original
    X = df[df.columns.difference(["class"])]
    y = df["class"]
    X, y = preprocess_data(X, y, "classification")
    return X, y

def get_rice_data():
    data = fetch_ucirepo(id=545)
    df = data.data.original
    X = df[df.columns.difference(["Class"])]
    y = df["Class"]
    X, y = preprocess_data(X, y, "classification")
    return X, y


def get_bank_data():
    data = fetch_ucirepo(id=222)
    df = data.data.original
    X = df[df.columns.difference(["y"])]
    y = df["y"]
    X, y = preprocess_data(X, y, "classification")
    return X, y


def get_adult_data():
    data = fetch_ucirepo(id=2)
    df = data.data.original
    X = df[df.columns.difference(["income"])]
    y = df["income"]
    y = y.apply(lambda s: s.strip().strip('.'))
    X, y = preprocess_data(X, y, "classification")
    return X, y




# regression datasets
def get_realestate_data():
    data = fetch_ucirepo(id=477)
    df = data.data.original
    del df["No"]
    X = df[df.columns.difference(["Y house price of unit area"])]
    y = df["Y house price of unit area"]
    X, y = preprocess_data(X, y, "regression")
    return X, y

def get_wine_data():
    data = fetch_ucirepo(id=186)
    df = data.data.original
    X = df[df.columns.difference(["quality"])]
    y = df["quality"]
    X, y = preprocess_data(X, y, "regression")
    return X, y


def get_abalone_data():
    data = fetch_ucirepo(id=1)
    df = data.data.original
    X = df[df.columns.difference(["Rings"])]
    y = df["Rings"]
    X, y = preprocess_data(X, y, "regression")
    return X, y


def get_bike_data():
    data = fetch_ucirepo(id=275)
    df = data.data.original
    del df["instant"], df["dteday"], df["casual"], df["registered"]
    X = df[df.columns.difference(["cnt"])]
    y = df["cnt"]
    X, y = preprocess_data(X, y, "regression")
    return X, y


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
    X, y = get_adult_data()
    print (X.shape, y.shape)

    # print (X)
    # print (np.max(y), np.min(y))
