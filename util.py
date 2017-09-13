# -*- cording: utf-8 -*-

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import config

def init_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)

def load_train_data(path):
    df = pd.read_csv(path)
    text = [i for i in df["text"].tolist()]
    label = [i for i in df["label"].tolist()]
    return label, text

def evaluate_logistic_regressioni(feature, label, c):
    # split data
    train_x, valid_x, train_t, valid_t = train_test_split(feature, label, test_size=config.VALID_RATIO, random_state=config.SEED)
    # train
    model = LogisticRegression(C=c)
    model.fit(train_x, train_t)
    # evaluate validation data
    predict = model.predict(valid_x)
    _, _, f, _ = precision_recall_fscore_support(valid_t, predict, beta=0.5)
    score = np.mean(f)
    return score
