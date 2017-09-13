# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import config

parser = argparse.ArgumentParser(description='generate train & test data from KNB corpus')
parser.add_argument("input", type=str, help="input KNB corpus2 dir path")
parser.add_argument("output", type=str, help="output dir path")

def load_data(corpus_path):
    x = []
    t = []
    for k, path in get_path_data(corpus_path):
        df = pd.read_csv(path, encoding="EUC-JP", delimiter="\t", header=None)
        x.extend([i for i in df[1].tolist()])
        t.extend([k for _ in df[1].tolist()])
    train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=config.TEST_RATIO, random_state=config.SEED)
    return to_df(train_x, train_t, "train"), to_df(test_x, test_t, "test")

def get_path_data(corpus_path):
    files = os.listdir(corpus_path)
    keys = [f.split(".")[0] for f in files]
    files_full_path = glob.glob(corpus_path + "/*")
    return [(k, v) for k, v in zip(keys, files_full_path)]

def to_df(x, t, dtype):
    ids = ["{0}_{1:04d}".format(dtype, i) for i in range(len(x))]
    df = pd.DataFrame([(i, tt, xx) for i, xx, tt in zip(ids, x, t)])
    df.columns = ["id", "label", "text"]
    return df

if __name__ == '__main__':

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    train, test_ans = load_data(input_path)
    test = test_ans.loc[:,['id','text']]

    train.to_csv(output_path + "/" + config.TRAIN_FILE, index=False, encoding="utf-8")
    test.to_csv(output_path + "/" + config.TEST_FILE, index=False, encoding="utf-8")
    test_ans.to_csv(output_path + "/" + config.ANSWER_FILE, index=False, encoding="utf-8")
