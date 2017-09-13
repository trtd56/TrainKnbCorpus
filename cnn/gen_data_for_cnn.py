# -*- cording: utf-8 -*-

import argparse
import pandas as pd

from wiki_sep_by_janome import wakati_janome, wakati


parser = argparse.ArgumentParser(description='generate train data from KNB corpus')
parser.add_argument("train", type=str,
                    help="train data path")
parser.add_argument("test", type=str,
                    help="test data path")
parser.add_argument("output", type=str,
                    help="output dir path")


def get_hinshi(txt):
    txt = txt.replace("\n", " ")
    tokens = wakati.tokenize(txt)
    return " ".join([i.part_of_speech.split(',')[0] for i in tokens])

def get_train_data(path, mode):
    df = pd.read_csv(path)
    df_list = []
    for _, row in df.iterrows():
        sep_text = wakati_janome(row["text"])
        hinshi = get_hinshi(row["text"])
        if mode == "train":
            new_row = (row["id"], row["label"], sep_text, hinshi)
        elif mode == "test":
            new_row = (row["id"], sep_text, hinshi)
        df_list.append(new_row)
    df_list = pd.DataFrame(df_list)
    if mode == "train":
        df_list.columns = ["id", "label", "text", "hinshi"]
    elif mode == "test":
        df_list.columns = ["id", "text", "hinshi"]
    return df_list

if __name__ == "__main__":
    args = parser.parse_args()

    train = get_train_data(args.train, "train")
    test = get_train_data(args.test, "test")

    train.to_csv(args.output + "/train_data.csv", index=None)
    test.to_csv(args.output + "/test_data.csv", index=None)
