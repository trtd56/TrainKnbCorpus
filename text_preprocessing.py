# -*- cording: utf-8 -*-

import argparse
import mojimoji
import pandas as pd
from janome.tokenizer import Tokenizer

import config

parser = argparse.ArgumentParser(description='preprocessing train & test data')
parser.add_argument("train", type=str,
                    help="train data path")
parser.add_argument("test", type=str,
                    help="test data path")
parser.add_argument("sw", type=str,
                    help="Slothlib's stop word file path")
parser.add_argument("output", type=str,
                    help="output dir path")

Stop_word = []

def load_stop_word(path):
    global Stop_word
    Stop_word = Stop_word + pd.read_csv(path, header=None)[0].tolist()

def norm_text(text):
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.lower()
    return text

wakati = Tokenizer()
def get_janome_token(txt):
    txt = txt.replace("\n", " ")
    tokens = wakati.tokenize(txt)
    base_form = [i.base_form for i in tokens]
    hinshi = [i.part_of_speech.split(',')[0] for i in tokens]
    return (base_form, hinshi)

def number_filter(word):
    try:
        int(word)
        return '<NUM>'
    except ValueError:
        return word

def hinshi_filter(word, hinshi):
    if hinshi in ["名詞", "動詞"]:
        return word

def stop_word_filter(word):
    global Stop_word
    if not word in Stop_word:
        return word

def filter_ignore_words(token):
    words, hinshi = token
    text = []
    for w, h in zip(words, hinshi):
        w = number_filter(w)
        w = hinshi_filter(w, h)
        w = stop_word_filter(w)
        text.append(w)
    text = " ".join([i for i in text if not i is None])
    if len(text) > 0:
        return text
    return "<BLANK>"

def text_preprocessing(row):
    text = row["text"]
    text = norm_text(text)
    token = get_janome_token(text)
    text = filter_ignore_words(token)
    return text

def get_train_data(path):
    df = pd.read_csv(path)
    df["text"] = df.apply(text_preprocessing, axis=1)
    return df

if __name__ == "__main__":
    args = parser.parse_args()

    load_stop_word(args.sw)

    train = get_train_data(args.train)
    test = get_train_data(args.test)

    train.to_csv(args.output + "/" + config.PRE_TRAIN_FILE, index=None)
    test.to_csv(args.output + "/" + config.PRE_TEST_FILE, index=None)
