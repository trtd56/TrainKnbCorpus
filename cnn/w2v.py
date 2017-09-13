# -* coding: utf-8 -*-

import argparse
import pandas as pd
from gensim.models import word2vec

parser = argparse.ArgumentParser(description='generate wikipedia corpus for fasttext')
parser.add_argument("wiki", type=str,
                    help="sep wiki data path")
parser.add_argument("out", type=str,
                    help="output dir path")

N_VEC = 200
MIN_COUNT = 5
WINDOW = 15
WORKERS = 3

if __name__ == '__main__':
    sentences = word2vec.Text8Corpus(args.wiki)
    model = word2vec.Word2Vec(sentences, size=N_VEC, min_count=MIN_COUNT, window=WINDOW, workers=WORKERS)
    model.save(args.out + "/wiki_word2vec.model")
