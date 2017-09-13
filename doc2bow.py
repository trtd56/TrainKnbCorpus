# -*- cording: utf-8 -*-

import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

import config
from util import init_seed, load_train_data, evaluate_logistic_regressioni

parser = argparse.ArgumentParser(description='parameter search')
parser.add_argument("train", type=str,
                    help="train data path")
parser.add_argument("out", type=str,
                    help="output dir path")

MAX_DF = 0.5
MIN_DF = 2
N_COMPONENTS = 300
C = 1


class Doc2bow():
    __COUNT_VECTORIZER_FILE = "count_vectorizer.pkl"
    __PCA_FILE = "pca.pkl"

    def __init__(self):
        pass

    def set_param(self, max_df, min_df, n_components):
        self.count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
        self.pca = PCA(n_components=n_components)

    def fit(self, text):
        # Fit Bow
        self.count_vectorizer.fit(text)
        bow = self.count_vectorizer.transform(text)
        bow = bow.toarray()
        # Fit PCA
        self.pca.fit(bow)

    def transform(self, text):
        bow = self.count_vectorizer.transform(text)
        bow = bow.toarray()
        feature = self.pca.transform(bow)
        return feature

    def save(self, dir_path):
        joblib.dump(self.count_vectorizer, dir_path + "/" + self.__COUNT_VECTORIZER_FILE)
        joblib.dump(self.pca, dir_path + "/" + self.__PCA_FILE)

    def load(self, dir_path):
        self.count_vectorizer = joblib.load(dir_path + "/" + self.__COUNT_VECTORIZER_FILE)
        self.pca = joblib.load(dir_path + "/" + self.__PCA_FILE)

if __name__ == "__main__":
    args = parser.parse_args()
    out_path = args.out
    init_seed()

    # load train data
    label, text = load_train_data(args.train)

    doc2bow = Doc2bow()
    doc2bow.set_param(MAX_DF, MIN_DF, N_COMPONENTS)
    doc2bow.fit(text)
    vec = doc2bow.transform(text)
    f = evaluate_logistic_regressioni(vec, label, C)
    print("max_df:{0} , min_df: {1}, n_components: {2}, C: {3}\n[f_score] -> {4}".format(MAX_DF, MIN_DF, N_COMPONENTS, C, f))
