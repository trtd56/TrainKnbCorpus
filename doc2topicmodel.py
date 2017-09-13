# -*- cording: utf-8 -*-

import argparse
from gensim import corpora, models, matutils

import config
from util import init_seed, load_train_data, evaluate_logistic_regressioni

parser = argparse.ArgumentParser(description='parameter search')
parser.add_argument("train", type=str,
                    help="train data path")
parser.add_argument("out", type=str,
                    help="output dir path")

MAX_DF = 0.5
MIN_DF = 1
N_TOPIC = 100
MODE = "lda"
C = 1

class Doc2TopicModel():
    __DICT_FILE = "word_id_dict.txt"

    def __init__(self):
        pass

    def set_param(self, max_df, min_df, n_topic, mode):
        self.max_df = max_df
        self.min_df = min_df
        self.n_topic = n_topic
        if not mode in ["lsi", "lda"]:
            raise Exception("unknown mode.\nmode is 'lsi' or 'lda'")
        self.mode = mode

    def fit(self, text):
        wakati = [i.split() for i in text]
        # make word dictionary
        self.dictionary = corpora.Dictionary(wakati)
        self.dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df)
        # BoW
        bow_corpus = [self.dictionary.doc2bow(i) for i in wakati]
        # TF-IDF
        self.tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = self.tfidf[bow_corpus]
        # train topic model
        if self.mode == "lsi":
            self.model = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=self.n_topic)
        elif self.mode == "lda":
            self.model = models.LdaModel(corpus_tfidf, id2word=self.dictionary, num_topics=self.n_topic)

    def transform(self, text):
        wakati = [i.split() for i in text]
        bow_corpus = [self.dictionary.doc2bow(i) for i in wakati]
        corpus_tfidf = self.tfidf[bow_corpus]
        corpus_model = self.model[corpus_tfidf]
        feature = matutils.corpus2dense(corpus_model, num_terms=self.n_topic).T
        return feature

    def save(self, dir_path):
        self.dictionary.save_as_text(out_path + "/" + self.__DICT_FILE)
        self.model.save(out_path + "/" + self.mode + ".pkl")

    def load(self, dir_path):
        self.count_vectorizer = joblib.load(dir_path + "/" + self.__COUNT_VECTORIZER_FILE)
        self.pca = joblib.load(dir_path + "/" + self.__PCA_FILE)

if __name__ == "__main__":
    args = parser.parse_args()
    out_path = args.out
    init_seed()

    # load train data
    label, text = load_train_data(args.train)

    doc2lda = Doc2TopicModel()
    doc2lda.set_param(MAX_DF, MIN_DF, N_TOPIC, MODE)
    doc2lda.fit(text)
    vec = doc2lda.transform(text)
    f = evaluate_logistic_regressioni(vec, label, C)
    print("max_df:{0} , min_df: {1}, n_topic: {2}, C: {3}\n[f_score] -> {4}".format(MAX_DF, MIN_DF, N_TOPIC, C, f))
