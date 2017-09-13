# -*- cording: utf-8 -*-

from itertools import chain
from collections import defaultdict

class Doc2id():
    __MAX_COUNT = 256000
    __MIN_COUNT = 1
    __IGNORE = -1

    def __init__(self, docs):
        # word to id
        words = list(chain.from_iterable([i.split() for i in docs]))
        w_freq = defaultdict(int)
        for w in words:
            w_freq[w] += 1
        w_freq = {w: f for w, f in w_freq.items() if self.__MIN_COUNT <= f <= self.__MAX_COUNT}
        self.__n_vocab = len(w_freq) + 1

        self.w_dic = defaultdict(int)
        for i, v in enumerate(w_freq.keys()):
            self.w_dic[v] += i + 1

        # sentence max
        self.max_l = max(len(i.split()) for i in docs)

    def padding(self, arr):
        n = self.max_l - len(arr)
        for _ in range(n):
            arr.append(self.__IGNORE)
        return arr

    def __call__(self, doc):
        doc = doc.split()
        idx = [self.w_dic[d] for d in doc]
        idx = idx[:self.max_l]
        return self.padding(idx)

    def get_n_vocab(self):
        return self.__n_vocab

    def get_max_l(self):
        return self.max_l

class Lab2id():

    def __init__(self, labs):
        self.l_dic = defaultdict(int)
        for i, v in enumerate(sorted(list(set(labs)))):
            self.l_dic[v] = i
        self.i_dic = {v:k for k, v in self.l_dic.items()}
        self.lab_size = len(self.l_dic)

    def lab2id(self, lab):
        return self.l_dic[lab]

    def id2lab(self, i):
        return self.i_dic[i]

    def lab2onehot(self, lab):
        index = self.lab2id(lab)
        return [1 if i == index else 0 for i in range(self.lab_size)]

    def get_n_label(self):
        return self.lab_size
