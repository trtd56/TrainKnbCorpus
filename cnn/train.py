# -*- cording: utf-8 -*-

import random
import argparse
import gensim
import numpy as np
import pandas as pd
import chainer.functions as F
from chainer import optimizer, optimizers

from net import CNN
from str2id import Doc2id, Lab2id

parser = argparse.ArgumentParser(description='train data by CNN and fasttext')
parser.add_argument("train", type=str,
                    help="train data path")
parser.add_argument("test", type=str,
                    help="test data path")
parser.add_argument("w2v", type=str,
                    help="w2v model path")
parser.add_argument("output", type=str,
                    help="output dir path")

SEED = 0
VALID_RATIO = 0.1

N_UNITS = 512
W_DECAY = 0.001
G_CLIP = 5.0
N_EPOCH = 100
N_BATCH = 20

def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)

def load_train_data(path, test=False):
    df = pd.read_csv(path)
    docs = df["text"].tolist()
    hinshi = df["hinshi"].tolist()
    if not test:
        labs = df["label"].tolist()
        return docs, hinshi, labs

    ids = df["id"].tolist()
    return docs, hinshi, ids

def shuffle_list(l):
    rand_i = random.sample(range(len(l)), len(l))
    return [l[i] for i in rand_i]

def shuffle_train_data(doc, hinshi, label):
    rand_i = random.sample(range(len(label)), len(label))
    d = [doc[i] for i in rand_i]
    h = [hinshi[i] for i in rand_i]
    l = [label[i] for i in rand_i]
    return d, h, l

def parse_batch(batch):
    x = [x for _, x in batch]
    x1 = np.array([i[0] for i in x], dtype=np.float32)
    x2 = np.array([i[1] for i in x], dtype=np.int32)
    x = [(i, j) for i, j in zip(x1, x2)]
    t = np.array([t for t, _ in batch], dtype=np.int32)
    return x, t

def generate_bath(data, size):
    data = shuffle_list(data)
    batch = []
    for d in data:
        batch.append(d)
        if len(batch) >= size:
            yield parse_batch(batch)
            batch = []

def get_vec(doc, model, max_l, size=200):
    vec_list = []
    ignore = np.array([0.0 for _ in range(size)], dtype=np.float32)
    for d in doc.split():
        try:
            vec = model[str(d)]
        except KeyError:
            vec = ignore
        vec_list.append(vec)
    n = max_l - len(vec_list)
    if n > 0:
        for _ in range(n):
            vec_list.append(ignore)
    else:
        vec_list = vec_list[:max_l]
    return vec_list

if __name__ == "__main__":
    args = parser.parse_args()

    # load data
    docs, hinshi, labs = load_train_data(args.train)

    # split train and valid
    set_seed()
    docs, hinshi, labs = shuffle_train_data(docs, hinshi, labs)
    n_valid = int(len(docs) * VALID_RATIO)
    train_docs = docs[n_valid:]
    train_hinshi = hinshi[n_valid:]
    train_labs = labs[n_valid:]
    valid_docs = docs[:n_valid]
    valid_hinshi = hinshi[:n_valid]
    valid_labs = labs[:n_valid]

    # doc2id
    doc2id = Doc2id(train_docs)
    max_l = doc2id.get_max_l()
    lab2id = Lab2id(train_labs)
    n_label = lab2id.get_n_label()

    # hinshi2id
    hinshi2id = Doc2id(hinshi)
    n_vocab = hinshi2id.get_n_vocab()

    w2v = gensim.models.word2vec.Word2Vec.load(args.w2v)  # word2vec
    #w2v = gensim.models.KeyedVectors.load_word2vec_format(args.w2v, binary=False)  # fasttext

    train_x = [get_vec(i, w2v, max_l) for i in train_docs]
    train_t = [lab2id.lab2onehot(i) for i in train_labs]
    train_h = [hinshi2id(i) for i in train_hinshi]
    train_x = [(x, h) for x, h  in zip(train_x, train_h)]
    train = [(t, x) for t, x in zip(train_t, train_x)]

    valid_x = [get_vec(i, w2v, max_l) for i in valid_docs]
    valid_t = [lab2id.lab2onehot(i) for i in valid_labs]
    valid_h = [hinshi2id(i) for i in valid_hinshi]
    valid_x = [(x, h) for x, h  in zip(valid_x, valid_h)]
    valid = [(t, x) for t, x in zip(valid_t, valid_x)]

    mlp = CNN(n_vocab, N_UNITS, n_label)
    opt = optimizers.Adam()
    opt.setup(mlp)
    opt.add_hook(optimizer.WeightDecay(W_DECAY))
    opt.add_hook(optimizer.GradientClipping(G_CLIP))

    best_loss = 10000
    count = 0
    for epoch in range(N_EPOCH):
        # train
        t_loss = []
        for x, t in generate_bath(train, N_BATCH):
            mlp.cleargrads()
            loss = mlp(x, t, train=True)
            loss.backward()
            opt.update()
            t_loss.append(loss.data)
        t_loss = sum(t_loss)/len(t_loss)
        # valid
        x_v, t_v = parse_batch(valid)
        v_loss = mlp(x_v, t_v).data
        if best_loss > v_loss:
            best_loss = v_loss
            count = 0
            mlp.save()
        else:
            count += 1
            if count > 5:
                break
        print("{0}\t{1}\t{2}\t{3}".format(epoch, t_loss, v_loss, count))

    # test & predict
    t_docs, t_hinshi, ids = load_train_data(args.test, test=True)
    test_x = [get_vec(i, w2v, max_l) for i in t_docs]
    test_h = [hinshi2id(i) for i in t_hinshi]
    test_x = [(x, h) for x, h  in zip(test_x, test_h)]
    test = [(0, x) for x in test_x]
    x_t, _ = parse_batch(test)
    mlp.load()
    hyp = mlp(x_t).data
    h_list = []
    for h in hyp:
        # predict simgle label
        #hi = h.argmax()
        #h_list.append(lab2id.id2lab(hi))
        # predict muti label
        hi = [i for i, v in enumerate(h) if v > 0.5]
        if len(hi) == 0:
            h_list.append("None")
        else:
            h_list.append(";".join([lab2id.id2lab(i) for i in hi]))
    df = pd.DataFrame({"0":ids, "1":h_list, "2":t_docs})
    df.columns = ["id", "label", "text"]
    df.to_csv(args.output + "/hyp_cnn.csv", index=None)
