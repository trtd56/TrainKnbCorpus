# -*- coding: utf-8 -*-

import numpy as np
from chainer import Chain, Variable, serializers, initializers
import chainer.functions as F
import chainer.links as L


class CNN(Chain):

    def __init__(self, n_vocab, n_units, n_out, filter_size=(3, 4, 5), use_dropout=0.5, ignore_label=-1):
        initializer = initializers.HeNormal()
        super(CNN, self).__init__(
            word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
            conv1 = L.Convolution2D(None, n_units, (filter_size[0], 200 + n_units), stride=1, pad=(filter_size[0], 0), initialW=initializer),
            conv2 = L.Convolution2D(None, n_units, (filter_size[1], 200 + n_units), stride=1, pad=(filter_size[1], 0), initialW=initializer),
            conv3 = L.Convolution2D(None, n_units, (filter_size[2], 200 + n_units), stride=1, pad=(filter_size[2], 0), initialW=initializer),
            norm1 = L.BatchNormalization(n_units),
            norm2 = L.BatchNormalization(n_units),
            norm3 = L.BatchNormalization(n_units),
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(None, n_out),
        )
        self.filter_size = filter_size
        self.use_dropout = use_dropout

    def forward(self, x, train):
        dropout = self.use_dropout if train else 0
        x_vec = Variable(np.array([i[0] for i in x], dtype=np.float32), volatile=not train)
        x_id = Variable(np.array([i[1] for i in x], dtype=np.int32), volatile=not train)
        x_id = self.word_embed(x_id)
        x_id = F.dropout(x_id, ratio=dropout)
        x = F.concat((x_vec, x_id), axis=2)
        x = F.expand_dims(x, axis=1)
        x1 = F.relu(self.norm1(self.conv1(x), test=not train))
        x1 = F.max_pooling_2d(x1, self.filter_size[0])
        x2 = F.relu(self.norm2(self.conv2(x), test=not train))
        x2 = F.max_pooling_2d(x2, self.filter_size[1])
        x3 = F.relu(self.norm3(self.conv3(x), test=not train))
        x3 = F.max_pooling_2d(x3, self.filter_size[2])
        x = F.concat((x1, x2, x3), axis=2)
        x = F.dropout(F.relu(self.l1(x)), ratio=dropout)
        x = self.l2(x)
        return x

    def __call__(self, x, t=None, train=False):
        y = self.forward(x, train)
        if t is None:
            return F.sigmoid(y)
        else:
            return F.sigmoid_cross_entropy(y, t)

    def save(self, path="./model/"):
        serializers.save_npz(path + "cnn_model.npz", self)

    def load(self, path="./model/"):
        serializers.load_npz(path + "cnn_model.npz", self)
