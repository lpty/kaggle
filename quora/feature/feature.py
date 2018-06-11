# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from quora import config
from quora.util.represenation.vector import W2V, Glove


class ClassicalFeature(object):

    def __init__(self, mode):
        file = config.clean_train_file if mode == 'train' else config.clean_test_file
        self.extract_stage = []
        self.data = pd.read_csv(file, index_col=0)
        self.feature = config.feature_path.format(mode+'_'+self.__class__.__name__)
        self.mode = mode
        self.split = False

    def extract(self):
        for stage in self.extract_stage:
            stage()
        self.save()

    def save(self):
        if self.mode == 'test':
            columns = ['question1', 'question2']
        else:
            columns = ['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
        self.data = self.data.drop(columns, axis=1)
        if self.split:
            columns = self.data.columns.tolist()[1:]
            for column in columns:
                index = columns.index(column)
                drop_columns = columns[:index] + columns[index+1:]
                data = self.data.drop(drop_columns, axis=1)
                data.to_csv(self.feature+'_'+column, index=False)
        else:
            self.data.to_csv(self.feature, index=False)

    @classmethod
    def load(cls, mode='train'):
        files = sorted([f for f in os.listdir(config.feature_path[:-2]) if f.startswith(mode)])
        files = [f for f in files if f.replace(mode+'_', '') in config.feature_columns]
        print(files)
        features = [pd.read_csv(config.feature_path.format(f)).fillna(0) for f in files]
        return cls.merge(features, mode)

    @classmethod
    def merge(cls, features, mode):
        assert len(features), 'Feature not exits.'
        feature = features[0]
        on_id = 'id' if mode == 'train' else 'test_id'
        for f in features[1:]:
            feature = pd.merge(feature, f, how='left', on=on_id)
        feature = feature.drop([on_id], axis=1)
        return feature


class ClassicalBatchFeature(object):

    def __init__(self, mode='train'):
        self.mode = mode
        self.batch_size = 200
        self.features = ClassicalFeature.load(mode=self.mode).values
        if self.mode == 'train':
            self.labels = pd.read_csv(config.origin_train_file)['is_duplicate'].values
        else:
            self.labels = []

    def __iter__(self):
        count = 0
        while True:
            x = self.features[count*self.batch_size:(count+1)*self.batch_size]
            x = [[xx] for xx in x]
            y = self.labels[count*self.batch_size:(count+1)*self.batch_size]
            tensor_x = torch.Tensor(x)
            tensor_y = torch.LongTensor(y)
            yield tensor_x, tensor_y
            if (count+1) * self.batch_size >= len(self.labels):
                break
            count += 1


class MatchPyramidFeature(object):

    def __init__(self, mode='train'):
        if mode == 'train':
            # self.file = config.clean_train_file
            self.file = config.origin_train_file
        else:
            # self.file = config.clean_test_file
            self.file = config.origin_test_file
        self.mode = mode
        self.data = pd.read_csv(self.file)
        self.batch_size = config.model_params['MatchPyramid']['batch_size']
        # self.w2v = W2V.load()
        self.w2v = Glove.load()

    def __iter__(self):
        matrix_x, matrix_y, max_height, max_width, count = [], [], 0, 0, 0
        for row_index, row in self.data.iterrows():
            seq1 = [self.w2v[w] for w in str(row['question1']).split() if w in self.w2v]
            seq2 = [self.w2v[w] for w in str(row['question2']).split() if w in self.w2v]
            seq1 = seq1 if len(seq1) else [np.zeros((300,))]
            seq2 = seq2 if len(seq2) else [np.zeros((300,))]
            max_height = max(len(seq2), max_height)
            max_width = max(len(seq1), max_width)
            matrix = np.array([[np.dot(s1, s2) for s1 in seq1] for s2 in seq2])
            matrix_x.append(matrix)
            if self.mode == 'train':
                matrix_y.append(int(row['is_duplicate']))
            count += 1
            if count == self.batch_size or row_index+1 == self.data.shape[0]:
                yield self.pack(matrix_x, matrix_y, max_height, max_width)
                matrix_x, matrix_y, max_height, max_width, count = [], [], 0, 0, 0

    @staticmethod
    def pack(matrix_x, matrix_y, max_height, max_width):
        tensor_x = []
        for matrix in matrix_x:
            height, width = matrix.shape
            height_bias = (max_height - height) // 2
            width_bias = (max_width - width) // 2
            matrix = np.pad(matrix, ((height_bias, max_height - height - height_bias),
                                     (width_bias, max_width - width - width_bias)), 'constant', constant_values=(0,))
            tensor_x.append([matrix])
        tensor_x = torch.Tensor(tensor_x)
        tensor_y = torch.LongTensor(matrix_y)
        return tensor_x, tensor_y


class LSTMFeature(object):

    def __init__(self, mode='train'):
        if mode == 'train':
            self.file = config.clean_train_file
        else:
            self.file = config.clean_test_file
        self.mode = mode
        self.data = pd.read_csv(self.file)
        self.batch_size = config.model_params['LSTM']['batch_size']
        self.w2v = W2V.load()

    def __iter__(self):
        matrix_x_1, matrix_x_2, matrix_y, count = [], [], [], 0
        for row_index, row in self.data.iterrows():
            seq1 = [self.w2v[w] for w in str(row['question1']).split() if w in self.w2v]
            seq2 = [self.w2v[w] for w in str(row['question2']).split() if w in self.w2v]
            matrix_x_1.append(seq1 if len(seq1) else [np.zeros((300,))])
            matrix_x_2.append(seq2 if len(seq2) else [np.zeros((300,))])
            if self.mode == 'train':
                matrix_y.append(int(row['is_duplicate']))
            count += 1
            if count == self.batch_size or row_index+1 == self.data.shape[0]:
                tensor_x_1_index, tensor_x_1 = self.pack(matrix_x_1)
                tensor_x_2_index, tensor_x_2 = self.pack(matrix_x_2)
                tensor_y = torch.LongTensor(matrix_y)
                yield tensor_x_1_index, tensor_x_1, tensor_x_2_index, tensor_x_2, tensor_y
                matrix_x_1, matrix_x_2, matrix_y, count = [], [], [], 0

    @staticmethod
    def pack(matrix_x):
        m = sorted([[i, len(v), v] for i, v in enumerate(matrix_x)], key=lambda x: -x[1])
        m_index = [i for i, _, _ in m]
        m_length = torch.LongTensor([l for _, l, _ in m])
        m_value = torch.Tensor([v + [np.zeros((300,)) for _ in range(m_length[0] - len(v))] for _, _, v in m])
        m_pack = pack_padded_sequence(m_value, m_length, batch_first=True)
        return m_index, m_pack


if __name__ == '__main__':
    f = MatchPyramidFeature()
    c = 0
    for i in f:
        c += len(i[-1])
        print(c)
