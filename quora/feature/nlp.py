# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from quora.feature.feature import ClassicalFeature
from quora.util.represenation.tfidf import TFIDF
from quora.util.represenation.vector import W2V
from quora.core.multip import map_reduce


class KeyWordDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.idf = TFIDF.load()
        self.w2v = W2V.load()
        self.extract_stage = [self._extract]
        self.idf_vocab = self.idf.get_feature_names()

    def calculate_keyword(self, q, count=3):
        csr = self.idf.transform([q])
        index = csr.indices.tolist()
        values = csr.data.tolist()
        keyword_index = sorted(zip(index, values), key=lambda x: -x[1])
        keywords = [self.idf_vocab[i[0]] for i in keyword_index]
        keywords = [word for word in keywords if word in self.w2v]
        return keywords[:min(count, len(keywords))]

    def calculate_vector(self, keywords):
        vectors = np.array([self.w2v[word] for word in keywords])
        vector = np.mean(vectors, axis=0)
        return vector

    def calculate(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        q1_keywords = self.calculate_keyword(q1)
        q2_keywords = self.calculate_keyword(q2)
        if not len(q1_keywords) or not len(q2_keywords):
            return [0.0, 0.0, 0.0]
        q1_vector = self.calculate_vector(q1_keywords)
        q2_vector = self.calculate_vector(q2_keywords)
        cos_sim = 1 - cosine(q1_vector, q2_vector)
        euclidean_sim = 1 - euclidean(q1_vector, q2_vector)
        manhattan_sim = 1 - cityblock(q1_vector, q2_vector)
        return [cos_sim, euclidean_sim, manhattan_sim]

    def _extract(self):
        columns = ['KeyWord_cosine', 'KeyWord_euclidean', 'KeyWord_manhattan']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class TFIDFDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.idf = TFIDF.load()
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        q1_idf = self.idf.transform([q1])[0]
        q2_idf = self.idf.transform([q2])[0]
        tf_idf_distance = 1 - cosine(q1_idf.toarray(), q2_idf.toarray())
        q1_tf_idf_sum = np.sum(q1_idf.data)
        q2_tf_idf_sum = np.sum(q2_idf.data)
        q1_tf_idf_mean = np.mean(q2_idf.data)
        q2_tf_idf_mean = np.mean(q2_idf.data)
        return [tf_idf_distance, q1_tf_idf_sum, q2_tf_idf_sum, q1_tf_idf_mean, q2_tf_idf_mean]

    def _extract(self):
        columns = ['TFIDFDistance', 'TFIDFSum_1', 'TFIDFSum_2', 'TFIDFMean_1', 'TFIDFMean_2']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class NGramTFIDFDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.idf = TFIDF.load()
        self.extract_stage = [self._extract]
        self.idf_vocab = self.idf.get_feature_names()

    def calculate_gram(self, seq, n):
        seq = [word for word in seq if word in self.idf_vocab]
        n_gram = [seq[i: n+i] for i in range(len(seq)) if n+i <= len(seq)]
        if not n_gram: n_gram = [seq]
        return n_gram

    def calculate_sim(self, seq1, seq2):
        seq1_idfs = [self.idf.transform([' '.join(seq)])[0] for seq in seq1]
        seq2_idfs = [self.idf.transform([' '.join(seq)])[0] for seq in seq2]
        sims = [1 - cosine(seq1_idf.toarray(), seq2_idf.toarray())
                for seq1_idf in seq1_idfs for seq2_idf in seq2_idfs]
        seq1_sim = np.mean([max(sims[i:i+len(seq2_idfs)]) for i in range(0, len(sims), len(seq2_idfs))])
        seq2_sim = np.mean([max(sims[i::len(seq2_idfs)]) for i in range(len(seq2_idfs))])
        sim = np.mean([seq1_sim, seq2_sim])
        return sim

    def calculate_gram_sim(self, seq1, seq2, n):
        n_gram_q1 = self.calculate_gram(seq1, n)
        n_gram_q2 = self.calculate_gram(seq2, n)
        n_gram_sim = self.calculate_sim(n_gram_q1, n_gram_q2)
        return n_gram_sim

    def calculate(self, row):
        q1 = str(row['question1']).split()
        q2 = str(row['question2']).split()
        one_gram_sim = self.calculate_gram_sim(q1, q2, 1)
        two_gram_sim = self.calculate_gram_sim(q1, q2, 2)
        three_gram_sim = self.calculate_gram_sim(q1, q2, 3)
        return [one_gram_sim, two_gram_sim, three_gram_sim]

    def _extract(self):
        columns = ['NGramTFIDF_one', 'NGramTFIDF_two', 'NGramTFIDF_three']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class TFIDFDimReduce(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.idf = TFIDF.load()
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        q1_idf = self.idf.transform([q1])
        q2_idf = self.idf.transform([q2])

    def _extract(self):
        data = {}
        columns = ['TFIDFDistance']
        for _, row in self.data.iterrows():
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


if __name__ == '__main__':
    NGramTFIDFDistance().extract()
