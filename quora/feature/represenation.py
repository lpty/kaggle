# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from quora.core.multip import map_reduce
from quora.util.similarity.wmd import wmd
from quora.util.represenation.vector import W2V
from quora.util.represenation.vector import D2V
from quora.util.represenation.tfidf import TFIDF
from quora.feature.feature import ClassicalFeature


class Word2VecDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]
        self.model = W2V.load()

    def calculate(self, row):
        seq1 = str(row['question1']).split()
        seq2 = str(row['question2']).split()
        seq1 = [word for word in seq1 if word in self.model]
        seq2 = [word for word in seq2 if word in self.model]
        if len(seq1) == 0 or len(seq2) == 0:
            return [0.0, 0.0, 0.0]
        vec_seq1 = [self.model[x] for x in seq1]
        vec_seq2 = [self.model[x] for x in seq2]
        vec_seq1 = np.array(vec_seq1).mean(axis=0)
        vec_seq2 = np.array(vec_seq2).mean(axis=0)
        cos_sim = 1 - cosine(vec_seq1, vec_seq2)
        euclidean_sim = 1 - euclidean(vec_seq1, vec_seq2)
        manhattan_sim = 1 - cityblock(vec_seq1, vec_seq2)
        return [cos_sim, euclidean_sim, manhattan_sim]

    def _extract(self):
        columns = ['W2V_cosine', 'W2V_euclidean', 'W2V_manhattan']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class WordMoverDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]
        self.model = W2V.load()

    def calculate(self, row):
        return 1 - wmd(str(row['question1']).split(), str(row['question2']).split(), self.model)

    def _extract(self):
        self.data['WordMoverDistance'] = [self.calculate(row) for _, row in self.data.iterrows()]


class W2VWeightDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.idf = TFIDF.load()
        self.w2v = W2V.load()
        self.extract_stage = [self._extract]
        self.idf_vocab = self.idf.get_feature_names()

    def calculate_keyword(self, q):
        csr = self.idf.transform([q])
        index = csr.indices.tolist()
        values = csr.data.tolist()
        keyword_index = zip(index, values)
        keywords = [(self.idf_vocab[i[0]], i[1]) for i in keyword_index]
        keywords = [word for word in keywords if word[0] in self.w2v]
        return keywords

    def calculate_vector(self, keywords):
        vectors = np.array([self.w2v[word[0]]*word[1] for word in keywords])
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
        columns = ['W2VWeight_cosine', 'W2VWeight_euclidean', 'W2VWeight_manhattan']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class NGramW2VDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.w2v = W2V.load()
        self.extract_stage = [self._extract]

    def calculate_gram(self, seq, n):
        seq = [word for word in seq if word in self.w2v]
        n_gram = [seq[i: n+i] for i in range(len(seq)) if n+i <= len(seq)]
        if not n_gram: n_gram = [seq]
        return n_gram

    def calculate_sim(self, seq1, seq2):
        seq1_w2vs = [np.mean([self.w2v[word] for word in seq], axis=0) for seq in seq1]
        seq2_w2vs = [np.mean([self.w2v[word] for word in seq], axis=0) for seq in seq2]
        sims = [1 - cosine(seq1_w2v, seq2_w2v) for seq1_w2v in seq1_w2vs for seq2_w2v in seq2_w2vs]
        seq1_sim = np.mean([max(sims[i:i+len(seq2_w2vs)]) for i in range(0, len(sims), len(seq2_w2vs))])
        seq2_sim = np.mean([max(sims[i::len(seq2_w2vs)]) for i in range(len(seq2_w2vs))])
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
        columns = ['NGramW2V_one', 'NGramW2V_two', 'NGramW2V_three']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class Doc2VectorDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]
        self.model = D2V.load()

    def calculate(self, row):
        seq1 = str(row['question1']).split()
        seq2 = str(row['question2']).split()
        vec_seq1 = self.model.infer_vector(seq1)
        vec_seq2 = self.model.infer_vector(seq2)
        cos_sim = 1 - cosine(vec_seq1, vec_seq2)
        return cos_sim

    def _extract(self):
        self.data['Doc2VectorDistance'] = [self.calculate(row) for _, row in self.data.iterrows()]


if __name__ == '__main__':
    W2VWeightDistance(mode='test').extract()
