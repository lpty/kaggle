# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import math
import distance
import pandas as pd
from itertools import combinations_with_replacement
from quora import config
from quora.core.multip import map_reduce
from quora.feature.feature import ClassicalFeature
from quora.util.similarity.strike_a_match import strike_a_match
from quora.core.const import Punctuations, StopWords


class Concurrence(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in StopWords.StopWordsEN:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).split():
            if word not in StopWords.StopWordsEN:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            return 0.
        else:
            return 1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol

    def _extract(self):
        self.data['Concurrence'] = [self.calculate(row) for _, row in self.data.iterrows()]


class ConcurrenceTFIDF(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self.init_idf,
                              self._extract]

    def init_idf(self):
        doc_nums = len(self.data)
        idf = {}
        for _, row in self.data.iterrows():
            doc = str(row['question1']).split() + str(row['question2']).split()
            for word in doc:
                idf[word] = idf.get(word, 0) + 1
        for word in idf:
            idf[word] = math.log(doc_nums / idf[word])
        self.idf = idf

    def calculate(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).split():
            if word not in StopWords.StopWordsEN:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).split():
            if word not in StopWords.StopWordsEN:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] * self.idf.get(w, 0) for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] * self.idf.get(w, 0) for w in q2words if w in q1words])
        n_tol = sum([q1words[w] * self.idf.get(w, 0) for w in q1words]) + \
                sum([q2words[w] * self.idf.get(w, 0) for w in q2words])
        if 1e-6 > n_tol:
            return 0.
        else:
            return 1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol

    def _extract(self):
        self.data['ConcurrenceTFIDF'] = [self.calculate(row) for _, row in self.data.iterrows()]


class SpecialConcurrence(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._init_prob, self._extract]
        self.filter_words = list(Punctuations.PUNCTUATIONS) + list(StopWords.StopWordsEN)

    def _init_prob(self):
        prob = {}
        data = pd.read_csv(config.clean_train_file, index_col=0)
        for _, row in data.iterrows():
            q1 = set([word for word in str(row['question1']).split() if word not in self.filter_words])
            q2 = set([word for word in str(row['question2']).split() if word not in self.filter_words])
            special_words = list(q1 & q2)
            special_label = str(row['is_duplicate'])
            for word in special_words:
                prob[word] = prob.get(word, {'0': 0, '1': 0})
                prob[word][special_label] += 1
        for word, word_count in prob.items():
            prob[word] = int(word_count['1']) / (int(word_count['1'] + int(word_count['0'])))
        self.prob = prob

    def calculate(self, row):
        q1 = set([word for word in str(row['question1']).split() if word not in self.filter_words])
        q2 = set([word for word in str(row['question2']).split() if word not in self.filter_words])
        special_words = list(q1 & q2)
        prob = 1
        for word in special_words:
            prob = prob * self.prob.get(word, 1)
        return [prob]

    def _extract(self):
        columns = ['SpecialConcurrence']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class InterrogativeWord(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]
        # uni_words = ['what', 'whi', 'which', 'how', 'where', 'when', 'if', 'can', 'should']
        # do_words = ['doe', 'do', 'did']
        # be_words = ['is', 'are']
        # will_words = ['will', 'would']
        # self.words = uni_words + do_words + be_words + will_words
        self.words = ['what', 'whi', 'which', 'how', 'where', 'when']
        self.columns = ['_'.join(word) for word in combinations_with_replacement(self.words, 2)]

    def count(self, words):
        matrix = [0 for _ in self.words]
        for index in range(len(self.words)):
            matrix[index] = matrix[index] + 1 if self.words[index] in words else matrix[index]
        return matrix

    def calculate(self, row):
        q1 = str(row['question1']).split()
        q2 = str(row['question2']).split()
        q1_matrix = self.count(q1)
        q2_matrix = self.count(q2)
        res = [0 for _ in self.columns]
        for i in range(len(q1_matrix)):
            if q1_matrix[i] == 0: continue
            for j in range(len(q2_matrix)):
                if q2_matrix[j] == 0: continue
                column = '{}_{}'.format(self.words[min(i, j)], self.words[max(i, j)])
                res[self.columns.index(column)] += 1
        return res

    def _extract(self):
        columns = self.columns
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class LongestCommonSeq(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1'])
        seq2 = str(row['question2'])
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        m = [[0 for _ in range(len(seq2)+1)] for _ in range(len(seq1)+1)]
        for p1 in range(1, len(seq1)+1):
            for p2 in range(1, len(seq2)+1):
                if seq1[p1-1] == seq2[p2-1]:
                    m[p1][p2] = m[p1 - 1][p2 - 1] + 1
                else:
                    m[p1][p2] = max(m[p1 - 1][p2], m[p1][p2 - 1])
        a = m[-1][-1]
        b = max(len(seq1), len(seq2))
        return a / b

    def _extract(self):
        self.data['LongestCommonSeq'] = [self.calculate(row) for _, row in self.data.iterrows()]


class LevenshteinDistance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1'])
        seq2 = str(row['question2'])
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        if len(seq1) > len(seq2):
            seq1, seq2 = seq2, seq1
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        seq1_length = len(seq1) + 1
        seq2_length = len(seq2) + 1
        distance_matrix = [list(range(seq2_length)) for _ in range(seq1_length)]
        for i in range(1, seq1_length):
            for j in range(1, seq2_length):
                deletion = distance_matrix[i - 1][j] + 1
                insertion = distance_matrix[i][j - 1] + 1
                substitution = distance_matrix[i - 1][j - 1]
                if seq1[i - 1] != seq2[j - 1]:
                    substitution += 1
                distance_matrix[i][j] = min(insertion, deletion, substitution)
        return 1 - distance_matrix[seq1_length - 1][seq2_length - 1] / float(seq2_length - 1)

    def _extract(self):
        self.data['LevenshteinDistance'] = [self.calculate(row) for _, row in self.data.iterrows()]


class Distance(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1'])
        seq2 = str(row['question2'])
        jaccard = distance.jaccard(seq1, seq2)
        sorensen = distance.sorensen(seq1, seq2)
        return [jaccard, sorensen]

    def _extract(self):
        columns = ['Jaccard', 'Sorensen']
        data = map_reduce(self.data, self.calculate, columns, n=4)
        for column in columns:
            self.data[column] = data[column]


class StrikeAMatch(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1'])
        seq2 = str(row['question2'])
        return strike_a_match(seq1, seq2)

    def _extract(self):
        self.data['StrikeAMatch'] = [self.calculate(row) for _, row in self.data.iterrows()]


class CharLengthDiff(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        return abs(len(q1) - len(q2))

    def _extract(self):
        self.data['CharLengthDiff'] = [self.calculate(row) for _, row in self.data.iterrows()]


class CharLengthDiffRatio(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        return min(len(q1), len(q2)) / max(len(q1), len(q2))

    def _extract(self):
        self.data['CharLengthDiffRatio'] = [self.calculate(row) for _, row in self.data.iterrows()]


class WordLengthDiff(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1']).split()
        q2 = str(row['question2']).split()
        return abs(len(q1) - len(q2))

    def _extract(self):
        self.data['WordLengthDiff'] = [self.calculate(row) for _, row in self.data.iterrows()]


class WordLengthDiffRatio(ClassicalFeature):

    def __init__(self, mode='train'):
        ClassicalFeature.__init__(self, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        q1 = str(row['question1']).split()
        q2 = str(row['question2']).split()
        return min(len(q1), len(q2)) / max(len(q1), len(q2))

    def _extract(self):
        self.data['WordLengthDiffRatio'] = [self.calculate(row) for _, row in self.data.iterrows()]


if __name__ == '__main__':
    SpecialConcurrence().extract()
