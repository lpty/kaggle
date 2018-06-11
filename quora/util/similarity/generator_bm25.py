#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
An corpus generator supported version from: gensim.summarization.bm25
"""


import math
from six import iteritems
from six.moves import xrange


PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class GeneratorBM25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    corpus : list of list of str
        Corpus of documents.
    f : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    df : dict
        Dictionary with terms frequencies for whole `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed terms frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    #
    def __init__(self, corpus, alpha):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.

        """
        self.alpha = alpha
        self.corpus_size = 0
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.index_map = {}
        self.initialize()

    def initialize(self):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        avgdl_tmp = []
        for document in self.corpus:
            index_map = document[0]
            document = document[1:]
            self.index_map[self.corpus_size] = int(index_map)

            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

            self.corpus_size += 1
            avgdl_tmp.append(float(len(document)))

        self.avgdl = sum(avgdl_tmp) / self.corpus_size

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.
        average_idf : float
            Average idf in corpus.

        Returns
        -------
        float
            BM25 score.

        """
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        average_idf : float
            Average idf in corpus.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            if score > self.alpha:
                scores.append((self.index_map[index], score))
        return scores


class LineSentence(object):

    def __init__(self, corpus_path):
        self.corpus = open(corpus_path, 'r')

    def __iter__(self):
        for line in self.corpus:
            res = line.split()
            yield res
        self.corpus.close()
