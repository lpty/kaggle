# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import KeyedVectors
from quora.core.const import Punctuations, StopWords
from quora import config


class W2V(object):

    corpus_path = config.root_path.format('w2v.txt')

    @classmethod
    def init_corpus(cls):
        filter_words = list(Punctuations.PUNCTUATIONS) + list(StopWords.StopWordsEN)
        fw = open(cls.corpus_path, 'a')
        data = pd.read_csv(config.clean_test_file, index_col=0)
        for row in data['question1']:
            q1 = ' '.join([w for w in str(row).split() if w not in filter_words])
            fw.write(q1 + '\n')
        for row in data['question2']:
            q1 = ' '.join([w for w in str(row).split() if w not in filter_words])
            fw.write(q1 + '\n')

    @classmethod
    def train(cls):
        model = Word2Vec(sentences=LineSentence(cls.corpus_path), size=300, window=5, min_count=5, workers=4)
        model.wv._save(config.model_path.format('w2v.model'))

    @classmethod
    def load(cls):
        return KeyedVectors.load(config.model_path.format('w2v.model'))


class D2V(object):

    corpus_path = config.root_path.format('w2v.txt')

    @classmethod
    def train(cls):
        model = Doc2Vec(documents=TaggedLineDocument(cls.corpus_path), vector_size=300, window=5, min_count=1, workers=4)
        model.save(config.model_path.format('d2v.model'))

    @classmethod
    def load(cls):
        return Doc2Vec.load(config.model_path.format('d2v.model'))


class Glove(object):

    @classmethod
    def load(cls):
        glove = {}
        f = open(config.model_path.format('glove.txt'), 'r')
        for line in f:
            sub = line.split(maxsplit=1)
            if len(sub) < 2: continue
            word = sub[0]
            vector = sub[1]
            vector = np.array([float(v) for v in vector.split()])
            glove[word] = vector
        f.close()
        return glove


if __name__ == '__main__':
    W2V.init_corpus()
    W2V.train()
    w2v = W2V.load()
    D2V.train()
    d2v = D2V.load()
