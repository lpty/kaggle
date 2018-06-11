# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.externals import joblib
from quora.util.similarity.generator_bm25 import LineSentence, GeneratorBM25
from quora.core.const import Punctuations, StopWords
from quora import config


class BM25(object):

    corpus_path = config.root_path.format('bm25.txt')

    @classmethod
    def init_corpus(cls):
        filter_words = list(Punctuations.PUNCTUATIONS) + list(StopWords.StopWordsEN)
        fw = open(cls.corpus_path, 'w')
        train_data = pd.read_csv(config.clean_train_file, index_col=0).drop(['id', 'is_duplicate', 'qid1', 'qid2'], axis=1)
        test_data = pd.read_csv(config.clean_test_file, index_col=0).drop(['test_id'], axis=1)
        data = train_data.append(test_data)
        seq2id = joblib.load(config.model_path.format('seq2id.model'))
        for _, row in data.iterrows():
            print(_)
            seq1 = str(row['question1']).strip()
            seq2 = str(row['question2']).strip()
            n1 = seq2id[seq1]
            n2 = seq2id[seq2]
            seq1 = ' '.join([str(n1)]+[w for w in seq1.split() if w not in filter_words])
            seq2 = ' '.join([str(n2)]+[w for w in seq2.split() if w not in filter_words])
            fw.write(seq1 + '\n')
            fw.write(seq2 + '\n')
        fw.close()

    @classmethod
    def train(cls):

        bm25 = GeneratorBM25(LineSentence(cls.corpus_path), alpha=30)
        average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

        res = []
        for doc in LineSentence(cls.corpus_path):
            print(doc[0])
            scores = bm25.get_scores(doc[1:], average_idf)
            res.append((int(doc[0]), scores))

        joblib.dump(bm25, config.model_path.format('bm25.model'))
        joblib.dump(res, config.model_path.format('bm25_res.model'))

    @classmethod
    def load(cls, model=False):
        bm25_res = joblib.load(config.model_path.format('bm25_res.model'))
        if model:
            bm25 = joblib.load(config.model_path.format('bm25.model'))
            return bm25, bm25_res
        else:
            return bm25_res


if __name__ == '__main__':
    # BM25.init_corpus()
    BM25.train()
    # BM25.load()
