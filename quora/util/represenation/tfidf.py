# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA, TruncatedSVD, NMF, LatentDirichletAllocation
from quora import config


class TFIDF(object):

    @classmethod
    def train(cls):
        train_data = pd.read_csv(config.clean_train_file)
        test_data = pd.read_csv(config.clean_test_file)
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(
            train_data['question1'].tolist() + train_data['question2'].tolist() +
            test_data['question1'].tolist() + test_data['question2'].tolist()).astype(str)
        tfidf.fit(tfidf_txt)
        joblib.dump(tfidf, config.model_path.format('tfidf.model'))

    @classmethod
    def load(cls):
        model = joblib.load(config.model_path.format('tfidf.model'))
        return model


class DimReduce(object):

    @classmethod
    def init_corpus(cls):
        idf = TFIDF.load()
        train_data = pd.read_csv(config.clean_train_file)
        test_data = pd.read_csv(config.clean_test_file)
        data = train_data['question1'].tolist() + train_data['question2'].tolist() + \
               test_data['question1'].tolist() + test_data['question2'].tolist()
        del train_data
        del test_data
        idf_data = [idf.transform([str(seq)])[0].toarray() for seq in data[:10]]
        return idf_data

    @classmethod
    def train(cls):
        idf_data = cls.init_corpus()
        algorithms = {'PCA': IncrementalPCA, 'SVD': TruncatedSVD, 'NMF': NMF, 'LDA': LatentDirichletAllocation}
        n_components_list = [50, 100, 300, 500]
        for algorithm, alg in list(algorithms.items())[:1]:
            for n_components in n_components_list:
                reduce = alg(n_components=n_components)
                reduce.fit(idf_data)
                if algorithm in ['PCA', 'SVD']:
                    variance = str(round(sum(reduce.explained_variance_ratio_), 5))
                elif algorithm in ['NMF']:
                    variance = str(round(sum(reduce.reconstruction_err_), 5))
                else:
                    variance = '0'
                model_name = '{}_{}_{}.model'.format(algorithm, str(n_components), variance)
                joblib.dump(reduce, config.model_path.format(model_name))

    @classmethod
    def load(cls, name):
        model = joblib.load(config.model_path.format('{}.model'.format(name)))
        return model


if __name__ == '__main__':
    DimReduce.train()
