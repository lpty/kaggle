# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
from sklearn.externals import joblib
from quora.util.represenation.bm25 import BM25
from quora import config


class UnDirectGraph(object):

    @classmethod
    def train(cls):
        graph = nx.Graph()
        seq2id = {}
        train_data = pd.read_csv(config.clean_train_file, index_col=0).drop(['id', 'is_duplicate', 'qid1', 'qid2'], axis=1)
        test_data = pd.read_csv(config.clean_test_file, index_col=0).drop(['test_id'], axis=1)
        data = train_data.append(test_data)
        for _, row in data.iterrows():
            seq1 = str(row['question1']).strip()
            seq2 = str(row['question2']).strip()
            seq2id[seq1] = seq2id.get(seq1, len(seq2id))
            seq2id[seq2] = seq2id.get(seq2, len(seq2id))
            graph.add_edge(seq2id[seq1], seq2id[seq2])
        joblib.dump(seq2id, config.model_path.format('seq2id.model'))
        joblib.dump(graph, config.model_path.format('un_direct_graph.model'))

    @classmethod
    def load(cls, graph=True):
        print('Load seq2id...')
        seq2id = joblib.load(config.model_path.format('seq2id.model'))
        if graph:
            print('Load un_direct_graph...')
            graph = joblib.load(config.model_path.format('un_direct_graph.model'))
            print('Load End.')
            return seq2id, graph
        else:
            print('Load End.')
            return seq2id

    @classmethod
    def train_page_rank(cls):
        graph = joblib.load(config.model_path.format('un_direct_graph.model'))
        page_rank = nx.pagerank(graph)
        joblib.dump(page_rank, config.model_path.format('page_rank.model'))

    @classmethod
    def load_page_rank(cls):
        page_rank = joblib.load(config.model_path.format('page_rank.model'))
        return page_rank


class UnDirectWeightGraph(object):

    @classmethod
    def train(cls):
        weights = BM25.load()
        graph = nx.Graph()
        for n1, weight in weights:
            print(n1)
            for n2, w in weight[n1]:
                if n1 == n2: continue
                graph.add_edge(n1, n2, weight=w)
        joblib.dump(graph, config.model_path.format('un_direct_weight_graph.model'))

    @classmethod
    def load(cls):
        seq2id = joblib.load(config.model_path.format('seq2id.model'))
        graph = joblib.load(config.model_path.format('un_direct_weight_graph.model'))
        return seq2id, graph


if __name__ == '__main__':
    # UnDirectGraph.train()
    UnDirectGraph.train_page_rank()
