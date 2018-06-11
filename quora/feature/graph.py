# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import networkx as nx
from quora.feature.feature import ClassicalFeature
from quora.util.represenation.g import UnDirectGraph, UnDirectWeightGraph


class Graph(ClassicalFeature):

    def __init__(self, gtype, mode='train'):
        ClassicalFeature.__init__(self, mode)
        if gtype == 'concurrence':
            self.seq2id, self.graph = UnDirectGraph.load()
        else:
            self.seq2id, self.graph = UnDirectWeightGraph.load()

    @staticmethod
    def gen_degrees(graph):
        max_degrees = {}
        edges = graph.edges()
        for edge in edges:
            for n in edge:
                max_degrees[n] = max_degrees.get(n, 0) + 1
        return max_degrees

    @staticmethod
    def gen_components(graph):
        max_components = {}
        components = nx.connected_components(graph)
        for component in components:
            for n in component:
                max_components[n] = max(max_components.get(n, 0), len(component))
        return max_components

    @staticmethod
    def gen_cliques(graph):
        n2clique = {}
        n_cliques = []
        cliques = nx.find_cliques(graph)
        for clique in cliques:
            for n in clique:
                n2clique[n] = n2clique.get(n, []) + [len(n_cliques)]
            n_cliques.append(list(clique))
        return n2clique, n_cliques

    @staticmethod
    def gen_page_rank(graph):
        page_rank = UnDirectGraph.load_page_rank()
        return page_rank

    @staticmethod
    def gen_hits(graph):
        hits_h, hits_a = nx.hits(graph)
        return hits_h, hits_a


class Statistics(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]
        self.max_degrees = Graph.gen_degrees(self.graph)
        self.max_components = Graph.gen_components(self.graph)
        self.n2clique, self.n_cliques = Graph.gen_cliques(self.graph)

    def calculate_max_degree(self, seq):
        n = self.seq2id[seq]
        return self.max_degrees[n]

    def calculate_max_connected(self, seq):
        n = self.seq2id[seq]
        return self.max_components[n]

    def calculate_max_clique(self, seq):
        n = self.seq2id[seq]
        clique_id = self.n2clique[n]
        return max([len(clique) for clique in self.n_cliques[clique_id]])

    def calculate_related_clique(self, seq1, seq2):
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        clique_id = self.n2clique[n1]
        res = [len(clique) for clique in self.n_cliques[clique_id] if n2 in clique]
        return len(res), max(res)

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        seq1_max_degree = self.calculate_max_degree(seq1)
        seq2_max_degree = self.calculate_max_degree(seq2)

        max_connected = self.calculate_max_connected(seq1)

        return [seq1_max_degree, seq2_max_degree, max_connected]

    def _extract(self):
        data = {}
        columns = ['seq1_max_degree', 'seq2_max_degree', 'max_connected']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class Structure(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]
        self.n2clique, self.n_cliques = Graph.gen_cliques(self.graph)

    def calculate_max_clique(self, seq):
        n = self.seq2id[seq]
        clique_ids = self.n2clique[n]
        return max([len(self.n_cliques[clique_id]) for clique_id in clique_ids])

    def calculate_related_clique(self, seq1, seq2):
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        clique_ids = self.n2clique[n1]
        res = [len(self.n_cliques[clique_id]) for clique_id in clique_ids if n2 in self.n_cliques[clique_id]]
        return len(res), max(res)

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()

        seq1_max_clique = self.calculate_max_clique(seq1)
        seq2_max_clique = self.calculate_max_clique(seq2)

        related_max_clique, related_num_clique = self.calculate_related_clique(seq1, seq2)

        return [seq1_max_clique, seq2_max_clique, related_max_clique, related_num_clique]

    def _extract(self):
        data = {}
        columns = ['seq1_max_clique', 'seq2_max_clique', 'related_max_clique', 'related_num_clique']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class PageRank(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]
        self.page_rank = Graph.gen_page_rank(self.graph)

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        page_rank_1 = self.page_rank[n1] * 1e6
        page_rank_2 = self.page_rank[n2] * 1e6
        return [page_rank_1, page_rank_2]

    def _extract(self):
        data = {}
        columns = ['page_rank_1', 'page_rank_2']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class Hits(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]
        self.hits_h, self.hits_a = Graph.gen_hits(self.graph)

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        hits_h_1 = self.hits_h[n1] * 1e6
        hits_a_1 = self.hits_a[n1] * 1e6
        hits_h_2 = self.hits_h[n2] * 1e6
        hits_a_2 = self.hits_a[n2] * 1e6
        return [hits_h_1, hits_a_1, hits_h_2, hits_a_2]

    def _extract(self):
        data = {}
        columns = ['hits_h_1', 'hits_a_1', 'hits_h_2', 'hits_a_2']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class ShortestPath(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        self.graph.remove_edge(n1, n2)
        if nx.has_path(self.graph, n1, n2):
            shortest_path = nx.shortest_path(n1, n2)
        else:
            shortest_path = -1
        self.graph.add_edge(n1, n2)
        return [shortest_path]

    def _extract(self):
        data = {}
        columns = ['shortest_path']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class Neighbor(Graph):

    def __init__(self, gtype='concurrence', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        neighbor_1 = self.graph.neighbors(n1)
        neighbor_2 = self.graph.neighbors(n2)
        neighbor_num = len(set(neighbor_1) & set(neighbor_2))
        return [neighbor_num]

    def _extract(self):
        data = {}
        columns = ['neighbor_num']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightStatistics(Statistics):

    def __init__(self, gtype='similarity', mode='train'):
        Statistics.__init__(self, gtype, mode)

    def _extract(self):
        data = {}
        columns = ['w_seq1_max_degree', 'w_seq2_max_degree', 'w_max_connected']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightStructure(Structure):

    def __init__(self, gtype='similarity', mode='train'):
        Structure.__init__(self, gtype, mode)

    def _extract(self):
        data = {}
        columns = ['w_seq1_max_clique', 'w_seq2_max_clique', 'w_related_max_clique', 'w_related_num_clique']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightPageRank(PageRank):

    def __init__(self, gtype='similarity', mode='train'):
        Graph.__init__(self, gtype, mode)

    def _extract(self):
        data = {}
        columns = ['w_page_rank_1', 'w_page_rank_2']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightHits(Hits):

    def __init__(self, gtype='similarity', mode='train'):
        Graph.__init__(self, gtype, mode)

    def _extract(self):
        data = {}
        columns = ['w_hits_h_1', 'w_hits_a_1', 'w_hits_h_2', 'w_hits_a_2']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightShortestPath(ShortestPath):

    def __init__(self, gtype='similarity', mode='train'):
        Graph.__init__(self, gtype, mode)

    def _extract(self):
        data = {}
        columns = ['w_shortest_path']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


class WeightNeighbor(Graph):

    def __init__(self, gtype='similarity', mode='train'):
        Graph.__init__(self, gtype, mode)
        self.extract_stage = [self._extract]

    def calculate(self, row):
        seq1 = str(row['question1']).strip()
        seq2 = str(row['question2']).strip()
        n1 = self.seq2id[seq1]
        n2 = self.seq2id[seq2]
        neighbor_1 = self.graph.neighbors(n1)
        neighbor_2 = self.graph.neighbors(n2)
        neighbor_num = len(set(neighbor_1) & set(neighbor_2))
        return [neighbor_num]

    def _extract(self):
        data = {}
        columns = ['neighbor_num']
        for _, row in self.data.iterrows():
            print(_)
            res = self.calculate(row)
            for column in columns:
                data[column] = data.get(column, []) + [res[columns.index(column)]]
        for column in columns:
            self.data[column] = data[column]


if __name__ == '__main__':
    Structure().extract()
