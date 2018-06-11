# -*- coding: utf-8 -*-
from pyemd import emd
from numpy import zeros, double, sum as np_sum
from gensim.corpora.dictionary import Dictionary
import scipy.spatial.distance


def wmd(document1, document2, model):
    # Remove out-of-vocabulary words.
    document1 = [token for token in document1 if token in model]
    document2 = [token for token in document2 if token in model]
    if len(document1) == 0 or len(document2) == 0:
        return 1.
    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)
    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in list(dictionary.items()):
        for j, t2 in list(dictionary.items()):
            distance_matrix[i, j] = scipy.spatial.distance.cosine(model[t1], model[t2])
    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        return 0.

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)
    # Compute WMD.
    res = emd(d1, d2, distance_matrix)
    return res if res >= 0 else 1
