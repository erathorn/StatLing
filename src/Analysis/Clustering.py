# coding: utf-8
import collections
import itertools

import igraph
import numpy as np


def igraph_clustering(matrix, threshold, method='infomap'):
    """
    Wrapper around several of igraph's community structure finding algorithm
    implementations, most notably InfoMap.
    This is originally taken from and builds on LingPy's infomap_clustering
    function.
    """

    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]

    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if i < j:
                if cell > threshold:
                    G.add_edge(i, j, weight=cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                                    vertex_weights=None)

    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                                              initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i, comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i

    return D


def cluster(dataset, scores, threshold=0.5, method='infomap'):
    """
    Cluster the dataset's synonymous words into cognate sets based on distance
    scores between each pair of words. Return a dict mapping concepts to frozen
    sets of frozen sets of Word tuples.
    The second arg should be a {(word1, word2): distance} dict where the keys
    are sorted Word named tuples tuples. The keyword args are passed on to the
    InfoMap algorithm.
    """
    clusters = {}

    for concept, words in dataset.get_concepts().items():
        if len(words) <= 1: continue

        matrix = np.zeros((len(words), len(words),))

        for (i, word1), (j, word2) in itertools.combinations(enumerate(words), 2):
            key = (word1, word2) if word1 < word2 else (word2, word1)
            matrix[j, i] = matrix[i, j] = scores[key]

        index_labels = igraph_clustering(matrix, threshold, method)

        cog_sets = collections.defaultdict(set)
        for index, label in index_labels.items():
            cog_sets[label].add(words[index])

        clusters[concept] = frozenset([frozenset(s) for s in cog_sets.values()])

    return clusters


def calc_b_cubed(true_labels, labels):
    """
    Calculate the B-cubed (precision, recall, F-score) of a list of cognate set
    labels against the gold-standard version of the same list.
    This function is a (just slightly) modified version of the b_cubed function
    of PhyloStar's CogDetect library.
    """
    precision = [0.0] * len(true_labels)
    recall = [0.0] * len(true_labels)

    for i, l in enumerate(labels):
        match = 0.0
        prec_denom = 0.0
        recall_denom = 0.0
        for j, m in enumerate(labels):
            if l == m:
                prec_denom += 1.0
                if true_labels[i] == true_labels[j]:
                    match += 1.0
                    recall_denom += 1.0
            elif l != m:
                if true_labels[i] == true_labels[j]:
                    recall_denom += 1.0
        precision[i] = match / prec_denom
        recall[i] = match / recall_denom

    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f_score = 2.0 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    return avg_precision, avg_recall, avg_f_score


def calc_f_score(true_clusters, pred_clusters):
    """
    Calculate the B-cubed F-score of a dataset's cognate sets against their
    gold-standard. This is the metric used to evaluate the performance of the
    cognacy identification algorithms.
    Both args should be dicts mapping concepts to frozen sets of frozen sets of
    Word named tuples. The first comprises the gold-standard clustering and the
    second comprises the inferred one.
    It is assumed that both clusterings comprise the same data and that there
    is at most one word per concept per doculect. An AssertionError is raised
    if these assumptions do not hold true.
    """
    f_scores = []
    prec = []
    rec = []

    for concept in true_clusters.keys():

        assert concept in pred_clusters, str(concept)

        true_labels = {}
        pred2 = {}

        for index, cog_set in enumerate(true_clusters[concept]):
            label = 'true:{}'.format(index)
            for lang in true_clusters[concept][cog_set]:
                true_labels[lang] = label

        for index, cog_set in enumerate(pred_clusters[concept]):
            label = 'pred:{}'.format(index)
            for lang in pred_clusters[concept][cog_set]:
                pred2[lang] = label

        pred_labels = {}
        for k, v in pred2.iteritems():
            if k in true_labels.keys():
                pred_labels[k] = v
            else:
                1
        assert set(true_labels.keys()) == set(pred_labels.keys()), str(concept)
        sorted_keys = sorted(true_labels.keys())
        true_labels_list = [true_labels[label] for label in sorted_keys]
        pred_labels_list = [pred_labels[label] for label in sorted_keys]
        p, r, f = calc_b_cubed(true_labels_list, pred_labels_list)
        f_scores.append(f)
        prec.append(p)
        rec.append(r)

    return prec, rec, f_scores
