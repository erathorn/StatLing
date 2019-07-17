"""

Still here for legacy reasons
"""

import numpy as np

import EvolutionaryModel
import src.C_Extensions.algorithms_cython as calc_c
import src.C_Extensions.pairwise_agglo as c_pw


class EvolutionaryModelPairwiseBottomUp(EvolutionaryModel.EvolutionaryModel):
    """
    Likelihood computation for pairwise cognacy statements
    """
    __slots__ = ("length_scores", "ind_ct")

    def __init__(self, emission_model, transition_model, data):
        super(EvolutionaryModelPairwiseBottomUp, self).__init__(emission_model, transition_model, data)
        self.collected_emission = np.zeros((self.n_lang_pairs, self.em_mod.alpha_size, self.em_mod.alpha_size),
                                           dtype=np.float64, order="C")
        self.length_scores = self.data.n_concepts
        for node in self.tree.terminal_nodes:
            l_ind = self.data.lang_ind_dict[node.name]
            node.cluster_store[:, 0, 0, 0] = l_ind

        self.candidates += [(self.sample_pi, self.revert_pi, "restriction", {})]

    def likelihood_language_sensitive(self, tc=False):
        # type: (bool) -> float
        """
        This function wraps the likelihood calculation.
        First the parameters are collected.
        Then the collected information is passed to the C code.
        Then the likelihood calculator is called which performs the calculation of the tree likelihood.

        :param tc: indicator if model parameters need to be recalculated
        :type tc: bool
        :return: likelihood of the model
        :rtype: float
        """
        if not tc:
            # storage parameters
            self.info_for_language_pair()

        # calculate word pair scores in C
        self.data.related_scores = calc_c.calculate_scores_big_v2(self.data.all_words, self.data.n_concepts,
                                                                  self.ind_ct,
                                                                  self.data.n_languages,
                                                                  self.collected_emission, self.em_mod.gap_log,
                                                                  self.collected_transition,
                                                                  0.0,
                                                                  self.data.alphabet_size, self.tr_mod.l,
                                                                  self.tr_mod.mu)

        ll = self.likelihood_calculator()
        self.data.binarise_bottom_up()
        ll1 = self.felsenstein_algorithm()

        return ll + ll1

    def info_for_language_pair(self):

        """
        This function collects the parameters for a specific language pair

        :param lang_pair: a particular language pair
        :return: word pairs, emission matrix, gap scores, transition values
        """

        self.diag_func() if self.em_mod.sound_model.necessary else None
        self.ind_ct = np.zeros((self.data.n_languages, self.data.n_languages), dtype=np.int32)
        # create model
        for ind, (l1, l2) in enumerate(self.data.language_pairs):
            t = self.tree.times[l1][l2]
            self.ind_ct[self.data.languages.index(l1)][self.data.languages.index(l2)] = ind
            self.ind_ct[self.data.languages.index(l2)][self.data.languages.index(l1)] = ind
            if l1 != l2:
                self.em_mod.create_model(time=t)
                self.tr_mod.set_model(t)
            self.collected_emission[ind] = self.em_mod.emission_mat_log
            self.collected_transition[ind] = self.tr_mod.trans_vals

    def likelihood_calculator(self):
        # type: () -> float
        """
        calculate the likelihood of the tree from pairwise alignment scores

        :return likelihood of the current model
        """

        tree_it = self.tree.post_order
        rs_all = np.swapaxes(self.data.related_scores, 0, 2)

        sc_d = np.vstack(rs_all[c_i].ravel(order="C") for c_i in range(self.length_scores))

        for node in tree_it:
            if not node.terminal:

                # get left and right child
                left = node.left  # type: Tree.Node
                right = node.right  # type: Tree.Node

                l_cs = left.cluster_store
                r_cs = right.cluster_store
                """
                This approach is a bottom-up clustering method. Inspired by agglomerative clustering approaches. 
                Clusters from the daughter nodes are merged if their joint similarity score is higher than if they 
                remain separate.
                """

                node.cluster_store.fill(-1)
                score = c_pw.concept_it(self.n_concepts, l_cs, r_cs, sc_d, node.cluster_store, self.data.merge_prob,
                                        self.data.no_merge_prob)

                node.accumulated_likelihood_rel = score

            else:
                # node is a terminal node, not much to do here
                node.likelihood_rel = 0.0  # n_zer(self.length_scores)

                # dimensions (concept, cluster, scores, languages)
                l_ind = self.data.lang_ind_dict[node.name]

                z_i = np.where(rs_all[:, l_ind, l_ind] == 0)
                node.cluster_store[:, 0, 0, 36] = 1
                node.cluster_store[z_i, 0, 0, 36] = 0
                node.cluster_store[:, 0, 1, 0] = rs_all[:, l_ind, l_ind]

        return self.tree.root.accumulated_likelihood_rel
