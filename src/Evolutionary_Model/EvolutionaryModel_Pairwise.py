"""
@author: erathorn
@date: July 2019
@version: 1.0
"""
import itertools
import random

import numpy as np

import EvolutionaryModel
import src.C_Extensions.algorithms_cython as calc_c

n_zer = np.zeros
n_emp = np.empty
it_prod = itertools.product


class EvolutionaryModelPairwise(EvolutionaryModel.EvolutionaryModel):
    """
    Likelihood computation for pairwise cognacy statements
    """
    __slots__ = ("length_scores", "ix_dict", "correction_parameter", "time_store", "t_list", "lam", "CACHE")

    def __init__(self, emission_model, transition_model, data):

        super(EvolutionaryModelPairwise, self).__init__(emission_model, transition_model, data)
        self.collected_emission = np.zeros((self.n_lang_pairs, self.em_mod.alpha_size, self.em_mod.alpha_size),
                                           dtype=np.float64, order="C")
        self.time_store = np.zeros(self.n_lang_pairs, dtype=np.float64, order="C")
        self.length_scores = self.data.n_concepts
        self.correction_parameter = random.uniform(1, 0)
        self.CACHE = {}

        self.ix_dict = {}
        self.t_list = np.empty(self.n_lang_pairs, dtype=np.double, order="C")
        self.lam = 1.0

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
            self.diag_func() if self.em_mod.sound_model.necessary else None

        calc_c.new_calc(self.data.left_stack, self.data.right_stack,
                        self.tr_mod.a, self.tr_mod.r, self.data.tree.time_store,
                        self.em_mod.sound_model.q_eigenvectors,
                        self.em_mod.sound_model.q_eigenvectors_inv,
                        self.em_mod.sound_model.q_eigenvalues,
                        self.em_mod.sound_model.freq_log, 0.0, self.data.alphabet_size, self.data.related_scores)

        return self.likelihood_calculator()

    def likelihood_calculator(self):
        # type: () -> float
        """
        calculate the likelihood of the tree from pairwise alignment scores

        :return likelihood of the current model
        """

        rel_sc = self.data.related_scores[:, :, 0]
        unrel_sc = self.data.related_scores[:, :, 1]
        lpair_ind_dict = self.data.lpair_ind_dict

        for node in self.tree.post_order:
            if not node.terminal:

                # get left and right child
                left = node.left  # type: Tree.Node
                right = node.right  # type: Tree.Node
                left_arr = left.arity  # type: np.ndarray
                right_arr = right.arity  # type: np.ndarray
                t = right.incident_length + left.incident_length
                # get all terminal children of left and right
                l_term = (left.name,) if left.terminal else tuple(sorted(left.get_terminal()))
                r_term = (right.name,) if right.terminal else tuple(sorted(right.get_terminal()))

                # get size of cartesian product of terminal children of left and right
                arity_container = left_arr * right_arr
                ind = np.where(arity_container == 0.0)
                arity_container[ind] = 1.0
                node.arity = left_arr + right_arr

                # get denominator
                denominator_rel = left.likelihood_rel + right.likelihood_rel
                denominator_rel[ind] = 0.0

                # sum up all alignment scores to this point
                if (l_term, r_term) not in self.CACHE:
                    self.CACHE[(l_term, r_term)] = np.ix_(
                        [lpair_ind_dict[p] for p in itertools.product(l_term, r_term)])
                    self.CACHE[(r_term, l_term)] = self.CACHE[(l_term, r_term)]
                fast_inds = self.CACHE[(l_term, r_term)]

                scores_node_rel = np.sum(rel_sc[fast_inds], axis=0)
                scores_node_unrel = np.sum(unrel_sc[fast_inds], axis=0)
                # get likelihood of this node
                ll_rel = (scores_node_rel - denominator_rel) / arity_container
                node.likelihood_rel = ll_rel
                t1 = np.exp(-t)
                t2 = 1.0 - t1
                # calculate accumulated likelihood
                node.accumulated_likelihood_rel = np.sum(np.log(t1 * np.exp(ll_rel) + t2 * np.exp(
                    scores_node_unrel))) + left.accumulated_likelihood_rel + right.accumulated_likelihood_rel

            else:
                # node is a terminal node, not much to do here
                node.likelihood_rel = n_zer(self.length_scores)

        return self.tree.root.accumulated_likelihood_rel
