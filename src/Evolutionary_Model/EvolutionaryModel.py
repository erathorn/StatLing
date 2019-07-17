"""
@author: Johannes Wahle
@date: July 2018
@version: 1.0

"""

import itertools
import math
import random
from abc import abstractmethod, ABCMeta

import numpy as np

import src.C_Extensions.helper as c_help
import src.C_Extensions.sample as c_sam
import src.SamplerSettings.Prior as Prior
import src.Utils.slice_sampling as slice_sam

# number of times slice sampling resamples if initial sample failed
resample_max = 10


class EvolutionaryModel(object):
    """
    This class implements the Evolutionary model, which is at the heart of the MCMC
    """
    __metaclass__ = ABCMeta

    __slots__ = ("em_mod", "tr_mod", "data", "tree", "n_lang_pairs", "n_concepts", "tc", "candidates", "old_stack",
                 "diag_func", "collected_emission", "collected_transition", "pi")

    def __init__(self, emission_model, transition_model, data):
        """
        :param emission_model:
        :type emission_model: EmissionModel.EmissionModel.EmissionModel
        :param transition_model:
        :type transition_model: StatAlign_Jo.C_Extensions.cythoncode.Trans_doubleM.Transition_doubleM
        :param data:
        :type data: Utils.Data_class.DataClass
        """

        self.em_mod = emission_model
        self.tr_mod = transition_model
        self.data = data
        self.tree = self.data.tree

        self.n_lang_pairs = len(self.data.language_pairs)
        self.n_concepts = len(self.data.concepts)

        self.tc = True

        self.candidates = []

        self.candidates += self.em_mod.sampling_functions_classes * Prior.emission_sample_class
        self.candidates += self.em_mod.sampling_functions_frequencies * Prior.emission_sample_freq
        self.candidates += self.tr_mod.sampling_functions * Prior.transition_sample
        self.candidates += self.tree.sampling_functions_local_topology * Prior.tree_sample_topology
        self.candidates += self.tree.sampling_functions_local_branch * Prior.tree_sample_branches

        # initialize stack for sampling
        self.old_stack = []

        # get function name to top level for speed up reasons
        self.diag_func = self.em_mod.sound_model.diagonalize_q_matrix

        # create matrices which are always needed to avoid reoccurring initialization
        self.collected_emission = np.zeros((self.n_lang_pairs, self.em_mod.alpha_size, self.em_mod.alpha_size),
                                           dtype=np.float64, order="C")
        self.collected_transition = np.zeros((self.n_lang_pairs, len(self.tr_mod.trans_vals)), dtype=np.float64,
                                             order="C")

        self.pi = random.uniform(0, 1)

    @abstractmethod
    def likelihood_calculator(self):
        # type: () -> float
        pass

    @abstractmethod
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

        pass

    def felsenstein_algorithm(self):
        """
        This is the felsenstein algorithm

        still here for legacy reasons

        :return:
        """

        for node in self.tree.post_order:

            if node.terminal:
                pass

            else:
                left_daughter = node.left
                right_daughter = node.right

                m1 = self.felsenstein_q(left_daughter.incident_length)
                m2 = self.felsenstein_q(right_daughter.incident_length)

                """
                Iteration over cognate classes is done in cython.                    
                """
                rs = np.empty(left_daughter.state_probs.shape)
                c_help.inner_loop_felsenstein(left_daughter.state_probs,
                                              right_daughter.state_probs,
                                              len(left_daughter.state_probs),
                                              m1, m2, rs)
                """
                calculation of cognate class likelihoods needs to be done
                
                iterate over classes 
                - keep track of number of class members
                - efficient iteration required
                
                """
                node.state_probs = rs

        z = np.sum(np.log(np.sum(self.tree.root.state_probs * [self.pi, 1 - self.pi], axis=1)))
        self.tree.root.likelihood_rel = z

        return self.tree.root.likelihood_rel

    def felsenstein_q(self, t):
        """
        This is the restriction site model as proposed in Felsenstein 1981

        Still here for legacy reasons

        :return: substitution prob array
        """
        exp_t = np.exp(-t)
        t_ = (1 - exp_t)
        zz = exp_t + t_ * self.pi
        zo = t_ * (1 - self.pi)
        oz = t_ * self.pi
        oo = exp_t + t_ * (1 - self.pi)
        return np.array([[zz, zo], [oz, oo]])

    def sample_pi(self):
        """
        Still here for legacy reasons
        :return:
        """
        self.pi, m_h_ratio, old = c_sam.sample_uniform_single(Prior.dmax_freq, self.pi, 0.0, 1.0)

        return self.pi, m_h_ratio, 0.0, old

    def revert_pi(self, old):
        """
        Still here for legacy reasons
        :return:
        """
        self.pi = old

    """
    next move
    """

    def next_slice(self, current_likelihood):
        # type: (float) -> tuple
        """
        Next slice move

        :param current_likelihood: current likelihood of the model
        :type current_likelihood: float
        :return: new likelihood and acceptance count
        :rtype: tuple
        """
        current_likelihood_slice = current_likelihood + np.log(np.random.uniform())

        move = random.choice([1, 2, 3])

        resample_counter = 0
        if move == 1:
            # tree slice sampling

            vec = self.tree.all_edges[:-1]
            scale = Prior.dmax_time_width
            new, lower, upper = slice_sam.slice_sample(vec, scale)

            old = self.tree.set_all_edges(new)
            nll = self.likelihood_language_sensitive(tc=False)
            while np.any(new <= 0.0) | (nll < current_likelihood_slice) | math.isnan(nll):
                if resample_counter > resample_max:
                    self.tree.set_all_edges(old)
                    return current_likelihood, 0.0
                new = slice_sam.slice_shrink(new, lower, upper, old)
                self.tree.set_all_edges(new)
                nll = self.likelihood_language_sensitive(tc=False)
                resample_counter += 1

            return nll, 1.0
        else:
            if move == 2:
                # sample frequency
                vec = np.array([i for i in self.em_mod.sound_model.frequencies])
                setfun = self.em_mod.sound_model.set_frequency
                scale = Prior.freq_scale

            else:
                # sample evo
                vec = np.array([i for i in self.em_mod.sound_model.evo_values])
                setfun = self.em_mod.sound_model.set_evovals
                scale = Prior.evo_scale

            size = len(vec)
            vertices = slice_sam.initial_simplex(vec, size, scale)
            xb, vb = slice_sam.slice_sample_simplex(vertices, vec)
            x = np.dot(vertices, xb)
            resample_counter = 0

            old = setfun(x)
            nll = self.likelihood_language_sensitive(tc=False)
            while np.any(x < 0.0) | np.any(x > 1.0) | (nll < current_likelihood_slice) | math.isnan(nll):
                if resample_counter > resample_max:
                    setfun(old)
                    return current_likelihood, 0.0
                vertices = slice_sam.shrink_simplex(vertices, vec, xb, x, vb)
                xb, vb = slice_sam.slice_sample_simplex(vertices, vec)
                x = np.dot(vertices, xb)
                setfun(x)
                nll = self.likelihood_language_sensitive(tc=False)
                resample_counter += 1

            return nll, 1.0

    def next_step_RWM(self, w_size=2):
        # type: (int) -> (float, str)
        """
        This function proposes the next moves of the parameters

        :param w_size: number of steps to perform
        :return: metropolis hastings ratio and names performed steps
        """

        random.shuffle(self.candidates)

        operation = []

        # the stack keeps track of the moves performed in this iteration and stores the information
        # necessary to revert a particular move
        tc = True
        self.old_stack = []
        m_h_ratio = 0.0
        prior_prob = 0.0

        for i in range(w_size):
            ind = i % len(self.candidates)
            move, rev_func, op, mykwargs = self.candidates[ind]
            # move: particular move, implemented in Trans, Tree or EmissionModel
            # rev_func: function reverting the effect of move
            # op: string identifier of the move (necessary for console output)

            t_c, mh, pp, oi = move(**mykwargs)  # type: (bool, float,float, tuple|list)
            # res: identifier if to calculate, metropolis hastings ratio, revert information

            tc = tc and t_c
            operation.append(op)
            self.old_stack.append((rev_func, oi))
            m_h_ratio += mh
            prior_prob += pp

        return m_h_ratio, prior_prob, "+".join(operation), tc

    def revert_RWM(self):
        # type: () -> None
        """
        This function reverts the move proposed in self.next_step by emptying the stack
        of moves performed in the previous step.

        :return: None
        """

        while self.old_stack:
            revert_function, revert_params = self.old_stack.pop()
            # revert_function: specific revert function
            # revert_params: parameters of the revert function
            revert_function(*itertools.chain(revert_params)) if isinstance(revert_params, tuple) else revert_function(
                revert_params)
