"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import math
import random

import numpy as np

import src.EmissionModel.EmissionModel_Sound as src_em_EM
import src.EmissionModel.Feature_Single as src_em_FS
import src.Evolutionary_Model.EvolutionaryModel_Pairwise as src_evm_pw
import src.Evolutionary_Model.EvolutionaryModel_PairwiseBottomUp as src_evm_pw_BU
import src.TransitionModel.Trans_cy_wrap as TransModel
from src.Utils.Writer import DataWriter


class MCMC(object):

    def __init__(self, e_model, output_dir, sample):
        """
        This class handles the MCMC stuff on a high level.
        Its only function is to provide a framework for the MCMC run.

        :param e_model: The evolutionary model for which to run the MCMC estimation
        :param output_dir: Where to store the information and sampled parameters
        :param sample: indicates whether cognate classes are sampled or not
        """

        self.evo_model = e_model
        self.likelihoods = []

        # create Data writer class
        self.writer = DataWriter(self.evo_model, output_dir, sample)

    @staticmethod
    def create_emission_model(sound_model, data, params=None):
        if params is None:
            emmod = src_em_FS.FeatureSoundsSingle(names=data.alphabet, model=sound_model)
        else:
            emmod = src_em_FS.FeatureSoundsSingle.from_dict(params)
        return src_em_EM.EmissionModelSound(alphabet=data.alphabet, sound_mod=emmod)

    @staticmethod
    def create_transition_model_KM03(a, r):
        return TransModel.Trans_KM03(a=a, r=r, time=1.0)

    @staticmethod
    def create_transition_model_TKF91(mu, lambd):
        return TransModel.Trans_TKF91(mu=mu, lambd=lambd, time=1.0)

    @classmethod
    def create_mcmc(cls, parameters, data, temperature=1, tr_params=None, em_params=None):
        # type: (dict, src.Utils.Data_class, int, dict, dict) -> MCMC

        em = cls.create_emission_model(sound_model=parameters["Sound Model"], data=data, params=em_params)
        if parameters["Transition_Model"] == "KM03":
            a = random.uniform(0, 1)
            r = random.uniform(0, 1)
            if tr_params is not None:
                a = tr_params["a"]
                r = tr_params["r"]
            tr = cls.create_transition_model_KM03(a=a, r=r)

        elif parameters["Transition_Model"] == "TKF91":
            mu = random.uniform(0, 1)
            lam = random.uniform(0, mu)
            if tr_params is not None:
                mu = tr_params["mu"]
                lam = tr_params["l"]
            tr = cls.create_transition_model_TKF91(lambd=lam, mu=mu)

        else:
            raise Exception("The Transition Model is " + str(parameters["Transition_Model"]) + " is not defined")

        sample = True if parameters["sample_cognates"] else False
        if parameters["bottom-up"] is True:
            ev_m = src_evm_pw_BU.EvolutionaryModelPairwiseBottomUp(emission_model=em, transition_model=tr, data=data)
        else:
            ev_m = src_evm_pw.EvolutionaryModelPairwise(emission_model=em, transition_model=tr, data=data)
        return cls(ev_m, parameters["Output"], sample=sample)

    def randomize_starting_point(self, n_rands, window_size):
        # type: (int, int) -> None
        """
        Randomize the starting point, to make the different chains more independent

        :param n_rands: number of randomization steps
        :param window_size: window size for sampling
        """

        next_move = self.evo_model.next_step_RWM
        n = 0
        while n < n_rands:
            m, pp, _, _ = next_move(w_size=window_size)
            if np.isinf(m) or np.isinf(pp):
                self.evo_model.revert_RWM()
            n += 1

    def estimate(self, n_iterations, thinning, wsize):
        # type: (int, int, int) -> None
        """
        This function is the heart of the MCMC. Here Metropolis-Hastings method is employed.
        It wraps the estimation and does the acceptance decision.

        :param n_iterations: number of iterations the MCMC runs
        :type n_iterations: int
        :param thinning: thinning parameter, save model state every n-th generation, where n = thinning
        :type thinning: int
        :param wsize: windowsize for sampling
        :type wsize: int
        """

        # set counter
        counter = 0

        # create log files
        self.writer.create_params()

        # calculate initial model parameters
        curr_likelihood = self.evo_model.likelihood_language_sensitive(tc=False)

        # create local variables
        next_move = self.evo_model.next_step_RWM
        ll_computation = self.evo_model.likelihood_language_sensitive

        # store initial parameters
        self.writer.save_state(counter, curr_likelihood, True)

        counter += 1

        n_accept = 0.0

        step = 1
        while step < n_iterations:

            if random.choice([True, False]):
                curr_likelihood, nacc = self.evo_model.next_slice(curr_likelihood)
                n_accept += nacc
            else:
                # rwm move

                m_h_ratio, prior, candidate, tc = next_move(wsize)

                # calculate new likelihood
                new_likelihood = -np.inf if (np.isinf(m_h_ratio) or np.isinf(prior)) else ll_computation(tc=tc)

                # sanity check
                if math.isnan(new_likelihood):
                    print self.evo_model.tr_mod.a, self.evo_model.tr_mod.r
                    print self.evo_model.em_mod.sound_model.frequencies
                    print self.evo_model.em_mod.sound_model.evo_values
                    print self.evo_model.data.tree.time_store
                    raise Exception("likelihood is NaN. This is really bad.")

                # calculate acceptance ratio
                ll_ratio = (new_likelihood - curr_likelihood)
                n_score = ll_ratio + m_h_ratio + prior
                rho = min(0.0, n_score)

                u = np.log(random.uniform(0, 1))

                if u < rho:
                    # accept move
                    n_accept += 1.0
                    curr_likelihood = new_likelihood

                else:
                    # reject move
                    self.evo_model.revert_RWM()

            # store parameters if desired for current iteration
            if step % thinning == 0:
                rho_acc = n_accept / step
                print str(step) + " ----- " + str(rho_acc) + " ----- " + str(curr_likelihood)
                self.writer.save_state(counter, curr_likelihood, False)
                counter += 1

            step += 1
