"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import math
import random

import numpy as np
from mpi4py import MPI

import src.Evolutionary_Model.EvolutionaryModel_Pairwise as src_evm_pw
import src.Evolutionary_Model.EvolutionaryModel_PairwiseBottomUp as src_evm_pw_BU
import src.MCMC as MCMC

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class MCMC_MC3(MCMC.MCMC):

    def __init__(self, e_model, output_dir, sample, swap_store):
        super(MCMC_MC3, self).__init__(e_model, output_dir, sample)

        self.log_likelihood = None
        self.temperature = 1
        swap_store = swap_store[np.where((swap_store[::, 0] == rank) | (swap_store[::, 1] == rank))]
        self.swap_store = swap_store

    @classmethod
    def create_mcmc(cls, parameters, data, temperature=1.0, swap_store=None, tr_params=None, em_params=None):
        # type: (dict, src.Utils.Data_class, float, iter, dict, dict) -> MCMC_MC3

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
        mcmc_cls = cls(ev_m, parameters["Output"], sample=sample, swap_store=swap_store)
        mcmc_cls.temperature = temperature

        return mcmc_cls

    def compare_with_other(self, other_ll, other_temp):
        # type: (float, float) -> bool
        """
        compare this instance with other mcmc instance to decide if swap is desired

        :param other_ll: likelihood of other instance
        :type other_ll: float
        :param other_temp: temperature of other instance
        :type other_temp: float
        :return: boolean indicating if swap is desired
        :rtype: bool
        """
        swap_prob = (other_ll * self.temperature + self.log_likelihood * other_temp) - (
                self.log_likelihood * self.temperature + other_ll * other_temp)

        r = min(0.0, swap_prob)
        u = np.log(random.uniform(0, 1))
        return u < r

    def get_next_swap(self, step):
        # type: (int) -> tuple
        """
        Get generation where next swap should happen

        :param step: current generation
        :type step: int
        :return: info for next swap generation
        :rtype: tuple
        """
        self.swap_store = self.swap_store[np.where(self.swap_store[::, 2] > step)]
        if len(self.swap_store) == 0:
            return None, None, None
        return self.swap_store[0]

    def MC3_estimate(self, n_iterations, thinning, wsize):
        # type: (int, int, int) -> None
        """
        Do the MCMC estimation for the heated framework

        :param n_iterations: number of iterations to run
        :type n_iterations: int
        :param thinning: thinning parameter
        :type thinning: int
        :param wsize: window size for random walk MCMC
        :type wsize: int
        """
        # set counter
        counter = 0

        # create log files
        if self.temperature == 1:
            self.writer.create_params()

        # calculate initial model parameters
        self.log_likelihood = self.evo_model.likelihood_language_sensitive(tc=False)

        # create local variables
        next_move = self.evo_model.next_step_RWM
        ll_computation = self.evo_model.likelihood_language_sensitive

        # store initial parameters
        if self.temperature == 1:
            self.writer.save_state(counter, self.log_likelihood, True)

        counter += 1

        rho_acc = 0.0
        n_accept = 0.0
        step = 1

        # calculate window size

        don, rec, swap_iteration = self.get_next_swap(step)

        while step < n_iterations:
            if self.temperature == 1:

                if step % 10 == 0:
                    print(str(step) + " ----- " + str(rho_acc) + " ----- " + str(self.log_likelihood))

            # the proposal move is performed inside the evolutionary model
            if random.choice([True, False, False]):
                self.log_likelihood, nacc = self.evo_model.next_slice(self.log_likelihood)
                n_accept += nacc
            else:
                m_h_ratio, prior_prob, candidate, tc = next_move(wsize)

                # calculate new likelihood
                new_likelihood = -np.inf if np.isinf(m_h_ratio) else ll_computation(tc=tc)

                # sanity check
                if math.isnan(new_likelihood):
                    self.evo_model.revert_RWM()

                else:
                    # calculate acceptance ratio
                    ll_ratio = (new_likelihood - self.log_likelihood) + prior_prob
                    ll_ratio *= self.temperature
                    n_score = (ll_ratio + m_h_ratio)

                    rho = min(0.0, n_score)

                    u = np.log(random.uniform(0, 1))

                    if u < rho:
                        # accept move
                        n_accept += 1.0
                        self.log_likelihood = new_likelihood

                    else:
                        # reject move
                        self.evo_model.revert_RWM()

                rho_acc = n_accept / step

            # store parameters if desired for current iteration
            if self.temperature == 1:
                if step % thinning == 0:
                    self.writer.save_state(step // thinning, self.log_likelihood, False)
                    counter += 1

            if swap_iteration == step:
                # attempt swap
                self.try_swap(don, rec, swap_iteration)

                # get next swap iteration
                don, rec, swap_iteration = self.get_next_swap(step)

            step += 1

    def try_swap(self, don, rec, swap_iteration):
        # type: (int, int, int) -> None
        """
        Try a swap at the current generation

        :param don: donor chain
        :type don: int
        :param rec: receiving chain
        :type rec: int
        :param swap_iteration: current iteration used as a tag
        :type swap_iteration: int
        """
        if don != rec:
            if don == rank:
                # send data to other process to perform comparison
                mail = np.array([self.temperature, self.log_likelihood])
                comm.Send([mail, MPI.DOUBLE], dest=rec, tag=swap_iteration)

                # prepare receiving compare result
                mail = np.empty(1)
                comm.Recv(mail, source=rec, tag=swap_iteration)

                temp = mail[0]
                if temp == 10:
                    # no swap is desired
                    pass
                else:
                    # swap is desired, so own temperature will be set to received temperature
                    self.temperature = temp

            else:
                # prepare receiving data
                mail = np.empty(2)
                comm.Recv(mail, source=don, tag=swap_iteration)

                # unpack data
                temp, ll = mail
                # do comparison
                desired = self.compare_with_other(ll, temp)
                if desired:
                    # swap is desired, so current temperature of the chain will be sent
                    mail = np.array([self.temperature])
                    comm.Send(mail, dest=don, tag=swap_iteration)
                    # set own temperature to previously received temperature
                    self.temperature = temp

                else:
                    # no swap desired
                    mail = np.array([10.0])
                    comm.Send(mail, dest=don, tag=swap_iteration)
