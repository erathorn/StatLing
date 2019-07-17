"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import random

import numpy as np

import src.SamplerSettings.Prior as Prior


class EmissionModelSound(object):
    __slots__ = ("alphabet", "alpha_size", "sound_model", "evo", "gap_log", "emission_mat_log", "set_exp_sound",
                 "sampling_functions_classes", "iterator", "sampling_functions_frequencies")

    def __init__(self, alphabet, sound_mod):
        """
        This class provides the interface for the sound model and creates the emission model

        :param alphabet: alphabet of used symbols
        :param sound_mod: Sound model to use

        """

        self.alphabet = alphabet
        self.alpha_size = len(self.alphabet)

        self.sound_model = sound_mod

        self.evo = self.sound_model.evo_paras

        # initialize empty gap and matrix values
        self.gap_log = np.zeros(self.alpha_size, np.double, order="C")
        self.emission_mat_log = np.zeros((self.alpha_size, self.alpha_size), np.double, order="C")

        self.set_exp_sound = self.sound_model.set_exponential

        self.sampling_functions_classes = [(self.sample_evo, self.revert, "evo", {})]
        self.sampling_functions_frequencies = [(self.sample_frequency, self.revert, "frequency", {})]

    def __iter__(self):
        # type: () -> EmissionModelSound
        """
        provides iterator

        """

        self.iterator = [self.sound_model]
        return self

    def next(self):
        # type: () -> tuple|StopIteration
        """
        provides next element in iterator

        """

        while self.iterator:
            k = self.iterator.pop()
            return 0, k

        raise StopIteration

    def set_exp(self, time):
        # type: (float) -> None
        """
        This functions passes on the time parameter to the underlying sound model

        :param time: evolutionary time parameter
        :type time: float
        """
        self.sound_model.set_exponential(time=time)

    def calculate_emission_mat_log(self):
        # type: () -> None
        """
        This functions calculates the the emission matrix for the PHMM and the gap emission
        """

        p_mat = self.sound_model.pt_log

        # here the gap emission probabilities are set
        self.gap_log = self.sound_model.freq_log

        self.emission_mat_log = np.add(p_mat.T, self.sound_model.freq_log).T

    def create_model(self, time):
        # type: (float) -> None
        """

        :param time: evolutionary time
        :type time: float
        """

        self.sound_model.set_exponential(time=time)

        self.calculate_emission_mat_log()

    def sample_frequency(self):
        # type: () -> (bool, float, (str, tuple))
        """
        This function passes on frequency sampling to underlying sound model
        """
        index = random.randint(0, self.alpha_size - 1)
        _, ratio, prior, old = self.sound_model.sample_frequency(d_max=Prior.dmax_freq, index=index)

        return False, ratio, prior, ("freq", old)

    def sample_evo(self):
        # type: () -> (bool, float, (str, tuple))

        index = random.randint(0, len(self.sound_model.evo_values) - 1)
        _, ratio, prior, old = self.sound_model.sample_evo(d_max=Prior.dmax_evo, index=index)

        return False, ratio, prior, ("evo", old)

    def revert(self, move, old):
        # type: (str, tuple) -> None
        """
        This function passes on the revert move information to the underlying sound model

        :param move: identifier which move to revert
        :param old: old information
        """
        if move == "freq":
            self.sound_model.revert(frequency=old)
        elif move == "evo":
            self.sound_model.revert(evo=old)
        elif move == "evo_second":
            self.sound_model.revert(evo_second=old)
        else:
            raise Exception("unknown move")
