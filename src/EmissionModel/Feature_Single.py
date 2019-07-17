# coding: utf-8
import collections
import csv
import itertools

import numpy as np
import scipy.linalg as sp_la
import scipy.linalg.lapack as sp_la_lp
import scipy.stats as spst
from numpy import linalg as np_la

import asjp_dolgo_map
import ipa_dolgo_map
import src.C_Extensions.Feature_Single_functions as c_feat
import src.C_Extensions.helper as c_help
import src.C_Extensions.sample as c_sam


class FeatureSoundsSingle(object):
    __slots__ = ("names", "_size", "frequencies", "evo_map", "evo_ind_map", "evo_values", "evo_paras", "_diri_weights",
                 "_freq_weights", "_iu", "_il", "_d", "pt_log", "freq_log", "fortran_matrix_1", "fortran_matrix_2",
                 "w", "q_eigenvectors", "q_eigenvectors_inv", "q_eigenvalues", "necessary", "b_mat", "d_mat", "_eye",
                 "evo_ind_list")

    def __init__(self, names, model):
        # type: (list, str) -> FeatureSoundsSingle

        self.names = names
        self._size = len(self.names)

        # initialize frequencies
        self.frequencies = np.random.uniform(size=self._size)
        self.frequencies /= np.sum(self.frequencies)

        # initialize evolutionary parameters
        self.evo_map = []
        self.create_evo_map(model)
        self.evo_ind_map = self.evo_indices(self.evo_map)

        self.evo_values = np.random.uniform(size=len(self.evo_ind_map.keys()) + 1)
        self.evo_values /= sum(self.evo_values)

        self.evo_paras = np.ones(len(self.evo_map), dtype=np.double)

        self.reset_evo_paras()

        self._diri_weights = [1] * len(self.evo_values)

        self._freq_weights = [1] * len(self.frequencies)

        # get matrix indices
        t1, t2 = np.triu_indices(self._size, 1)
        self._iu = np.asarray(t1, dtype=np.int, order="C")
        self._il = np.asarray(t2, dtype=np.int, order="C")
        self._d = np.diag_indices(self._size)

        # create necessary matrices
        self.pt_log = np.zeros((self._size, self._size), dtype=np.double, order="C")
        self.freq_log = np.zeros(self._size, dtype=np.double, order="C")
        self.fortran_matrix_1 = np.zeros((self._size, self._size), dtype=np.double, order="F")
        self.fortran_matrix_2 = np.zeros((self._size, self._size), dtype=np.double, order="F")
        self.w = np.zeros((self._size, self._size), dtype=np.double, order="F")
        self.q_eigenvectors = np.zeros((self._size, self._size), dtype=np.double, order="F")
        self.q_eigenvalues = None
        self.q_eigenvectors_inv = None

        # create necessity to recalculate Q
        self.necessary = True

        self.b_mat = np.zeros((self._size, self._size), dtype=np.double, order="C")
        self.d_mat = np.zeros((self._size, self._size), dtype=np.double, order="F")

        self._eye = np.eye(self._size)

    def evo_indices(self, ev_map):
        """
        Clean up the indices

        :param ev_map: index mapping
        :return: dictionary with mappin
        """
        evo_ind_map = collections.defaultdict()
        unique_values = set(ev_map)
        t = np.array(ev_map)
        for unique_value in unique_values:
            evo_ind_map[unique_value] = np.where(t == unique_value)
        self.evo_ind_list = [np.where(t == i) for i in sorted(unique_values)]
        return evo_ind_map

    def create_evo_map(self, model):
        # type: (str) -> None
        """
        This function creates a map, such that each sound pair is mapped to the respective
        Dolgopolsky class combination
        """

        # get Dolgopolsky classes
        if model == "ipa":
            classes = sorted(ipa_dolgo_map.maping.keys())
            rev_map = {i.decode("utf-8"): k for k, v in ipa_dolgo_map.maping.iteritems() for i in v}
        elif model == "asjp":
            classes = sorted(asjp_dolgo_map.mapping_classes.keys())
            rev_map = {i: k for k, v in asjp_dolgo_map.mapping_classes.iteritems() for i in v}
        else:
            raise Exception("This is really bad")

        # get necessary class combinations
        class_combs = list(itertools.combinations(classes, 2))

        class_combs += [("eq", "eq"), ("V", "R")]

        class_combs = [i for i in class_combs if 0 not in i]
        self.evo_map = []

        # set map
        for i, j in itertools.combinations(self.names, 2):

            ci = rev_map[i]
            cj = rev_map[j]
            if ci == cj:
                self.evo_map.append(class_combs.index(("eq", "eq")))
            else:
                if ci == 0 or cj == 0:
                    self.evo_map.append(class_combs.index(("V", "R")))
                else:
                    try:
                        self.evo_map.append(class_combs.index((ci, cj)))
                    except ValueError:
                        try:
                            self.evo_map.append(class_combs.index((cj, ci)))
                        except ValueError:
                            raise Exception

    def diagonalize_q_matrix(self):
        # type: () -> None
        """
        calculate Q matrix and eigenvalues and eigenvectors if necessary
        c.f. Felsenstein - Inferring Phylogenies (chapter 13)
        The naming of the variables follows the naming in Felsenstein.
        """

        # if self.necessary:
        self.freq_log = np.log(self.frequencies)

        self.create_q_c()

        # calculate root
        self.d_mat = np.sqrt(self.d_mat)

        self.b_mat = np.asfortranarray(self.b_mat)
        d_b_d = np.empty((self._size, self._size), dtype=np.float64, order="F")

        # get symmetric matrix
        # use SciPys BLAS wrappers
        c = sp_la.blas.dgemm(alpha=1.0, a=self.d_mat, b=self.b_mat)
        sp_la.blas.dgemm(alpha=1.0, a=c, b=self.d_mat, c=d_b_d, overwrite_c=1)

        # get eigenvalues and eigenvectors
        self.q_eigenvalues, u_mat = np_la.eigh(d_b_d)

        # calculate actual eigenvectors of Q-Matrix
        # use SciPys BLAS wrappers
        u_mat = np.asfortranarray(u_mat)
        sp_la.blas.dgemm(alpha=1.0, a=self.d_mat, b=u_mat, c=self.q_eigenvectors, overwrite_c=1)

        self.q_eigenvectors_inv = sp_la_lp.dgesv(self.q_eigenvectors, self._eye)[2]

        self.q_eigenvectors_inv = np.asfortranarray(self.q_eigenvectors_inv)
        self.necessary = False

    def set_exponential(self, time):
        # type: (float) -> None
        """
        this function calculates the matrix exponential of the Q-matrix

        :param time: evolutionary time
        """

        # use SciPys BLAS wrappers (in cython) for the dot product

        c_feat.double_dgemm(self.q_eigenvectors, self.w, self.fortran_matrix_1, self.q_eigenvectors_inv,
                            self.fortran_matrix_2, self._size, time, self.q_eigenvalues)

        self.pt_log = np.transpose(np.log(self.fortran_matrix_2))

    def revert(self, frequency=None, evo=None):
        # type: (list, list) -> None
        """
        This function reverts the suggested move

        :param frequency: frequency distribution
        :type frequency: list
        :param evo: evolutionary parameters
        :type evo: list
        """

        # recalculation of eigenvectors and eigenvalues is necessary
        self.necessary = True

        if frequency is not None:
            self.frequencies = np.array(frequency)

        if evo is not None:
            self.evo_values = np.array(evo)
            self.reset_evo_paras()

    def set_frequency(self, f):
        old = np.array([i for i in self.frequencies])
        f /= sum(f)
        self.frequencies = f
        self.necessary = True
        return old

    def set_evovals(self, f):
        old = np.array([i for i in self.evo_values])
        f /= sum(f)
        self.evo_values = f
        self.necessary = True
        self.reset_evo_paras()
        return old

    def sample_frequency(self, d_max, index):
        # type: (float, int) -> (list, float,float, list)
        """
        sample a new frequency vector

        :param d_max: window size
        :type d_max: float
        :return: new frequency vector and Metropolis Hastings Ratio and old parameters
        :rtype: (list, float,float, list)
        """

        prev_prior = spst.dirichlet.logpdf(self.frequencies, self._freq_weights)
        f, m_h_ratio, old = c_sam.sample_frequency(d_max=d_max, f=self.frequencies, index=index)

        new_prior = spst.dirichlet.logpdf(self.frequencies, self._freq_weights)
        self.necessary = True
        prior = new_prior - prev_prior
        return f, m_h_ratio, prior, old

    def sample_evo(self, d_max, index):
        # type: (float, int) -> (list, float,float, list)
        """
        sample new evolutionary parameters

        :param d_max: window size
        :type d_max: float
        :return: new list of evolutionary parameters and Metropolis Hastings Ratio and old parameters
        :rtype: (list, float,float, list)
        """

        prev_prior = spst.dirichlet.logpdf(self.evo_values, self._diri_weights)
        a, m_h_ratio, old = c_sam.sample_frequency(d_max=d_max, f=self.evo_values, index=index)

        new_prior = spst.dirichlet.logpdf(self.evo_values, self._diri_weights)
        self.reset_evo_paras()

        self.necessary = True
        prior = new_prior - prev_prior
        return a, m_h_ratio, prior, old

    def reset_evo_paras(self):
        # type: () -> None
        """
        This function sets the evolutionary parameters from the map and the actual values from the class
        """

        for inds, v in zip(self.evo_ind_list, self.evo_values):
            self.evo_paras[inds] = v

    def create_q_c(self):
        """
        This function creates a Q-Matrix for the the sound model, as well as two matrices which can be used
        to calculate the matrix exponential from symmetric matrices (c.f. Felsenstein -  Inferring Phylogenies)

        The code is written in cython for speed up
        """
        self.b_mat[:, :] = 0.0
        self.d_mat[:, :] = 0.0
        c_feat.create_q(self.b_mat, self.d_mat, self._size, self.frequencies, self.evo_paras, self._iu, self._il)

    def create_q(self):
        """
        This function creates a Q-Matrix for the the sound model, as well as two matrices which can be used
        to calculate the matrix exponential from symmetric matrices (c.f. Felsenstein -  Inferring Phylogenies)

        Remains for legacy reasons
        """

        # initialize Q and evo_matrix

        self.b_mat.fill(0.0)
        freq = self.frequencies
        # diagonal matrix of frequencies
        self.d_mat[self._d] = freq

        c_help.array_creation(freq, self.evo_paras, self._iu, self._il, self.b_mat)

        # get diagonals of Q-matrix
        diagonals = -np.sum(self.b_mat, axis=0)

        # set diagonals
        self.b_mat[self._d] = diagonals

        # get normalizing value
        norm = -1.0 / np.dot(freq, diagonals)

        # normalize
        self.b_mat *= norm

        # factor out frequencies from b_mat, so d*b = q
        self.b_mat[self._d] /= freq

    def to_file(self, file_name, header=True):
        # type: (str, bool) -> None
        """
        Writes current state of the sound model to a file

        :param file_name: filename to write to
        :param header: should the header be written
        """

        # collect the relevant information
        row = self.frequencies.tolist() + self.evo_values.tolist()

        # write header
        if header:
            headline = ["freq_" + str(i) for i in self.names] + ["clv_" + str(i) for i in range(len(self.evo_values))]
            with open(file_name, "a") as my_file:
                writer = csv.writer(my_file, delimiter="\t", lineterminator='\n')
                writer.writerow(headline)
            headline = ["evocl_" + str(i) for i in range(len(self.evo_map))]
            mapping = [self.evo_map[i] for i in range(len(self.evo_map))]
            with open(file_name + "_classes", "a") as my_file:
                writer = csv.writer(my_file, delimiter="\t", lineterminator='\n')
                writer.writerow(headline)
                writer.writerow(mapping)
        # write current state
        with open(file_name, "a") as my_file:
            writer = csv.writer(my_file, delimiter="\t", lineterminator='\n')
            writer.writerow(row)

    def to_dict(self):
        # type: () -> dict
        """
        save current state of the model as a dictionary

        :return: dictionary of current state
        """
        return {"names": self.names,
                "evo_vals": self.evo_values,
                "freqs": self.frequencies,
                "evo_map": self.evo_map}

    @classmethod
    def from_dict(cls, dct):
        # type: (dict) -> FeatureSoundsSingle
        """
        create an instance of FeatureSoundsSingle from a dictionary

        :param dct: dictionary with parameter values
        :return: instance of FeatureSoundsSingle
        """
        new_class = cls(names=dct["names"], model=dct["model"])
        new_class.evo_map = dct["evo_map"]
        new_class.frequencies = dct["freqs"]
        new_class.evo_values = dct["evo_vals"]
        new_class.reset_evo_paras()
        return new_class
