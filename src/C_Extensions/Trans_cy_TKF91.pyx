
"""
@author: Johannes Wahle
@date: March 2017
@version: 1.0
"""

import src.C_Extensions.sample as c
cimport src.C_Extensions.sample as c
import csv
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double log(double)

cdef extern from "math.h":
    double exp(double)

cdef extern from "math.h":
    double INFINITY

cdef class Transition_TKF91:
    """
    This class implements the transition model described in Thorne, Kishino & Felsenstein 1991
    """

    def __cinit__(self, double time, double lambd, double mu,  double dmax):
        """

        :param lambd: lambda
        :type lambd: double
        :param mu: mu
        :type mu: double
        :param time: evolutionary time
        :type time: double
        :param dmax: sampling width
        :type dmax: double
        """
        self.parameters = {}

        self.time = time
        self.dmax = dmax
        assert mu > lambd
        self.mu = mu
        self.l = lambd

        self.choice_list = ["r", "mu", "lambda"]

        self.calculate()

        self.sampling_functions = [(self.sample_mu, self.revert_mu, "trans",{}),
                                   (self.sample_l, self.revert_l, "trans",{})]


    def __iter__(self):
        self.count = 0
        self.iterator = [("lambda", self.l), ("mu", self.mu)]
        return self

    def __next__(self):
        if self.count == len(self.iterator):
            raise StopIteration
        self.count += 1
        return self.iterator[self.count - 1]

    @property
    def T00(self):
        return self._T00

    @T00.setter
    def T00(self, value):
        self._T00 = value

    @property
    def T01(self):
        return self._T01

    @T01.setter
    def T01(self, value):
        self._T01 = value

    @property
    def T02(self):
        return self._T02

    @T02.setter
    def T02(self, value):
        self._T02 = value

    @property
    def T0E(self):
        return self._T0E

    @T0E.setter
    def T0E(self, value):
        self._T0E = value

    @property
    def T10(self):
        return self._T10

    @T10.setter
    def T10(self, value):
        self._T10 = value

    @property
    def T11(self):
        return self._T11

    @T11.setter
    def T11(self, value):
        self._T11 = value

    @property
    def T12(self):
        return self._T12

    @T12.setter
    def T12(self, value):
        self._T12 = value

    @property
    def T1E(self):
        return self._T1E

    @T1E.setter
    def T1E(self, value):
        self._T1E = value

    @property
    def T20(self):
        return self._T20

    @T20.setter
    def T20(self, value):
        self._T20 = value

    @property
    def T21(self):
        return self._T21

    @T21.setter
    def T21(self, value):
        self._T21 = value

    @property
    def T22(self):
        return self._T22

    @T22.setter
    def T22(self, value):
        self._T22 = value

    @property
    def T2E(self):
        return self._T2E

    @T2E.setter
    def T2E(self, value):
        self._T2E = value

    @property
    def TS0(self):
        return self._TS0

    @TS0.setter
    def TS0(self, value):
        self._TS0 = value

    @property
    def TS1(self):
        return self._TS1

    @TS1.setter
    def TS1(self, value):
        self._TS1 = value

    @property
    def TS2(self):
        return self._TS2

    @TS2.setter
    def TS2(self, value):
        self._TS2 = value

    @property
    def TSE(self):
        return self._TSE

    @TSE.setter
    def TSE(self, value):
        self._TSE = value

    """
    now the log properties
    """

    @property
    def T00_log(self):
        return log(self._T00)

    @property
    def T01_log(self):
        return log(self._T01)

    @property
    def T02_log(self):
        return log(self._T02)

    @property
    def T0E_log(self):
        return log(self._T0E)

    @property
    def T10_log(self):
        return log(self._T10)

    @property
    def T11_log(self):
        return log(self._T11)

    @property
    def T12_log(self):
        return log(self._T12)

    @property
    def T1E_log(self):
        return log(self._T1E)

    @property
    def T20_log(self):
        return log(self._T20)

    @property
    def T21_log(self):
        return log(self._T21)

    @property
    def T22_log(self):
        return log(self._T22)

    @property
    def T2E_log(self):
        return log(self._T2E)

    @property
    def TS0_log(self):
        return log(self._TS0)

    @property
    def TS1_log(self):
        return log(self._TS1)

    @property
    def TS2_log(self):
        return log(self._TS2)

    @property
    def TSE_log(self):
        return log(self._TSE)

    @property
    def choice_list(self):
        return self._cl

    @choice_list.setter
    def choice_list(self, value):
        self._cl = value

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, value):
        self._params = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value
        self.parameters["r"] = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self.parameters["mu"] = value

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, value):
        self._l = value
        self.parameters["lambda"] = value

    @property
    def sampling_functions(self):
        return self._sampling_functions

    @sampling_functions.setter
    def sampling_functions(self, value):
        self._sampling_functions = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.parameters["time"] = value

    @property
    def trans_vals(self):
        return np.array([self.T00_log, self.T01_log, self.T02_log, self.T0E_log,
                         self.T10_log, self.T11_log, self.T12_log, self.T1E_log,
                         self.T20_log, self.T21_log, self.T22_log, self.T2E_log,
                         self.TS0_log, self.TS1_log, self.TS2_log, self.TSE_log,
                         ])

    cpdef void set_model(self, double time):

        self.time = time

        self.calculate()

    cdef void calculate(self):

        cdef double _B1, _B, mu_, exp1, lb_, T00, T01, T02, T10, T11, T12, T20, T21, T22, TS0, TS1, TS2, TSE, T0E, T1E, T2E

        _B1 = exp((self.l-self.mu)*self.time)
        _B = (1.0-_B1)/(self.mu-self.l*_B1)

        mu_ = (self.l / self.mu)
        exp1 = exp(-self.mu * self.time)
        lb_ = (1.0 - self.l * _B)

        T00 = (1.0 - self.l * _B) * mu_ * exp1
        T02 = lb_ * mu_ * (1 - exp1)
        T01 = self.l * _B
        T10 = (self.l * _B / (1.0 - exp1)) * exp1
        T11 = self.l * _B
        T12 = (1.0 - self.mu * _B / (1.0 - exp1))
        T20 = lb_ * self.l / self.mu * exp1
        T21 = lb_ * self.l / self.mu * (1.0 - exp1)
        T22 = self.l * _B
        TS0 = lb_ * self.l / self.mu * exp1
        TS1 = lb_ * self.l / self.mu * (1.0 - exp1)
        TS2 = self.l*_B
        TSE = lb_ * (1.0 - self.l / self.mu)
        T0E = (1.0 - (self.l * _B)) * (1.0 - mu_)
        T1E = (self.mu * _B / (1.0 - exp1)) * (1.0 - self.l / self.mu)
        T2E = lb_ * (1.0 - self.l / self.mu)

        self.T00 = T00#/(mu_**2)
        self.T01 = T01#(T01*T10)/(mu_*T00)
        self.T02 = T02#(T02*T20)/(mu_*T00)

        self.T10 = T10
        self.T20 = T20

        self.T11 = T11#/mu_
        self.T22 = T22#/mu_

        self.T12 = T12#/mu_
        self.T21 = T21#/mu_

        self.T0E = T0E
        self.T1E = T1E#(T00/T10)*T1E
        self.T2E = T2E#(T00/T20)*T2E

        self.TS0 = TS0#(1-mu_)**2
        self.TS1 = TS1#s0
        self.TS2 = TS2
        self.TSE = TSE

    cpdef tuple sample_l(self):
        """
        Sample a new lambda parameter
         
        :return: new value, metropolis hastings ratio, old value
        :rtype: (double, double, double) 
        """
        cdef tuple res = c.sample_uniform_single(self.dmax, self.l, 0.0, self.mu)

        self.l = res[0]

        return False, res[1],0.0, res[2]

    cpdef tuple sample_mu(self):
        """
        Sample a new mu parameter
         
        :return: new value, metropolis hastings ratio, old value
        :rtype: (double, double, double) 
        """
        cdef tuple res = c.sample_uniform_single(self.dmax, self.mu, self.l, 1.0)

        self.mu = res[0]

        return False, res[1],0.0, res[2]

    cpdef void revert_mu(self, double old):
        """
        This function reverts the suggested move
        
        :param old: rate parameter
        :type old: double
        """

        self.mu = old

    cpdef void revert_l(self, double old):
        """
        This function reverts the suggested move
        
        :param old: rate parameter
        :type old: double
        """

        self.l = old

    cpdef void to_file(self, int header, str file_name):
        """
        
        :param header: switch on header printing
        :type header: int 
        :param file_name: filename of target file
        :type file_name: str
        :return: 
        :rtype: None
        """
        cdef list head, row

        row = [self.mu, self.l]

        if header == 1:
            head = ["Trans_mu", "Trans_lambd"]
            with open(file_name, "a") as fout:
                writer = csv.writer(fout, delimiter="\t",lineterminator='\n')
                writer.writerow(head)
                writer.writerow(row)

        else:
            with open(file_name, "a") as fout:
                writer = csv.writer(fout, delimiter="\t",lineterminator='\n')
                writer.writerow(row)
