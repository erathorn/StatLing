
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

cdef class Transition_km03:
    """
    This class implements the transition model described in Knudsen & Miyamoto 2003
    """

    def __cinit__(self, double r, double time, double a, double dmax_a , double dmax_r):
        """

        :param r: insertion rate
        :type r: double
        :param time: evolutionary time
        :type time: double
        :param a: parameter of geometric distribution
        :type a: double
        """
        self.parameters = {}
        self.r = r
        self.mu = -1
        self.l = -1
        self.time = time
        if a is not None:
            assert a != 1.0
            assert a > 0
            self.a = a

        self.choice_list = ["a", "r"]

        if None not in (self.r, self.time, self.a):
            self.calculate()
        self.dmax_r = dmax_r
        self.dmax_a = dmax_a
        self.sampling_functions = [(self.sample_a, self.revert_a, "trans",{}),
                                   (self.sample_r, self.revert_r, "trans",{})]

    def __iter__(self):
        self.count = 0
        self.iterator = [("a", self.a), ("r",self.r)]
        return self

    def __next__(self):
        if self.count == len(self.iterator):
            raise StopIteration
        self.count += 1
        return self.iterator[self.count - 1]

    @property
    def T10(self):
        return self._T10

    @T10.setter
    def T10(self, value):
        self._T10 = value

    @property
    def T20(self):
        return self._T10

    @property
    def TS1(self):
        return self._T01

    @property
    def T01(self):
        return self._T01

    @T01.setter
    def T01(self, value):
        self._T01 = value

    @property
    def T12(self):
        return self._T12

    @T12.setter
    def T12(self, value):
        self._T12 = value

    @property
    def T00(self):
        return self._T00

    @T00.setter
    def T00(self, value):
        self._T00 = value

    @property
    def T22(self):
        return self._T11

    @property
    def T11(self):
        return self._T11

    @T11.setter
    def T11(self, value):
        self._T11 = value

    @property
    def T02(self):
        return self._T02

    @T02.setter
    def T02(self, value):
        self._T02 = value

    @property
    def T00_log(self):
        return self._T00_log

    @T00_log.setter
    def T00_log(self, value):
        self._T00_log = value

    @property
    def T01_log(self):
        return self._T01_log

    @T01_log.setter
    def T01_log(self, value):
        self._T01_log = value

    @property
    def T02_log(self):
        return self._T02_log

    @T02_log.setter
    def T02_log(self, value):
        self._T02_log = value

    @property
    def T0E_log(self):
        return self._T00_log

    @property
    def T10_log(self):
        return self._T10_log

    @T10_log.setter
    def T10_log(self, value):
        self._T10_log = value

    @property
    def T11_log(self):
        return self._T11_log

    @T11_log.setter
    def T11_log(self, value):
        self._T11_log = value

    @property
    def T12_log(self):
        return self._T12_log

    @T12_log.setter
    def T12_log(self, value):
        self._T12_log = value

    @property
    def T1E_log(self):
        return self._T10_log

    @property
    def T20_log(self):
        return self._T10_log

    @property
    def T21_log(self):
        return self._T12_log

    @property
    def T22_log(self):
        return self._T11_log

    @property
    def T2E_log(self):
        return self._T10_log

    @property
    def TS0_log(self):
        return self._T00_log

    @property
    def TS1_log(self):
        return self._T01_log

    @property
    def TS2_log(self):
        return self._T02_log

    @property
    def T2E(self):
        return self._T10

    @property
    def T0E(self):
        return self._T00

    @property
    def T1E(self):
        return self._T10

    @property
    def TS0(self):
        return self._T00

    @property
    def T21(self):
        return self._T12

    @property
    def TS2(self):
        return self._T02

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
    def indel(self):
        return self._indel

    @indel.setter
    def indel(self, value):
        self._indel = value

    @property
    def indel_prime(self):
        return self._indel_prime

    @indel_prime.setter
    def indel_prime(self, value):
        self._indel_prime = value


    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, value):
        self._l = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value
        self.parameters["r"] = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value
        self.parameters["a"] = value

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
        return np.array([self.T00_log, self.T01_log, self.T01_log, self.T0E_log,
                         self.T10_log, self.T11_log, self.T12_log, self.T1E_log,
                         self.T10_log, self.T12_log, self.T11_log, self.T1E_log,
                         self.T00_log, self.T01_log, self.T01_log, 0.0,
                         ])



    cdef double E10(self):
        cdef double a_, i
        a_ = ((self.a * (1 - self.a)) / (2 + 2 * self.a))
        i = ((7 - 7 * self.a) / 8)
        return (1 - self.a) + a_ * self.indel_prime - i * self.indel

    cdef double E11(self):
        cdef double sqare_a_ , a_
        sqare_a_ = ((self.a ** 2) / (1 - self.a ** 2))
        a_ = (1 - self.a) / 2
        return self.a + sqare_a_ * self.indel_prime + a_ * self.indel

    cdef double E12(self):
        cdef double square_a_, a_
        square_a_ = self.a ** 2 / (2 + 2 * self.a)
        a_ = (3 - 3 * self.a) / 8
        return square_a_ * self.indel_prime + a_ * self.indel

    cdef double E12_log(self):
        cdef double sprime, sprime2, aprime
        sprime = (3.0 * self.a ** 2 - 3.0) * self.indel
        sprime2 = 4.0 * self.a ** 2 * self.indel_prime
        aprime = sprime - sprime2
        return log(-aprime) - 3.0 * log(2)

    cdef double E1(self):
        return 1 + (self.a / (2 - 2 * self.a)) * self.indel_prime

    cdef double E1_log(self):
        cdef double en, de
        en = self.a * (self.indel_prime - 2.0) + 2.0
        de = 2.0 * (self.a - 1.0)
        return log(-en / de)

    cdef double E00(self):
        cdef double fp
        fp = ((1 - self.a) / (4 + 4 * self.a)) * self.indel_prime
        fp = 1 - fp
        fp *= self.indel
        return 1 - fp


    cpdef void set_model(self, double time):
        #cdef double a_, r_, time_
        #if paras["a"] is not None:
        #    a_ = paras["a"]
        #    assert a_ != 1.0
        #    assert a_ > 0
        #    self.a = a_
        #if paras["r"] is not None:
        #    r_ = paras["r"]
        #    self.r = r_
        #if paras["time"] is not None:
        #    time_ = paras["time"]
        self.time = time

        self.calculate()

    cdef void calculate(self):
        """
        calculate probability of event (indel) and successive event (indel_prime)
        """

        #assert self.a is not None
        #assert self.r is not None
        #assert self.time is not None
        cdef double two_rt, temp_exp, e00, e00_log, e1, e1_log, e11, e11_log, e12, e12_log
        #self.r = 0.1
        two_rt = 2.0 * self.r * self.time
        temp_exp = exp(-two_rt)
        self.indel = 1.0 - temp_exp
        self.indel_prime = 1.0 - (self.indel / two_rt)

        '''
        calculate expected transitions
        '''
        #e00 = self.E00()
        #e00_log = log(e00)
        #
        #e10 = self.E10()
        #e10_log = log(e10)
        #
        #e11 = self.E11()
        #e11_log = log(e11)
        #
        #e12 = self.E12()
        #e12_log = self.E12_log()
        #
        #e1 = self.E1()
        #e1_log = self.E1_log()

        '''
        set transitions
        '''

        self.T00 = 1.0 - self.indel
        self.T01 = 0.5 * self.indel
        self.T02 = 0.5 * self.indel
        # TransitionModel out of the match state
        #self.T00 = e00
        #self.T01 = 0.5 * (1 - self.T00)
        #self.T02 = self.T01

        #normer = self.T00+self.T01+self.T02

        #self.T00 /= normer
        #self.T01 /= normer
        #self.T02 /= normer

        #self.T00_log = e00_log
        #self.T01_log = log(1.0 - self.T00) - log(2)
        #self.T02_log = self.T01_log

        self.T00_log = log(self.T00)
        self.T01_log = log(self.T01)
        self.T02_log = log(self.T02)




        # TransitionModel out of first indel state

        self.T10 = (1.-self.a)*(1-self.indel)
        self.T11 = self.a
        self.T12 = (1.-self.a)*self.indel
        #self.T10 = e10 / e1
        #self.T11 = e11 / e1
        #self.T12 = e12 / e1

        #normer = self.T10 + self.T11#+ self.T12

        #self.T10 /= normer
        #self.T11 /= normer
        #self.T12 = 0.0

        self.T10_log = log(self.T10)
        self.T11_log = log(self.T11)
        self.T12_log = log(self.T12)

        #self.T10_log = e10_log - e1_log
        #self.T11_log = e11_log - e1_log
        #self.T12_log = e12_log - e1_log


    cpdef tuple sample_r(self):
        """
        sample rate parameter
        :param dmax: window size
        :type dmax: double
        :return: new value, metropolis hastings ratio, old value
        :rtype: (double, double, double)
        """

        cdef tuple res = c.sample_uniform_single(self.dmax_r, self.r, 0.0, 1.0)
        self.r = res[0]

        return False, res[1],0.0, res[2]

    cpdef tuple sample_a(self):
        """
        
        :param dmax: window size
        :type dmax: double 
        :return: new value, metropolis hastings ratio, old value
        :rtype: (double, double, double) 
        """
        cdef tuple res = c.sample_uniform_single(self.dmax_a, self.a, 0.0, 1.0)
        self.a = res[0]
        #self.old = res[2]
        #self.calculate()
        return False, res[1],0.0, res[2]

    cpdef tuple sample_wrap(self, str feat):
        """
        
        :param feat: feature to sample
        :type feat: str
        :return: new value, metropolis hastings ratio, old value
        :rtype: (double, double, double) 
        """
        if feat == "a":
            return self.sample_a()
        if feat == "r":
            return self.sample_r()

    cpdef void revert_r(self, double old):
        """
        This function reverts the suggested move
        :param r: rate parameter
        :type r: double
        :param a: parameter of geometric distribution
        :type a: double
        :param time: evolutionary time
        :type time: double
        :return: 
        :rtype: None
        """

        self.r = old

    cpdef void revert_a(self, double old):
        """
        This function reverts the suggested move
        :param r: rate parameter
        :type r: double
        :param a: parameter of geometric distribution
        :type a: double
        :param time: evolutionary time
        :type time: double
        :return: 
        :rtype: None
        """

        self.a = old

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
        row = [self.a, self.r]
        if header == 1:
            head = ["Trans_a", "Trans_r"]
            with open(file_name, "a") as fout:
                writer = csv.writer(fout, delimiter="\t",lineterminator='\n')
                writer.writerow(head)
                writer.writerow(row)
        else:
            with open(file_name, "a") as fout:
                writer = csv.writer(fout, delimiter="\t",lineterminator='\n')
                writer.writerow(row)

