
from libc.stdlib cimport rand, srand

import numpy as np
cimport numpy as np
DTYPE = np.float64
DTYPE64 = np.int64
DTYPE8 = np.int8
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE64_t
ctypedef np.int8_t DTYPE8_t



cdef extern from "limits.h":
    int RAND_MAX


# get logarithm from c library
cdef extern from "math.h":
    double log(double x)
    double exp(double x)
    double INFINITY


#!python

cpdef void set_srand_seed(int seed):
    srand(seed)




cpdef tuple sample_exponential(double d_max, double lambd, double time):
    cdef int sampled = 0
    cdef double left = max(0.0, time - d_max)
    cdef double span = time + d_max - left
    cdef double ratio, old_val, randnum


    old_val = time
    while sampled==0:
        randnum = rand()
        time = left + (randnum / RAND_MAX) * span
        if time != 1:
            if time > 0:
                sampled = 1
    ratio = log(expontial_pdf(lambd, time) / expontial_pdf(lambd, old_val))
    return time, ratio, old_val



cdef double expontial_pdf(double lambd, double x):
    if x>=0:
        return lambd*exp(-lambd*x)
    else:
        return 0



cdef void sample_uniform(double x, double d_max, double[:] output, double lower = 0.0, double upper=INFINITY):
    cdef int sampled = 0
    cdef unsigned long i = 0
    cdef double new_val, ratio, rand_num, span
    cdef double left = x - d_max


    if left < lower:
        left = lower
    if x+d_max > upper:
        span = upper - left
    else:
        span = x + d_max - left

    while sampled == 0:
        rand_num = rand()
        new_val = left + (rand_num / RAND_MAX) * span
        if lower < new_val < upper:
            sampled = 1
            x = new_val

    ratio = log((new_val + d_max - max(lower, new_val - d_max)) / span)
    output[0] = x
    output[1] = ratio


cpdef tuple sample_uniform_in_list(double d_max, list a, double lower = 0.0, double upper=INFINITY):
    """
    sample new evolutionary parameters
    :param d_max: window size for sampling 
    :type d_max: float
    :param a: current evolutionary parameters
    :type a: list
    :return: new parameters, metropolis hastings ratio, old parameters
    :rtype: (list, float, list)
    """

    cdef int k = rand() % len(a)
    cdef list oldparas = [0.0 for i in range(len(a))]
    cdef double ratio
    cdef double output[2]

    for i in range(len(a)):
        oldparas[i] = a[i]
    sample_uniform(a[k], d_max, output, lower, upper)
    a, ratio = output
    return a, ratio, oldparas

cpdef tuple sample_uniform_single(double d_max, double r, double lower, double upper):
    """
    sample new evolutionary rate parameter
    :param d_max: maximal window size 
    :type d_max: float|double
    :param r: current rate parameter
    :type r: float|double
    :return: new rate parameter, metropolis hastings ratio, old parameter
    :rtype: (float, float, float)
    """

    cdef double ratio, old_val
    cdef double output[2]
    old_val = r
    sample_uniform(r, d_max, output, lower, upper)
    r, ratio = output
    return r, ratio, old_val


cpdef tuple sample_frequency(double d_max, np.ndarray[np.float64_t, ndim=1] f, int index):
    """
    sample equilibrium frequencies
    :param d_max: window size for sampling 
    :type d_max: float
    :param f: old frequency parameters
    :type f: list
    :return: new frequency parameters, metropolis hastings ratio, old parameters
    :rtype: (list, float, list)
    """


    cdef double* result

    cdef double nval, ratio

    cdef unsigned long i, j
    cdef list oldparas = [0.0 for i in range(len(f))]
    cdef int sampled = 1
    cdef double output[2]

    for i in range(len(f)):
        oldparas[i] = f[i]

    while sampled == 1:
        sample_uniform(f[index], d_max, output, 0.0, 1.0)
        nval, ratio = output
        f[index] = nval
        sampled = 0
        for i in range(len(f)):
            if i != index:  # do not set it for the sampled parameter
                nparam = f[i] * (1.0 - nval) / (1.0 - oldparas[index])  # calculate new parameters
                if 0 < nparam < 1: # check if these parameters are in the correct range
                    f[i] = nparam
                else:
                    for j in range(len(f)):
                        f[j] = oldparas[j]
                    sampled = 1
                    break


    return f, ratio, oldparas


