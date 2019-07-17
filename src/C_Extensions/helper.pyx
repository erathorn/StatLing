
from cython.parallel import prange
import numpy as np
cimport numpy as np
import cython
DTYPE = np.float64
DTYPE64 = np.int64
DTYPE8 = np.int8
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE64_t
ctypedef np.int8_t DTYPE8_t


cdef extern from "math.h":
    double log(double x) nogil

cdef extern from "math.h":
    double sqrt(double x) nogil


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void array_creation(const double[:] vec1, const double[:] vec2, const long[:] ind1, const long[:] ind2, double[:,:] output_view):

    cdef int o = len(ind1)
    cdef Py_ssize_t i, x, y

    # vec 1 is frequency
    # vec 2 is evolutionary rate
    for i in prange(o, nogil=True):
        x = ind1[i]
        y = ind2[i]
        # set upper triangle
        output_view[x][y] = vec1[y]*vec2[i]
        # set lower triangle
        output_view[y][x] = vec1[x]*vec2[i]

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double dot(const double[:] vec, const double[:] vec2, const int m):
    cdef ssize_t i
    cdef double res = 0.0

    for i in prange(m, nogil=True):
        res += vec[i]*vec2[i]
    return res


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void inner_loop_felsenstein(double[:,:] state_probs_left,
                                  double[:,:] state_probs_right,
                                  int size,
                                  double[:,:] matrix_left,
                                  double[:,:] matrix_right,
                                  double[:,:] result):
        cdef int index
        cdef double l_z, l_o, r_z, r_o

        for index in prange(size, nogil = True):
            l_z = (state_probs_left[index][0] * matrix_left[0][0]) + (state_probs_left[index][1] * matrix_left[1][0])
            l_o = (state_probs_left[index][1] * matrix_left[1][1]) + (state_probs_left[index][0] * matrix_left[0][1])
            r_z = (state_probs_right[index][0] * matrix_right[0][0]) + (state_probs_right[index][1] * matrix_right[1][0])
            r_o = (state_probs_right[index][1] * matrix_right[1][1]) + (state_probs_right[index][0] * matrix_right[0][1])

            result[index][0] = l_z * r_z
            result[index][1] = l_o * r_o