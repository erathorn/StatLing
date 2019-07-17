
import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_blas as sp_la

cimport cython

import src.C_Extensions.helper as c_help
cimport src.C_Extensions.helper as c_help
from cython.parallel import prange

from libc.math cimport exp, log
from libc.stdlib cimport malloc, free


DTYPE = np.float64
DTYPE64 = np.int64
DTYPE8 = np.int8
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE64_t
ctypedef np.int8_t DTYPE8_t

#
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
cpdef void create_q(double[:,:] b_mat_view, double[:,:] d_mat_view,
                    int size, double[:] freq_view,
       double[:] evo_view,
       long[:] iu, long[:] il):

    """
    This function creates a Q-Matrix for the the sound model, as well as two matrices which can be used
    to calculate the matrix exponential from symmetric matrices (c.f. Felsenstein -  Inferring Phylogenies)
    """

    # initialize Q and evo_matrix

    cdef ssize_t i,j
    cdef double carry, norm
    cdef double[:] diag = np.zeros(size, dtype=DTYPE, order="c")

    # diagonal matrix of frequencies
    c_help.array_creation(freq_view, evo_view, iu, il, b_mat_view)

    # get diagonals of Q-matrix
    for i in range(size):
        carry = 0
        for j in range(size):
            if i == j:
                b_mat_view[i,j] = 0.0
            carry += b_mat_view[i,j]
        diag[i] = -1.0*carry

    # set diagonals
    for i in range(size):

        d_mat_view[i][i] = freq_view[i]
        b_mat_view[i][i] = diag[i]

    # get normalizing value

    norm = -1.0/ c_help.dot(freq_view, diag, size)

    for i in range(size):
        for j in range(size):
            b_mat_view[i,j] *= norm
            #if i != j:
            b_mat_view[i,j] /= freq_view[j]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
cdef void dgemm_thread_2D(double[::1,:] eigenvecs, double[::1,:] eigenvecs_inv,
                       int size, double time, double[:] eigenvals, double* out, double[:] freqs) nogil:

    cdef double alpha, beta
    cdef ssize_t i,j
    cdef int* size_ptr = &size
    cdef double* alpha_ptr = &alpha
    cdef double* beta_ptr = &beta
    cdef double* mat1_ptr = &eigenvecs[0,0]
    cdef double* mat4_ptr = &eigenvecs_inv[0,0]

    cdef double *fmat1 = <double *> malloc(size * size *sizeof(double))
    cdef double *w = <double *> malloc(size * size *sizeof(double))

    alpha = 1.0
    beta = 0.0

    for i in range(size):
        for j in range(size):
            if i == j:
                w[i*size+j] = exp(time*eigenvals[i])
            else:
                w[i*size+j] = 0

    try:
        sp_la.dgemm(transa="n", transb="n", m=size_ptr, n = size_ptr,
                k=size_ptr, alpha=alpha_ptr, a=mat1_ptr, lda = size_ptr,
                b=w, ldb=size_ptr, beta=beta_ptr, c = fmat1, ldc = size_ptr)
        sp_la.dgemm(transa="n", transb="n", m=size_ptr, n = size_ptr,
                k=size_ptr, alpha=alpha_ptr, a=fmat1, lda = size_ptr,
                b=mat4_ptr, ldb=size_ptr, beta=beta_ptr, c = out, ldc = size_ptr)
        for i in range(size):
            for j in range(size):
                out[i*size+j] = log(out[i*size+j])+freqs[i]

    finally:
        free(fmat1)
        free(w)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
cpdef void double_dgemm(double[::1,:] mat1,double[::1,:] mat2,double[::1,:] mat3,double[::1,:] mat4,double[::1,:] mat5,
                int size, double time, double[:] vec1):
        cdef double alpha, beta
        cdef ssize_t i,j
        cdef int* size_ptr = &size
        cdef double* alpha_ptr = &alpha
        cdef double* beta_ptr = &beta
        cdef double* mat1_ptr = &mat1[0,0]
        cdef double* mat2_ptr = &mat2[0,0]
        cdef double* mat3_ptr = &mat3[0,0]
        cdef double* mat4_ptr = &mat4[0,0]
        cdef double* mat5_ptr = &mat5[0,0]

        alpha = 1.0
        beta = 0.0

        for i in range(size):
            mat2[i][i] = exp(time*vec1[i])

        sp_la.dgemm(transa="n", transb="n", m=size_ptr, n = size_ptr,
                k=size_ptr, alpha=alpha_ptr, a=mat1_ptr, lda = size_ptr,
                b=mat2_ptr, ldb=size_ptr, beta=beta_ptr, c = mat3_ptr, ldc = size_ptr)
        sp_la.dgemm(transa="n", transb="n", m=size_ptr, n = size_ptr,
                k=size_ptr, alpha=alpha_ptr, a=mat3_ptr, lda = size_ptr,
                b=mat4_ptr, ldb=size_ptr, beta=beta_ptr, c = mat5_ptr, ldc = size_ptr)




