
import numpy as np
cimport numpy as np

import cython
from cython.parallel import prange
from cython cimport view
import src.C_Extensions.Feature_Single_functions as c_feat
cimport src.C_Extensions.Feature_Single_functions as c_feat
# get log and exp from c math
from libc.math cimport log, exp
from libc.stdlib cimport malloc, free



# get c implementation of the forward algorithm
cdef extern from "fw_log.h":
    double forward_log(char s1[], char s2[], long m, long n, double em_p[], double gx[], double gy[], double trans[], int alphabet_size, int dimension, double correction_parameter) nogil
    double random_score_c(char s1[], char s2[], long m, long n, double gx[]) nogil
    double random_score_c_single(char s1[], long m, double gx[]) nogil
    double viterbi_log(char s1[], char s2[], long m, long n, double em[], double gx[], double gy[], double trans[], int alphabet_size, char alignment[]) nogil
    double random_score_c_single_tkf(char s1[], long m, double gx[], double mu, double l) nogil
    double forward_log_2D_pointer(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size) nogil



# get infinity
cdef extern from "math.h":
    double INFINITY

# get numpy types
DTYPE = np.float64
DTYPE64 = np.int64
DTYPE32 = np.int32
DTYPE8 = np.int8
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE64_t
ctypedef np.int32_t DTYPE32_t
ctypedef np.int8_t DTYPE8_t


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cdef void calc_trans(double a, double r, double t, double* out) nogil:

    cdef double indel, indelprime, two_rt, temp_exp, v1, v2, v3,v4, e1, e10,e11,e12,t00,t10,t01,t11,t12

    two_rt = 2.0*r*t
    temp_exp = exp(-two_rt)
    indel = 1.0 - temp_exp
    indelprime = 1-(1/two_rt)*indel

    v1 = log(1.0-indel)
    v2 = log(0.5*indel)
    v3 = log((1.0-a)*(1-indel))
    v4 = log((1.0-a)*indel)


    out[0] = v1 #t00
    out[1] = v2 #t01
    out[2] = v2#t02
    out[3] = v1#t0E
    out[4] = v3#t10
    out[5] = log(a)#t11
    out[6] = v4#t12
    out[7] = v3#t1E
    out[8] = v3#t20
    out[9] = v4#t21
    out[10] =log(a)#t22
    out[11] =v3#t2E
    out[12] =v1#tS0
    out[13] =v2#tS1
    out[14] =v2#tS2
    out[15] =0.0#tSE


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cdef void internal(
              char* words_left,
              char* words_right,
              double a, double r, double t,
              double[::1,:] q_eigen_vec,
              double[::1,:] q_eigen_vec_inv,
              double[:] q_eigen_vals,
              double[:] freq_log,
              double non_existing,
              int alphabet_size,
              int n_words,
              double* out,
              int running_dim
              ) nogil:
#

    cdef ssize_t i
    cdef double *t_out = <double *> malloc(16*sizeof(double))
    cdef double *em_mat = <double *> malloc(alphabet_size* alphabet_size *sizeof(double))
    cdef int m,n, inc_off
    cdef int offset = running_dim*35*n_words
    cdef double P, rs, t1
    cdef double[:] gy_v = freq_log
    cdef double[:] gx_v = freq_log
    cdef double* gx_v_p
    cdef double* gy_v_p
    gx_v_p = &gx_v[0]
    gy_v_p = &gy_v[0]

    rs = 0
    try:
        calc_trans(a, r, t, t_out)
        c_feat.dgemm_thread_2D(q_eigen_vec, q_eigen_vec_inv, alphabet_size, t, q_eigen_vals, em_mat, freq_log)

        for i in range(n_words):

            inc_off = offset+i*35
            if words_left[inc_off+0] == -1 or words_right[inc_off+0] == -1:
                out[running_dim*n_words*2+i*2] = non_existing
                out[running_dim*n_words*2+i*2+1] = non_existing
            elif words_left[inc_off+33] != words_right[inc_off+33]:
                out[running_dim*n_words*2+i*2] = non_existing
                out[running_dim*n_words*2+i*2+1] = non_existing
            else:
                m = words_left[inc_off+34]
                n = words_right[inc_off+34]
                P = forward_log_2D_pointer(&words_left[inc_off+0],
                                       &words_right[inc_off+0],
                                       m,n,
                                       em_mat, gx_v_p, gy_v_p, t_out, alphabet_size)
                rs =random_score_c(&words_left[inc_off+0], &words_right[inc_off+0], m, n, gx_v_p)

                out[running_dim*n_words*2+i*2] = P
                out[running_dim*n_words*2+i*2+1] = rs

    finally:
        free(em_mat)
        free(t_out)

#
#
#
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef np.ndarray[np.float64_t, ndim=2] new_calc(np.ndarray[np.int8_t, ndim=3, mode="c"] accumulated_words_left,
                                                np.ndarray[np.int8_t, ndim=3, mode="c"] accumulated_words_right,
                                                double a,
                                                double r,
                                                np.ndarray[np.float64_t, ndim=1, mode="c"] time,
                                                np.ndarray[np.float64_t, ndim=2, mode="fortran"] q_eigen_vec,
                                                np.ndarray[np.float64_t, ndim=2, mode="fortran"] q_eigen_vec_inv,
                                                np.ndarray[np.float64_t, ndim=1, mode="fortran"] q_eigen_vals,
                                                np.ndarray[np.float64_t, ndim=1, mode="c"] freq_log,
                                                double non_existing,
                                                int alphabet_size,
                                                np.ndarray[np.float64_t, ndim=3, mode="c"] likmat
                                                ):

#
    cdef ssize_t i
    cdef int n_words =accumulated_words_left.shape[1]
    cdef int num_lang = time.shape[0]

    cdef double[:] gx_v = freq_log
    cdef char[:,:,:] words_v_left = accumulated_words_left
    cdef char[:,:,:] words_v_right = accumulated_words_right
    cdef double[::1,:] q_view_eig = q_eigen_vec
    cdef double[::1,:] q_view_eig_inv = q_eigen_vec_inv
    cdef double[:] q_view_eig_vals = q_eigen_vals

    # num_lang: number of language pairs
    # n_words: number of words/concepts
    cdef double[:,:,::1] lik_mat = likmat

    # here begins the parallel part
    for i in prange(num_lang, nogil=True):

        internal(&words_v_left[0,0,0], &words_v_right[0,0,0],
                 a,r,time[i],
                 q_view_eig, q_view_eig_inv, q_view_eig_vals,
                 gx_v,
                 non_existing, alphabet_size,n_words, &lik_mat[0,0,0],
                 i
                 )

    return np.array(lik_mat)

#!python
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef np.ndarray[np.float64_t, ndim=2] calculate_scores_big(np.ndarray[np.int8_t, ndim=3, mode="c"] accumulated_words_left,
                                                            np.ndarray[np.int8_t, ndim=3, mode="c"] accumulated_words_right,
                                                            np.ndarray[np.float64_t, ndim=3, mode="c"] big_mat,
                                                            np.ndarray[np.float64_t, ndim=1, mode="c"] big_gap,
                                                            np.ndarray[np.float64_t, ndim=2, mode="c"] big_trans,
                                                            double non_existing,
                                                            int alphabet_size,
                                                            double mu,
                                                            double l,
                                                            double correction_parameter,
                                                            np.ndarray[np.float64_t, ndim=1, mode="c"] time_list
                                                            ):
    """
    This function wraps the calculation of alignment scores. The calculation of the individual alignment scores between
    two words is done in C. To speed up the calculation the computation is done in parallel using multiple threads. For
    the parallel part GIL is released.  
    
    :param accumulated_words_left: collection of integer representation of words 
    :param accumulated_words_right: collection integer representation of words
    :param big_mat: emission matrix collection
    :param big_gap: gap score collection
    :param big_trans: transition score collection
    :param non_existing: how to treat non existing word pairs
    :param alphabet_size: alpahabet size
    :param n_classes: number of cognate classes per concept
    :param sample: are cognate classes sampled, if 1 they are, if 0 they are not
    :return: array of word pair scores
    """
    cdef ssize_t i, j, c
    cdef int n_words =accumulated_words_left.shape[1]
    cdef int num_lang = big_mat.shape[0]
    cdef int m, n, cognate_class_identifier, c_classes_intern

    cdef double[:,:,:] em_v = big_mat
    cdef double[:] gy_v = big_gap
    cdef double[:] gx_v = big_gap
    cdef double[:,:] trans_v = big_trans
    cdef char[:,:,:] words_v_left = accumulated_words_left
    cdef char[:,:,:] words_v_right = accumulated_words_right
    cdef double *em_v_p
    cdef double *gx_v_p
    cdef double *gy_v_p
    cdef double *trans_v_p

    cdef double p, rs, p1


    # num_lang: number of language pairs
    # n_words: number of words/concepts
    cdef double[:,:] lik_mat = np.empty((num_lang,n_words), dtype=np.double, order="c")
    em_v_p = &em_v[0,0,0]
    trans_v_p = &trans_v[0,0]
    gx_v_p = &gx_v[0]
    gy_v_p = &gy_v[0]
    # here begins the parallel part
    for i in prange(num_lang, nogil=True):
        for j in range(n_words):
            if words_v_left[i][j][0] == -1 or words_v_right[i][j][0] == -1:
                lik_mat[i][j] = non_existing

            elif words_v_left[i][j][33] != words_v_right[i][j][33]:
                lik_mat[i][j] = non_existing

            else:
                m = words_v_left[i,j,34]
                n = words_v_right[i,j,34]

                # get pointer


                # calculate the alignment score
                p = forward_log(&words_v_left[i,j,0], &words_v_right[i,j,0], m, n, em_v_p, gx_v_p, gy_v_p, trans_v_p, alphabet_size, i,
                                correction_parameter)
                #rs = random_score_c_single_tkf(&words_v_left[i,j,0], m, gx_v_p, mu, l) + random_score_c_single_tkf(&words_v_right[i,j,0], n, gx_v_p, mu, l)

                #p1 = p - rs

                lik_mat[i][j] = p#+time_list[i]#1 - log((exp(p1)+1))#[0] = p + log(1-exp(rs))
                #lik_mat[i][j][1] = rs# + log(1-exp(p))

    return np.array(lik_mat)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef np.ndarray[np.float64_t, ndim=2] calculate_scores_big_v2( np.ndarray[np.int8_t, ndim=3, mode="c"] all_words,
                                                                int n_words,
                                                                np.ndarray[np.int32_t, ndim=2, mode="c"] mapping,
                                                                int n_lang,
                                                                np.ndarray[np.float64_t, ndim=3, mode="c"] big_mat,
                                                                np.ndarray[np.float64_t, ndim=1, mode="c"] big_gap,
                                                                np.ndarray[np.float64_t, ndim=2, mode="c"] big_trans,
                                                                double non_existing,
                                                                int alphabet_size,
                                                                double l,
                                                                double mu,

                                                                ):
    """
    This function wraps the calculation of alignment scores. The calculation of the individual alignment scores between
    two words is done in C. To speed up the calculation the computation is done in parallel using multiple threads. For
    the parallel part GIL is released.  
    
    
    :param big_mat: emission matrix collection
    :param big_gap: gap score collection
    :param big_trans: transition score collection
    :param non_existing: how to treat non existing word pairs
    :param alphabet_size: alpahabet size
    :param n_classes: number of cognate classes per concept
    :param sample: are cognate classes sampled, if 1 they are, if 0 they are not
    :return: array of word pair scores
    """
    cdef ssize_t i, j, k

    cdef int m, n, ind

    cdef double[:,:,:] em_v = big_mat
    cdef double[:] gy_v = big_gap
    cdef double[:] gx_v = big_gap
    cdef double[:,:] trans_v = big_trans
    cdef char[:,:,:] words = all_words
    cdef int[:,:] my_map = mapping

    cdef double *em_v_p
    cdef double *gx_v_p
    cdef double *gy_v_p
    cdef double *trans_v_p

    cdef double p, p1

    # num_lang: number of language pairs
    # n_words: number of words/concepts
    cdef double[:,:,:] lik_mat = np.empty((n_lang,n_lang,n_words), dtype=np.double, order="c")

    gx_v_p = &gx_v[0]
    gy_v_p = &gy_v[0]
    em_v_p = &em_v[0,0,0]
    trans_v_p = &trans_v[0,0]
    # here begins the parallel part
    for i in prange(n_lang, nogil=True):
        for j in range(n_lang):
            if i>=j:
                for k in range(n_words):
                    if words[i][k][0] == -1 or words[j][k][0] == -1:
                        lik_mat[i][j][k] = non_existing
                        lik_mat[i][j][k] = non_existing

                    else:
                        m = words[i,k,34]
                        n = words[j,k,34]
                        if i != j:
                            ind = my_map[i,j]
                            # get pointer


                            # calculate the alignment score
                            p1 = forward_log(&words[i,k,0], &words[j,k,0], m, n, em_v_p, gx_v_p, gy_v_p, trans_v_p,
                                                   alphabet_size, ind, log(0.08)) - (random_score_c_single(&words[i,k,0], m, gx_v_p) +
                                                                     random_score_c_single(&words[j,k,0], n, gx_v_p))

                            p = p1 - log((exp(p1)+1))

                        else:

                            #p = random_score_c_single(&words[i,k,0], m, gx_v_p)
                            p = random_score_c_single(&words[i,k,0], m, gx_v_p)
                            #em_v_z_p = &em_v_z[0,0]
                        #    p = 0.0#forward_log(&words[i,k,0], &words[i,k,0], m, m, em_v_z_p, gx_v_z_p, gy_v_z_p_v_z, trans_v_p, alphabet_size)

                        lik_mat[i][j][k] = p#[0] = p + log(1-exp(rs))
                        lik_mat[j][i][k] = p# + log(1-exp(p))



    return np.array(lik_mat)



@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] calculaute_odds_scores(tuple loseq,
                              np.ndarray[np.float64_t, ndim=2] em,
                              np.ndarray[np.float64_t, ndim=1] gx, np.ndarray[np.float64_t, ndim=1] gy,
                              np.ndarray[np.float64_t, ndim=1] trans,
                              int alphabet_size):
    cdef int m, n, l
    cdef ssize_t i, j, k
    cdef double p, rs
    l = len(loseq[0])

    cdef double[:] lik_list_unrel = np.zeros(l, dtype=np.double)


    cdef char[:,:] slist = loseq[0]
    cdef char[:,:] slist2 = loseq[1]

    cdef double[:,:] em_v = em
    cdef double[:] gy_v = gy
    cdef double[:] gx_v = gx
    cdef double[:] trans_v = trans

    cdef double *em_v_p
    cdef double *gx_v_p
    cdef double *gy_v_p
    cdef double *trans_v_p

    cdef char[:,:,:] alignment_store = np.empty((l, 2, 34), dtype=np.int8)

    em_v_p = &em_v[0,0]
    trans_v_p = &trans_v[0]
    gx_v_p = &gx_v[0]
    gy_v_p = &gy_v[0]



    for i in prange(l, nogil = True):
        for j in range(2):
            for k in range(34):
                alignment_store[i,j,k] = -1

        if slist[i,0] == -1 or slist2[i,0] == -1:
            lik_list_unrel[i] = -INFINITY

        else:
            m = slist[i,34]
            n = slist2[i,34]

            p = viterbi_log(&slist[i,0], &slist2[i,0], m, n, em_v_p, gx_v_p, gy_v_p, trans_v_p, alphabet_size, &alignment_store[i,0,0])
            rs = random_score_c(&slist[i,0], &slist2[i,0], m, n, gx_v_p)

            lik_list_unrel[i] = p-rs

    return np.array(lik_list_unrel)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] calculaute_odds_scores_tkf(tuple loseq,
                              np.ndarray[np.float64_t, ndim=2] em,
                              np.ndarray[np.float64_t, ndim=1] gx, np.ndarray[np.float64_t, ndim=1] gy,
                              np.ndarray[np.float64_t, ndim=1] trans, double lambd, double mu, double t):

    cdef int m, n, l, alphabet_size
    cdef ssize_t i
    cdef double p, rs
    l = len(loseq[0])
    alphabet_size=len(gx)

    cdef double[:] lik_list_unrel = np.zeros(l, dtype=np.double)

    cdef char[:,:] slist = loseq[0]
    cdef char[:,:] slist2 = loseq[1]

    cdef double[:,:] em_v = em
    cdef double[:] gy_v = gy
    cdef double[:] gx_v = gx
    cdef double[:] trans_v = trans

    cdef double *em_v_p
    cdef double *gx_v_p
    cdef double *gy_v_p
    cdef double *trans_v_p


    em_v_p = &em_v[0,0]
    trans_v_p = &trans_v[0]
    gx_v_p = &gx_v[0]
    gy_v_p = &gy_v[0]
    cdef char [:,:] alignment = np.zeros((2,34), dtype=np.char)
    for i in range(l):

        if slist[i,0] == -1 or slist2[i,0] == -1:
            lik_list_unrel[i] = -INFINITY

        else:
            m = slist[i,34]
            n = slist2[i,34]


            p = viterbi_log(&slist[i,0], &slist2[i,0], m, n, em_v_p, gx_v_p, gy_v_p, trans_v_p, alphabet_size, &alignment[0,0])
            rs = random_score_c_single_tkf(&slist[i,0], m, gx_v_p, mu, lambd) + \
                 random_score_c_single_tkf(&slist2[i,0], n, gx_v_p, mu, lambd)


            lik_list_unrel[i] = p-rs

    return np.array(lik_list_unrel)





