# cython: language_level=3
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange
from libc.math cimport log, exp
DTYPE32 = np.int32
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE32_t


# get infinity
cdef extern from "math.h":
    double INFINITY

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int maxind(double[:] store, int size) nogil:
    cdef int i, rv = 0
    cdef double max_val = store[0]
    for i in range(1, size, 1):
        if store[i] > max_val:
            max_val = store[i]
            rv = i
    return rv


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef double spart_fast(double[:,:,:] cl_left,
                      double[:,:,:] cl_right,
                      double[:] storage,
                      double[:,:,:] new_cluster
                     ) nogil:
    """
    :param cl_left: cluster from the left daughter 
    :param cl_right: cluster from the right daughter
    :param storage: storage of alignment values
    :param new_cluster: cluster to be created for the current node
    :return: likelihood score
    
    The algorithm merges the most likely clusters in a greedy fashion.
    
    The maximum alignment value is looked for and then the corresponding action is performed, i.e. either merging or
    separating the clusters. Two clusters are only merged if both of them have not been merged or separated before.
    clusters which were separated before should not be merged in a step further up the tree. 

    """

    # get the number of clusters present at each of the daugher nodes
    cdef int lkl = <int> cl_left[36][0][0]
    cdef int lkr = <int> cl_right[36][0][0]
    # initialize temporary array to keep track of already merged clusters
    cdef int clust[36]

    # initialize necessary variables
    cdef int it, key_left, key_right, my_max, m, len_left, len_right, my_ind, size, lkr_
    cdef double v, value_left, value_right, score
    cdef ssize_t i, k ,l

    # initialize tracker
    for i in range(36):
        clust[i] = -1

    it = 0
    score = 0
    lkr_ = lkr * 2
    size = lkl*lkr_

    while 1:
        # get the index of the max value and the value
        my_max = maxind(storage, size)
        v = storage[my_max]

        if v == -INFINITY:
            # all values are checked
            break

        # value is found so it can be deleted
        storage[my_max] = -INFINITY

        # get indices

        key_left = my_max / lkr_
        key_right = (my_max - key_left * lkr_) / 2
        m = my_max-key_left*lkr_-key_right*2

        # get the necessary values
        value_right = cl_right[key_right][1][0]
        len_right = <int>cl_right[key_right][0][36]
        len_left = <int>cl_left[key_left][0][36]
        value_left = cl_left[key_right][1][0]

        if m == 0:
            # join

            if clust[lkl + key_right] == -1 and clust[key_left] == -1:
                # join can only happen if both clusters are not part of already existing clusters

                score += v
                # assign cluster the same value in tracker array
                clust[key_left] = it
                clust[lkl + key_right] = it

                # write the language into the cluster
                for i in range(len_left):
                    new_cluster[it][0][i] = cl_left[key_left][0][i]

                for i in range(len_right):
                    new_cluster[it][0][len_left+i] = cl_right[key_right][0][i]

                # write the value of the cluster
                new_cluster[it][1][0] = v

                # write the length of the cluster
                new_cluster[it][0][36] = len_left+len_right

                it += 1

        else:
            # no join

            if clust[lkl + key_right] == -1:
                # r cluster is not added

                score += value_right
                clust[lkl + key_right] = it

                # set value and length
                new_cluster[it][1][0] = value_right
                new_cluster[it][0][36] = len_right

                # get language identifiers
                for i in range(len_right):
                    new_cluster[it][0][i] = cl_right[key_right][0][i]

                it += 1

            if clust[key_left] == -1:
                # l cluster is not added

                score += value_left
                clust[key_left] = it

                # set value and length
                new_cluster[it][1][0] = value_left
                new_cluster[it][0][36] = len_left

                # get language identifiers
                for i in range(len_left):
                    new_cluster[it][0][i] = cl_left[key_left][0][i]

                it += 1

    new_cluster[36][0][0] = it


    return score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef void fpart_fast(double[:,:,:] cluster_left, double[:,:,:] cluster_right,
                     double[:] sc_d, double[:] storage,
                      double m_p, double n_p) nogil:

    cdef int lkl = <int> cluster_left[36][0][0]
    cdef int lkr = <int> cluster_right[36][0][0]

    cdef double v, v_l, v_r, summation
    cdef ssize_t i, j, k
    cdef int kl, kr, m, len1, len2, ind, lkr_

    lkr_ = lkr*2
    for i in range(0, lkl*lkr_, 2):


        kl = i / lkr_
        kr = (i - kl*lkr_)/2
        m = i-kl*lkr_-kr*2


        len1 = <int>cluster_left[kl][0][36]
        len2 = <int>cluster_right[kr][0][36]

        v_l = cluster_left[kl][1][0]
        v_r = cluster_right[kr][1][0]

        v = v_l + v_r

        v_l = 0 if len1 <= 1 else v_l
        v_r = 0 if len2 <= 1 else v_r

        len1 = 1 if len1 == 0 else len1
        len2 = 1 if len2 == 0 else len2


        summation = 0.0
        for k in range(len1):
            for j in range(len2):
                ind = <int>cluster_left[kl][0][k]*36+<int>cluster_right[kr][0][j]
                summation += sc_d[ind]

        storage[i] = summation-(len1+len2)
        storage[i + 1] = v



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.infer_types(False)
cpdef double concept_it(int n_conc, np.ndarray[np.float64_t, ndim=4, mode="c"] cluster_left,
                 np.ndarray[np.float64_t, ndim=4, mode="c"] cluster_right,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] score_d,
                 np.ndarray[np.float64_t, ndim=4, mode="c"] fin_clust,
                 double m_p, double n_p):

    cdef ssize_t i, j
    cdef double[:,:] storage_m=np.empty((n_conc,36*36*2), dtype=np.double, order="c")
    cdef double[:,:,:,:] cl_cleft_m = cluster_left
    cdef double[:,:,:,:] cl_right_m = cluster_right
    cdef double[:,:] score_d_m = score_d
    cdef double[:,:,:,:] fin_clust_m = fin_clust

    cdef double score = 0.0


    for i in prange(n_conc, nogil=True):
        for j in range(36*36*2):
            storage_m[i][j] = -INFINITY

        fpart_fast(cl_cleft_m[i], cl_right_m[i], score_d_m[i], storage_m[i], m_p, n_p)

        score += spart_fast(cl_cleft_m[i], cl_right_m[i], storage_m[i], fin_clust_m[i])

    return score