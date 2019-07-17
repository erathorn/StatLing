cimport numpy as np


cpdef double spart_fast(double[:,:,:] cluster_left,
                      double[:,:,:] cluster_right,
                      double[:] storage,
                      double[:,:,:] nst_2
                       ) nogil
cpdef void fpart_fast(double[:,:,:] cluster_left, double[:,:,:] cluster_right,
                    double[:] sc_d, double[:] storage,double m_p, double n_p) nogil

cpdef double concept_it(int n_conc, np.ndarray[np.float64_t, ndim=4, mode="c"] cluster_left,
                 np.ndarray[np.float64_t, ndim=4, mode="c"] cluster_right,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] score_d,
                 np.ndarray[np.float64_t, ndim=4, mode="c"] fin_clust,
                 double m_p, double n_p)
