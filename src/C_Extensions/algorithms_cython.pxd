cimport numpy as np


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
                                                            )
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
                                                                double mu
                                                                )

cpdef np.ndarray[np.float64_t, ndim=1] calculaute_odds_scores(tuple loseq,
                              np.ndarray[np.float64_t, ndim=2] em,
                              np.ndarray[np.float64_t, ndim=1] gx, np.ndarray[np.float64_t, ndim=1] gy,
                              np.ndarray[np.float64_t, ndim=1] trans, int alphabet_size)




cpdef np.ndarray[np.float64_t, ndim=1] calculaute_odds_scores_tkf(tuple loseq,
                              np.ndarray[np.float64_t, ndim=2] em,
                              np.ndarray[np.float64_t, ndim=1] gx, np.ndarray[np.float64_t, ndim=1] gy,
                              np.ndarray[np.float64_t, ndim=1] trans,
                               double lambd, double mu, double t)

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
                                                            np.ndarray[np.float64_t, ndim=3, mode="c"] likmat)

cdef void internal(char* words_left, char* words_right,
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
              ) nogil


cdef void calc_trans(double a, double r, double t, double* out) nogil
