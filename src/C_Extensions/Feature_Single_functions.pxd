cimport numpy as np

# cpdef tuple diagonalize_q_matrix(np.ndarray[np.float64_t, ndim=1, mode="c"] frequencies,
#                                  np.ndarray[np.float64_t, ndim=2, mode="fortran"] d_mat,
#                                  np.ndarray[np.float64_t, ndim=2, mode="c"] b_mat, int size,
#                                  np.ndarray[np.float64_t, ndim=1, mode="c"] evo_paras,
#                                  np.ndarray[np.int8_t, ndim=1, mode="c"] iu,
#                                  np.ndarray[np.int8_t, ndim=1, mode="c"] il)
# cdef tuple calc_eigvecs(double[::1,:] d_b_d, int size, double* alpha_ptr, double* d_ptr, double* beta_ptr)
#
# cpdef set_exponential(double time, int necessary, np.ndarray[np.float64_t, ndim=2, mode="fortran"] w,
#                       np.ndarray[np.float64_t, ndim=2, mode="fortran"] q_eigenvectors,
#                       np.ndarray[np.float64_t, ndim=1, mode="c"] q_eigenvalues,
#                       np.ndarray[np.float64_t, ndim=2, mode="fortran"] fortran_matrix_1,
#                       np.ndarray[np.float64_t, ndim=2, mode="fortran"] fortran_matrix_2,
#                       np.ndarray[np.float64_t, ndim=2, mode="c"]pt_log,
#                       bmat, dmat, int size, frequencies, iu,il, evo_paras)
cpdef void create_q(double[:,:] b_mat_view, double[:,:] d_mat_view,int size, double[:] freq_view, double[:] evo_view,long[:] iu, long[:] il)

cpdef void double_dgemm(double[::1,:] mat1,double[::1,:] mat2,double[::1,:] mat3,double[::1,:] mat4,double[::1,:] mat5,
                        int size, double time, double[:] vec1)



cdef void dgemm_thread_2D(double[::1,:] eigenvecs, double[::1,:] eigenvecs_inv,
                       int size, double time, double[:] eigenvals, double* out, double[:] freqs) nogil