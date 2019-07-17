cimport numpy as np


#cpdef void enlarge(const double[:] vec1, int[:] shape, double[:,:] output_view)
#cpdef void vec_to_mat_add(const double[:] vec, const double[:,:] mat, double[:,:] output_view)
cdef double dot(const double[:] vec, const double[:] vec2, const int m)
#cpdef void vec_to_mat_mult(const double[:] vec, const double[:,:] mat, double[:,:] output_view)
#cdef void log_and_trans_mat(const double[::1,:] mat, double[:,::1] out_view, int size)
#cdef void log_mat(double[:,:] mat, int m)
#cpdef void sum_up_vecs(const double[:,:] in_vec_stack, double [:] out_view)
cpdef void array_creation(const double[:] vec1, const double[:] vec2, const long[:] ind1, const long[:] ind2, double[:,:] output_view)

cpdef void inner_loop_felsenstein(double[:,:] state_probs_left,
                                  double[:,:] state_probs_right,
                                  int size,
                                  double[:,:] matrix_left,
                                  double[:,:] matrix_right,
                                  double[:,:] result)