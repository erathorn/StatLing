cimport numpy as np

cpdef tuple sample_uniform_in_list(double d_max, list a, double lower=*, double upper=*)
cpdef tuple sample_uniform_single(double d_max, double r, double lower, double upper)

cpdef tuple sample_frequency(double d_max, np.ndarray[np.float64_t, ndim=1] f, int index)
cdef double expontial_pdf(double lambd, double x)

cpdef tuple sample_exponential(double d_max, double lambd, double time)

cpdef void set_srand_seed(int seed)

cdef void sample_uniform(double x, double d_max, double[:] output, double lower = *, double upper=*)


