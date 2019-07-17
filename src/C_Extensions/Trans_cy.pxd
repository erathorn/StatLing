
cdef class Transition_km03:
    cpdef double _a, _r,_mu, _l, _time, _T00, _T01, _T02, _T00_log, _T01_log, _T02_log, _T10, _T11, _T12, _T10_log, _T11_log, _T12_log, _indel, _indel_prime, dmax_a,dmax_r, old
    cpdef list _sampling_functions
    cpdef object _params, _cl, iterator
    cpdef int count
    cpdef public tuple sample_wrap(self, str feat)
    cdef void calculate(self)
    cpdef public void set_model(self, double time)
    cpdef tuple sample_a(self)
    cpdef tuple sample_r(self)
    cdef double E00(self)
    cdef double E11(self)
    cdef double E10(self)
    cdef double E12(self)
    cdef double E12_log(self)
    cdef double E1(self)
    cdef double E1_log(self)
    cpdef public void revert_a(self, double old)
    cpdef public void revert_r(self, double old)
    cpdef public void to_file(self, int header, str file_name)