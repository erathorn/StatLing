cdef class Transition_TKF91:
    cpdef double _l, _r, _mu, _time, _T00, _T01, _T02, _T0E, _T10, _T11, _T12, _T1E, _T20, _T21, _T22, _T2E, _TS0, _TS1, _TS2, _TSE, dmax, old
    cpdef list _sampling_functions
    cpdef object _params, _cl, iterator
    cpdef int count

    cdef void calculate(self)
    cpdef public void set_model(self, double time)
    cpdef tuple sample_l(self)
    cpdef tuple sample_mu(self)

    cpdef public void revert_l(self, double old)
    cpdef public void revert_mu(self, double old)

    cpdef public void to_file(self, int header, str file_name)
