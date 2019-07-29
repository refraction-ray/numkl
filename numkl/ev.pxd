from numkl.lapackev cimport *

cdef int _dspevd(double[::1] ap, double[::1] w, double[:,::1] z, int matrix_layout, char jobz, char uplo) nogil
cdef int _syevd(lapack_t[:,::1] a, lapack_tr[::1]e, int matrix_layout, char jobz, char uplo) nogil
cdef int _syevr(lapack_t[:,::1] a, lapack_tr[::1]e, lapack_t[:,::1] z, lapack_int[::1] isuppz, int matrix_layout, char jobz, char range, char uplo,
                lapack_int* m, lapack_tr vl, lapack_tr vu, lapack_int il, lapack_int iu, lapack_tr abstol) nogil
