ctypedef float complex lapack_complex_float
ctypedef double complex lapack_complex_double
ctypedef fused lapack_t:
    float
    double
    lapack_complex_float
    lapack_complex_double
ctypedef fused lapack_tr:
    float
    double
ctypedef long lapack_int

cdef extern from "mkl.h" nogil:


    # ctypedef float complex lapack_complex_float
    # ctypedef double complex lapack_complex_double ##Error:  Complex external typedefs not supported?

    lapack_int LAPACKE_dspevd (int matrix_layout, char jobz, char uplo, lapack_int n, double* ap, double* w, double* z, lapack_int ldz)


    lapack_int LAPACKE_ssyevd (int matrix_layout, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* w)
    lapack_int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w)

    lapack_int LAPACKE_ssyevr (int matrix_layout, char jobz, char range, char uplo, lapack_int n, float* a, lapack_int lda, float vl, float vu,
                               lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* w, float* z, lapack_int ldz, lapack_int* isuppz)
    lapack_int LAPACKE_dsyevr (int matrix_layout, char jobz, char range, char uplo, lapack_int n, double* a, lapack_int lda, double vl, double vu,
                               lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w, double* z, lapack_int ldz, lapack_int* isuppz)


    lapack_int LAPACKE_cheevd (int matrix_order, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w)
    lapack_int LAPACKE_zheevd (int matrix_order, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w)
    ##TODO: get rid of the complex type unmatch warning
    lapack_int LAPACKE_cheevr(int matrix_layout, char jobz, char range, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda,
                              float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz)

    lapack_int LAPACKE_zheevr(int matrix_layout, char jobz, char range, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda,
                              double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz)



    cdef int LAPACK_ROW_MAJOR=101
    cdef int LAPACK_COL_MAJOR=102

