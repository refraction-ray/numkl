# cython: language_level=3

import cython
import numpy as np
from numkl.lapackev cimport *
from numkl.helper import errmsg, _info_error


@cython.boundscheck(False) ##TODO: change spevd to fused type
cdef int _dspevd(double[::1] ap, double[::1] w, double[:,::1] z, int matrix_layout, char jobz, char uplo) nogil:
    cdef lapack_int n, ldz
    n = w.shape[0]
    ldz = w.shape[0]

    return LAPACKE_dspevd(matrix_layout, jobz, uplo, n, &ap[0], &w[0],&z[0][0], ldz)

@cython.boundscheck(False)
cdef int _syevd(lapack_t[:,::1] a, lapack_tr[::1] e, int matrix_layout, char jobz, char uplo) nogil:
    cdef lapack_int n
    cdef lapack_int lda
    n = a.shape[1]
    lda = a.shape[0]
    if lapack_t is double and lapack_tr is double:
        return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, &a[0][0], lda, &e[0])
    elif lapack_t is float and lapack_tr is float:
        return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, &a[0][0], lda, &e[0])
    elif lapack_t is lapack_complex_float and lapack_tr is float:
        return LAPACKE_cheevd(matrix_layout, jobz, uplo, n, &a[0][0], lda, &e[0])
    elif lapack_t is lapack_complex_double and lapack_tr is double:
        return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, &a[0][0], lda, &e[0])
    else:
        raise TypeError(errmsg["wrongtype"])

@cython.boundscheck(False)
cdef int _syevr(lapack_t[:,::1] a, lapack_tr[::1] e, lapack_t[:,::1] z, lapack_int[::1] isuppz, int matrix_layout, char jobz, char range, char uplo,
                lapack_int* m, lapack_tr vl, lapack_tr vu, lapack_int il, lapack_int iu, lapack_tr abstol) nogil:
    cdef lapack_int n, lda, ldz
    n = a.shape[1]
    lda = n
    ldz = 1 if jobz == ord("N") else n
    if lapack_t is double and lapack_tr is double:
        return LAPACKE_dsyevr(matrix_layout, jobz, range, uplo, n, &a[0][0], lda, vl, vu, il, iu, abstol, m, &e[0], &z[0][0], ldz, &isuppz[0])
    elif lapack_t is float and lapack_tr is float:
        return LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, &a[0][0], lda, vl, vu, il, iu, abstol, m, &e[0], &z[0][0], ldz, &isuppz[0])
    elif lapack_t is lapack_complex_float and lapack_tr is float:
        return LAPACKE_cheevr(matrix_layout, jobz, range, uplo, n, &a[0][0], lda, vl, vu, il, iu, abstol, m, &e[0], &z[0][0], ldz, &isuppz[0])
    elif lapack_t is lapack_complex_double and lapack_tr is double:
        return LAPACKE_zheevr(matrix_layout, jobz, range, uplo, n, &a[0][0], lda, vl, vu, il, iu, abstol, m, &e[0], &z[0][0], ldz, &isuppz[0])
    else:
        raise TypeError(errmsg["wrongtype"])

@cython.boundscheck(False)
def syevd(a, int matrix_layout=0, jobz="V"):  ## WARNING: using layout default to 0, col major until the issue is resolved
    """
    Lower level python wrapper on ?syevd and ?heevd, the routine is auto chosen in runtime, depending on the dtype of matrix a.

    :param a: np.array, matrix
    :param matrix_layout: integer, default 0, odd number for row major order (C) while even number for col major order (F).
                        warn: currently, row major order may have unexpected results on eigenvectors.
    :param jobz: char, default "V", V for eigenpairs while N for only eigenvalues
    :return: np.array for eigenvalues when jobz="N"
            or tuple of np.array for eigenpairs when job="V"
    """
    jobz = ord(jobz) #REF: https://stackoverflow.com/questions/28002214/cython-typeerror-an-integer-is-required
    if a.dtype in [np.float32, np.float64]:
        e = np.zeros(a.shape[0], dtype=a.dtype)
    elif a.dtype == np.complex64:
        e = np.zeros(a.shape[0], dtype=np.float32)
    elif a.dtype == np.complex128:
        e = np.zeros(a.shape[0], dtype=np.float64)
    if matrix_layout%2 == 1:
        np.conj(a,a) ## hopefully a inplace transformation, it is needed since we are now relying on col major order.
        matrix_layout = LAPACK_ROW_MAJOR ## WARNING: there seems to be some bugs in mkl lapacke interface with row major layout, the returned eigenvectors is somewhat weird
        ## actually the error is due to only U part of the eigenvector matrix is returned! the lower part is kept as it is.
        ## issue tracker: https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/814791
        ## parallel studio 2019 Update 4 fixing this issue in Lapacke
    else:
        matrix_layout = LAPACK_COL_MAJOR
    ## MKL matrix layout doc: https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/cinterface.htm#matrixlayout
    ## lda is the dimension of major
    uplo = ord("U")
    # print(a.dtype)
    if a.dtype == np.float64:
        info = _syevd[double,double](a,e,matrix_layout,jobz, uplo)
    elif a.dtype == np.float32:
        info = _syevd[float,float](a,e,matrix_layout,jobz, uplo)
    elif a.dtype == np.complex64:
        info = _syevd[lapack_complex_float,float](a,e,matrix_layout,jobz, uplo)
    elif a.dtype == np.complex128:
        info = _syevd[lapack_complex_double,double](a,e,matrix_layout,jobz, uplo)
    else:
        raise TypeError(errmsg["wrongtype"])

    if info == 0:
        if (jobz == ord("V")):
            if matrix_layout%2 == 1:
                return e, np.array(a)
            else: # col major
                return e, np.array(a).T
        else:
            return e
    else:
        _info_error(info)

@cython.boundscheck(False)
def syevr(a, int matrix_layout=0, jobz="V", range="A", vl=0, vu=0, il=0, iu=0, abstol=0):
    """
    Lower level python wrapper on ?syevr, the routine is auto chosen in runtime, depending on the dtype of matrix a.
    Note syevr in general has low accuracy than syevd. Although stevr is claimed to be the last eigensolver by lapack.

    :param a: np.array, input matrix
    :param matrix_layout: integer, default 1, odd number for row major order (C) while even number for col major order (F).
    :param jobz: char, default "V", V for eigenpairs while N for only eigenvalues
    :param range: char, default "A". 'A': all eigenvalues will be found.
                        'V': all eigenvalues in the half-open interval (VL,VU] will be found.
                        'I': the IL-th through IU-th eigenvalues will be found. (staring from 1, both ends are closed)
    :param vl: float, see range
    :param vu: float see range
    :param il: int, see range
    :param iu: int, see range
    :param abstol: double, tolerance
    :return:
    """
    n = a.shape[0]
    if a.dtype in [np.float32, np.complex64]:
        e = np.zeros(n, dtype=np.float32)
    elif a.dtype in [np.float64, np.complex128]:
        e = np.zeros(n, dtype=np.float64)
    else:
        raise TypeError(errmsg["wrongtype"])
    z = np.zeros((n,n), dtype=a.dtype) if jobz == "V" else np.zeros((1,1), dtype=a.dtype)
    cdef long [::1] isuppzl = np.zeros(2*n, dtype=np.long)
    cdef lapack_int [::1] isuppz = <lapack_int[:2*n:1]> &isuppzl[0] ##TODO: hopefully some elegant way than this workaround
    if matrix_layout%2 == 1:
        np.conj(a,a)
        matrix_layout = LAPACK_ROW_MAJOR
    else:
        matrix_layout = LAPACK_COL_MAJOR
    uplo = ord("U")
    jobz = ord(jobz)
    range = ord(range)

    ##abstol=n * np.finfo(a.dtype).eps*np.linalg.norm(a, ord=1)/10000000 ## ? https://software.intel.com/en-us/node/521126

    cdef lapack_int m=0 ## somehow one need an initial value for m, otherwise, the returned m is wrong, cannot repruduce that by minimal code without mkl
    if a.dtype == np.float64:
        info = _syevr[double, double](a,e,z,isuppz,matrix_layout,jobz,range, uplo, &m, vl, vu, il, iu, abstol)
    elif a.dtype == np.float32:
        info = _syevr[float, float](a,e,z,isuppz,matrix_layout,jobz,range, uplo, &m, vl, vu, il, iu, abstol)
    elif a.dtype == np.complex64:
        info = _syevr[lapack_complex_float, float](a,e,z,isuppz,matrix_layout,jobz,range, uplo, &m, vl, vu, il, iu, abstol)
    elif a.dtype == np.complex128:
        info = _syevr[lapack_complex_double, double](a,e,z,isuppz,matrix_layout,jobz,range, uplo, &m, vl, vu, il, iu, abstol)
    else:
        raise TypeError(errmsg["wrongtype"])

    if info == 0:
        if jobz == ord("N"):
            return e, m
        else:
            if matrix_layout%2 == 1:
                return e, m, z, np.asarray(isuppz)
            else: # col major
                return e, m, z.T.conj(), np.asarray(isuppz)
        ##TODO: more fine grain on what to be returned and how to process the out from syevr
        ##TODO: still no idea what isuppz is
    else:
        _info_error(info)

@cython.boundscheck(False)
def dspevd(double[::1] ap, int matrix_layout=0, jobz="V"):
    n = int(-0.5+np.sqrt(1+8*ap.shape[0])/2)
    w = np.zeros(n)
    if jobz == "V":
        z = np.zeros((n,n))
    else:
        z = np.zeros((1,1))
    ## TODO: clarify the row or col major and its relation to packed storage
    matrix_layout = LAPACK_COL_MAJOR
    uplo = ord("U")
    jobz = ord(jobz)
    info = _dspevd(ap,w,z,matrix_layout,jobz,uplo)
    if info == 0:
        if (jobz == ord("V")):
            return w, np.array(z).T
        else:
            return w
    elif info <= 0:
        raise ValueError(errmsg['illegalinput']+str(-info))
    else:
        raise ValueError(errmsg["nonzeroinfo"])