from numkl.ev import syevd


def eighx(a):  # TODO: support broadcast?
    """
    Calculate eigenvalues and eigenvectors for Hermitian or real symmetric matrix.

    Notes: Using Lapack ?syevd and ?heevd routine as backend. The total memory space it required is in order N^2 (a included).
    The input matrix would be destroy in the routine, so copy the matrix by hand if you still need it after the calculation.

    :param a: np.array, matrix, broadcast is not supported now
    :return: e,v: tuple of np.array, one for eigenvalues and one for eigenvectors
    """
    return syevd(a, jobz="V")


def eigvalshx(a):
    """
    Calculate only eigenvalues for Hermitian or real symmetric matrix.

    Notes: Using Lapack ?syevd and ?heevd routine as backend. The total memory space it required is in order 3N^2 (a included).
    The input matrix would be destroy in the routine, so copy the matrix by hand if you still need it after the calculation.

    :param a: np.array, matrix, broadcast is not supported now
    :return: np.array, eigenvalues in order
    """
    return syevd(a, jobz="N")
