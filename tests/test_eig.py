from functools import partial

import numpy as np
import pytest

from numkl import eig, ev

a2 = np.array([[0, 1.0], [1, 0]])
a3 = np.array([[0, 1, 0], [1, 0.7, -2.8], [0, -2.8, 0]])
a2i = np.array([[0, 1j], [-1j, 1]])
a4 = np.random.rand(4, 4)
a4 = a4 + a4.T.conj()
a60 = np.random.rand(60, 60) + 1j * np.random.rand(60, 60)
a60i = a60 + a60.T.conj()
a101 = np.random.rand(101, 101)
a101 = a101 + a101.T.conj()
a101s = a101.astype(np.float32)

al = [a2, a2i, a3, a4, a60i, a101, a101s]


def get_al():
    global al
    bl = []
    for a in al:
        b = np.copy(a)
        bl.append(b)
    return bl


def allclose(a, b):
    if a.dtype in [np.float32, np.complex64]:
        return np.allclose(
            a, b, atol=1e-05
        )  ## result percision of float is lower than default atol
    elif a.dtype in [np.float64, np.complex128]:
        return np.allclose(a, b, atol=1e-09)
    else:
        raise TypeError()


def compare_eig(bl, method):
    for b in bl:
        en, vn = np.linalg.eigh(b)
        ex, vx = method(b)
        assert allclose(en, ex)
        for i in range(vn.shape[1]):
            assert allclose(np.linalg.norm(np.multiply(vn[:, i], vx[:, i]), ord=1), 1)


def test_eigh():
    bl = get_al()
    for b in bl:
        assert allclose(np.linalg.eigvalsh(b), eig.eigvalshx(b))
    bl = get_al()
    compare_eig(bl, eig.eighx)


@pytest.mark.parametrize("matrix_layout", [0, 1])
def test_syevd(matrix_layout):
    bl = get_al()
    for b in bl:
        en = np.linalg.eigvalsh(b)
        ex = ev.syevd(b, matrix_layout=matrix_layout, jobz="N")
        assert allclose(en, ex)
    bl = get_al()
    syevd0 = partial(ev.syevd, matrix_layout=matrix_layout, jobz="V")
    compare_eig(bl, syevd0)


"""
@pytest.mark.issue
@pytest.mark.xfail  # Intel MKL LAPACKE interface may has some unexpected return for row_major_order!! Fixed in psxe 2019.4
def test_syevd_issue():
    bl = get_al()
    for b in bl:
        en = np.linalg.eigvalsh(b)
        ex = ev.syevd(b, matrix_layout=1, jobz="N")
        assert np.allclose(en, ex)
    bl = get_al()
    syevd1 = partial(ev.syevd, matrix_layout=1, jobz="V")
    compare_eig(bl, syevd1)
"""

# a, int matrix_layout=1, jobz="V", range="A", vl=0, vu=0, il=0, iu=0, abstol=0
@pytest.mark.parametrize("matrix_layout", [0, 1])
def test_syevr(matrix_layout):
    bl = get_al()
    for b in bl:
        en = np.linalg.eigvalsh(b)
        ex, _ = ev.syevr(b, matrix_layout=matrix_layout, jobz="N", range="A")
        assert allclose(en, ex)
    bl = get_al()
    syevr0 = partial(ev.syevr, matrix_layout=matrix_layout, jobz="V", range="A")
    for b in bl:
        en, vn = np.linalg.eigh(b)
        ex, m, vx, _ = syevr0(b)
        assert allclose(en, ex)
        assert m == b.shape[0]
        for i in range(vn.shape[1]):
            assert allclose(np.linalg.norm(np.multiply(vn[:, i], vx[:, i]), ord=1), 1)


@pytest.mark.parametrize("matrix_layout", [0, 1])
def test_syevr_range(matrix_layout):
    bl = get_al()
    for b in bl:
        en = np.linalg.eigvalsh(b)
        ex, m = ev.syevr(
            b, matrix_layout=matrix_layout, jobz="N", range="I", il=2, iu=2
        )
        assert allclose(en[1], ex[0])
        assert m == 1
