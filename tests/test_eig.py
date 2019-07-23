from numkl import eig
from numkl import ev
import numpy as np
from functools import partial
import pytest

a2 = np.array([[0, 1.0], [1, 0]])
a3 = np.array([[0, 1, 0], [1, 0.7, -2.8], [0, -2.8, 0]])
a2i = np.array([[0, 1j], [-1j, 1]])
a4 = np.random.rand(4, 4)
a4 = a4 + a4.T.conj()
a101 = np.random.rand(101, 101)
a101 = a101 + a101.T.conj()

al = [a2, a3, a4, a101]


def get_al():
    global al
    bl = []
    for a in al:
        b = np.copy(a)
        bl.append(b)
    return bl


def compare_eig(bl, method):
    for b in bl:
        en, vn = np.linalg.eigh(b)
        ex, vx = method(b)
        assert np.allclose(en, ex)
        for i in range(vn.shape[1]):
            assert np.allclose(
                np.linalg.norm(np.multiply(vn[:, i], vx[:, i]), ord=1), 1
            )


def test_eigh():
    bl = get_al()
    for b in bl:
        assert np.allclose(np.linalg.eigvalsh(b), eig.eigvalshx(b))
    bl = get_al()
    compare_eig(bl, eig.eighx)


@pytest.mark.parametrize("matrix_layout", [0, 1])
def test_syevd(matrix_layout):
    bl = get_al()
    for b in bl:
        en = np.linalg.eigvalsh(b)
        ex = ev.syevd(b, matrix_layout=matrix_layout, jobz="N")
        assert np.allclose(en, ex)
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
        np.allclose(en, ex)
    bl = get_al()
    syevr0 = partial(ev.syevr, matrix_layout=matrix_layout, jobz="V", range="A")
    for b in bl:
        en, vn = np.linalg.eigh(b)
        ex, _, vx, _ = syevr0(b)
        assert np.allclose(en, ex)
        for i in range(vn.shape[1]):
            assert np.allclose(
                np.linalg.norm(np.multiply(vn[:, i], vx[:, i]), ord=1), 1
            )
