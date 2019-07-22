from numkl import eig
import numpy as np

a2 = np.array([[0, 1.0], [1, 0]])
a3 = np.array([[0, 1, 0], [1, 0.7, -2.8], [0, -2.8, 0]])
a2i = np.array([[0, 1j], [-1j, 1]])
a4 = np.random.rand(4, 4)
a4 = a4 + a4.T.conj()

al = [a2, a3, a2i, a4]


def test_eigh():
    bl = []
    for a in al:
        b = np.copy(a)
        bl.append(b)
    for b in bl:
        assert np.allclose(np.linalg.eigvalsh(b), eig.eigvalshx(b))
    bl = []
    for a in al:
        b = np.copy(a)
        bl.append(b)
    for b in bl:
        en, vn = np.linalg.eigh(b)
        ex, vx = eig.eighx(b)
        assert np.allclose(en, ex)
        for i in range(vn.shape[1]):
            assert np.allclose(
                np.linalg.norm(np.multiply(vn[:, i], vx[:, i]), ord=1), 1
            )
