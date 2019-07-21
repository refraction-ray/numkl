import numpy as np


def csr2upack(ma, dtype=np.float64):  ## change csr to col_major U part packed matrix
    n = ma.shape[0]
    indices = ma.indices
    indptr = ma.indptr
    data = ma.data
    hp = np.zeros(int(n * (n + 1) / 2), dtype=dtype)
    for i in range(n):
        for j, k in enumerate(indices[indptr[i] : indptr[i + 1]]):
            if i <= k:
                hp[int(i + k * (k + 1) / 2)] = data[indptr[i] + j]
    return hp
