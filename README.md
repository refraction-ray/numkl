# NUMKL

[![version](https://img.shields.io/pypi/v/numkl.svg)](https://pypi.org/project/numkl/)
[![conda](https://anaconda.org/refraction-ray/numkl/badges/version.svg)](https://anaconda.org/refraction-ray/numkl)


This package works as the python wrapper to directly call some MKL routines while keep the same interface with numpy.

## Install

Use `pip install numkl`

Or `conda install -c refraction-ray numkl`

You should make sure Intel MKL library and Intel compilers are installed and configured for relevant enviroment variables. Especially, environment variable `MKLROOT` is necessary for pip installation. And proper `LD_LIBRARY_PATH` is necessary in runtime.

Currently, you also need cython preinstalled in your python enviroment for pip installation. 

Note this package is in its very early age, no guarantee on successful installation and usage. And this package only supports linux. 

## Example

```python
import numpy as np
from numkl import eig
a = np.array([[0.,1.0],[1.0,0.]])
e,v = eig.eighx(a)
```

## Why

This package is not reinventing wheels like numpy, instead, it provide features that current numpy doesn't provide.

For the symmetric or Hermitian matrix eigenproblem, numpy has already provided the interface `numpy.linalg.eigh` and `numpy.linalg.eigvalsh`. By correctly configuring and linking, these two functions also can directly calling MKL routines. So why bother?

There are at least two aspects why numpy eigenproblem interface is not good enough:

1. The 32 bit int overflow and unable to calculate eigenproblem for large matrix. See [this issue](https://github.com/numpy/numpy/issues/13956). Note currently this issue cannot be solve by simply hacking the compiling parameters, instead one need to change the source code of numpy.
2. The memory waste due to keeping the input matrix. See [this issue](https://github.com/numpy/numpy/issues/14024). Actually, it costs two times of the space in numpy for getting all eigenvalues than directly using lapack routine.

In a word, this package is designed for "push-to-the-limit" style computations, where you can compute the eigenproblem for matrix dimension larger than 32766. And the interface is seamlessly integrated with numpy.