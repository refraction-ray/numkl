import os
import platform

## replace disutils and also watch out! https://stackoverflow.com/questions/29048623/does-setuptools-build-ext-behaves-differently-from-distutils-one
from Cython.Distutils import build_ext
from numpy.distutils.system_info import get_info

# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import Extension, find_packages, setup

from numkl import __author__, __version__

## cython loop import https://stackoverflow.com/questions/37471313/setup-requires-with-cython
## https://github.com/pypa/setuptools/issues/1317, Maybe only PEP518 is the way out, but still a long way to go.


try:
    mklroot = os.environ["MKLROOT"]
except KeyError:
    pass

# os.environ["CC"] = "icc"
# os.environ["LDSHARED"] = "icc -shared"
## try use intel compiler as introduced in https://software.intel.com/en-us/articles/thread-parallelism-in-cython

mkl_info = get_info("mkl")
libs = [
    "mkl_intel_ilp64",  ## mkl_rt and runtime MKL_INTERFACE_LAYER policy doesn't work well for ilp64,
    ## maybe directly linking to ilp interface is not a bad idea
    "mkl_intel_thread",
    "mkl_core",
    "iomp5",
    "pthread",
    "m",
]

lib_dirs = mkl_info.get("library_dirs")
if lib_dirs is None:
    if not mklroot:
        raise Exception("environment variable MKLROOT is not set")
    else:
        print("Using MKLROOT defined library path")
        lib_dirs = [mklroot + "/lib/intel64"]
include_dirs = mkl_info.get("include_dirs")
if include_dirs is None:
    if not mklroot:
        raise Exception("environment variable MKLROOT is not set")
    else:
        print("Using MKLROOT defined include path")
        include_dirs = [mklroot + "/include"]

osinfo = platform.system()
if osinfo == "Darwin":  # MacOS, clang
    # flags = ["-O3", "-openmp", "-march=native"]
    flags = []
    ## openmp + mkl on mac: https://zhuanlan.zhihu.com/p/48484576
    ## possible relevant posts: https://github.com/ContinuumIO/anaconda-issues/issues/8803
    ## not workable for now: "clang-4.0: error: no such file or directory: 'build/temp.macosx-10.9-x86_64-3.6/numkl/ev.o'"
    ## somehow in CI osx env, the .o file is not generated.
    ## or if using no flag or -fopenmp flag, the error is delayed to runtime when importing numkl.ev
    ## ImportError: dlopen(/Users/travis/miniconda3/conda-bld/numkl_1564029926743/_test_env_placehold/lib/python3.6/site-packages/numkl/ev.cpython-36m-darwin.so, 2):
    ## Symbol not found: _mkl_blas_caxpy
elif osinfo == "Linux":
    flags = ["-O3", "-fopenmp", "-xhost"]

with open("README.md", "r") as fh:
    long_description = fh.read()

ev = Extension(
    "numkl.ev",
    ["numkl/ev.pyx"],
    define_macros=[("MKL_ILP64",)],
    include_dirs=include_dirs,
    libraries=libs,
    library_dirs=lib_dirs,
    extra_compile_args=flags,
    # see https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations for cpu specific optimization flag
    extra_link_args=flags,  # -qopt-zmm-usage=high
)

setup(
    name="numkl",
    version=__version__,
    author=__author__,
    author_email="refraction-ray@protonmail.com",
    description="A thin cython/python wrapper on some routines from Intel MKL",
    long_description=long_description,
    url="https://github.com/refraction-ray/numkl",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy>=1.16", "cython>=0.29"],
    ext_modules=[ev],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ),
)

## wheel upload is not supported, see https://stackoverflow.com/questions/50690526/how-to-publish-binary-python-wheels-for-linux-on-a-local-machine
## also, it doesn't make sense to share the wheel at the beginning, due to the external mkl so library
