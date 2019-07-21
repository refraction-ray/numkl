import os

# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import setup, Extension, find_packages

## replace disutils and also watch out! https://stackoverflow.com/questions/29048623/does-setuptools-build-ext-behaves-differently-from-distutils-one
from Cython.Distutils import build_ext
## cython loop import https://stackoverflow.com/questions/37471313/setup-requires-with-cython
from numkl import __version__, __author__

try:
    mklroot = os.environ["MKLROOT"]
except KeyError:
    raise Exception("envarionment variable MKLROOT is not set")

os.environ["CC"] = "icc"
os.environ["LDSHARED"] = "icc -shared"
## try use intel compiler as introduced in https://software.intel.com/en-us/articles/thread-parallelism-in-cython

with open("README.md", "r") as fh:
    long_description = fh.read()

ev = Extension(
    "numkl.ev",
    ["numkl/ev.pyx"],
    define_macros=[("MKL_ILP64",)],
    include_dirs=[mklroot + "/include"],
    libraries=[
        "mkl_intel_ilp64",
        "mkl_intel_thread",
        "mkl_core",
        "iomp5",
        "pthread",
        "m",
        "dl",
    ],
    library_dirs=[mklroot + "/lib/intel64"],
    extra_compile_args=["-O3", "-fopenmp", "-xhost"],
    # see https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations for cpu specific optimization flag
    extra_link_args=["-O3", "-fopenmp", "-xhost"],  # -qopt-zmm-usage=high
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
        "Operating System :: OS Independent",
    ),
)

## wheel upload is not supported, see https://stackoverflow.com/questions/50690526/how-to-publish-binary-python-wheels-for-linux-on-a-local-machine
## also, it doesn't make sense to share the wheel at the beginning, due to the external mkl so library