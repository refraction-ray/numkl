import os
from distutils.core import setup
from distutils.extension import Extension
##TODO: move to setuptools?
from Cython.Distutils import build_ext

mklroot = os.environ['MKLROOT']

ev = Extension("ev",
                ["numkl/ev.pyx"],
                define_macros=[("MKL_ILP64",)],
                include_dirs=[mklroot + "/include"],
                libraries=["mkl_intel_ilp64", "mkl_intel_thread", "mkl_core", "iomp5", "pthread", "m", "dl"],
                library_dirs=[mklroot + "/lib/intel64"],
                extra_compile_args=["-O3", "-fopenmp", "-xhost"],
                # see https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations for cpu specific optimization flag
                extra_link_args=["-O3", "-fopenmp", "-xhost"]  # -qopt-zmm-usage=high
                )

setup(
    name="numkl",
    cmdclass={"build_ext": build_ext},
    ext_modules=
    [
        ev,
    ]
)
