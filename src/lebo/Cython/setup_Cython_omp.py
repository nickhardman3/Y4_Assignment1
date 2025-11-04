from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

if sys.platform.startswith("win"):
    extra_compile_args = ["/O2", "/openmp", "/fp:fast"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        name="Cython_omp",
        sources=["Cython_omp.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="LebwohlLasherCythonOMP",
    ext_modules=cythonize(extensions, language_level=3),
)
