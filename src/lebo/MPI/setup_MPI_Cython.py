from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="MPI_Cython",
    sources=["MPI_Cython.pyx"],
    include_dirs=[np.get_include()],
    libraries=["m"],  
    extra_compile_args=["-O3", "-ffast-math"],
    extra_link_args=["-lm"],
)

setup(
    name="MPI_Cython",
    ext_modules=cythonize([ext], language_level=3, annotate=False),
    zip_safe=False,
)

