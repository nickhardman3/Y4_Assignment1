from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="Cython",
        sources=["Cython.pyx"],
        include_dirs=[np.get_include()],
        language="c"
    )
]

setup(
    name="LebwohlLasherCython",
    ext_modules=cythonize(extensions, language_level=3),
)
