from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("scatter_cython.pyx", annotate=True),
    include_dirs=[numpy.get_include()],  # Add NumPy headers
)

# to compile run 
# python setup_scatter.py build_ext --inplace
# in terminal under the conda env scatter_env