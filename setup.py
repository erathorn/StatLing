import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from setuptools import setup, find_packages
from Cython.Distutils import build_ext

from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy

if sys.platform == "win32":
    openmp = '/Qopenmp'
    opti = '/O2'
    march = "/march=native"
else:
    openmp = "-fopenmp"
    opti = "-O2"
    march = "-march=native"

ext_modules = [

    Extension("src.C_Extensions.sample", ["src/C_Extensions/sample.pyx"], include_dirs=[".", scipy.get_include()],
              extra_compile_args=[opti, march],
              extra_link_args=[opti, march]
              ),
    Extension("src.C_Extensions.helper", ["src/C_Extensions/helper.pyx"],
              include_dirs=[".", numpy.get_include()], extra_compile_args=[openmp, opti, march],
              extra_link_args=[openmp, opti, march]),
    Extension("src.C_Extensions.Trans_cy", ["src/C_Extensions/Trans_cy.pyx"],
              include_dirs=["src/C_Extensions", numpy.get_include()]),
    Extension("src.C_Extensions.Trans_cy_TKF91", ["src/C_Extensions/Trans_cy_TKF91.pyx"],
              include_dirs=["src/C_Extensions", ".", numpy.get_include()]),
    Extension("src.C_Extensions.algorithms_cython", sources=["src/C_Extensions/fw_log.c","src/C_Extensions/algorithms_cython.pyx"],
              include_dirs=["src/C_Extensions", ".", numpy.get_include()], extra_compile_args=[openmp, opti, march],
              extra_link_args=[openmp, opti, march],
              libraries=["fw_log"], library_dirs=["src/C_Extensions"]
              ),
    Extension("src.C_Extensions.Feature_Single_functions", ["src/C_Extensions/Feature_Single_functions.pyx"],
               include_dirs=["src/C_Extensions/",".", numpy.get_include(), scipy.get_include()],extra_compile_args=[openmp, opti, march],
                extra_link_args=[openmp, opti, march]),
    Extension("src.C_Extensions.pairwise_agglo", ["src/C_Extensions/pairwise_agglo.pyx"],
               include_dirs=["src/C_Extensions/",".", numpy.get_include()], extra_compile_args=[openmp],
               extra_link_args=[openmp])

]

setup(name="C_Extensions",
      packages=find_packages(),
      ext_modules=cythonize(ext_modules),
      options={'build_ext': {'inplace': True}}
      )
