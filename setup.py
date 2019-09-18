#!/usr/bin/env python3

from setuptools import setup, find_packages
from Cython.Build import cythonize

import numpy as np
import pgmpy


import os

os.environ['CFLAGS'] = "-march=native"

USE_SIMD = False

if USE_SIMD:
    env_flags = {'SIMD': True}
else:
    env_flags = {'SIMD': False}

setup(
    name="pgmpy",
    version=pgmpy.__version__,
    description="A library for Probabilistic Graphical Models",
    packages=find_packages(exclude=["tests"]),
    author="Ankur Ankan",
    author_email="ankurankan@gmail.com",
    url="https://github.com/pgmpy/pgmpy",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
    ],
    long_description="https://github.com/pgmpy/pgmpy/blob/dev/README.md",
    install_requires=[],
    ext_modules=cythonize(["pgmpy/estimators/BGeScore.pyx",
                           "pgmpy/cython_backend/linear_algebra.pyx",
                           "pgmpy/cython_backend/covariance_simd.pyx",
                           "pgmpy/cython_backend/covariance.pyx"], annotate=True, compile_time_env=env_flags),
    include_dirs=[np.get_include()],
    zip_safe=False
)
