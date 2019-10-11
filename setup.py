#!/usr/bin/env python3
import distutils
from setuptools import setup, find_packages
from Cython.Build import cythonize

import subprocess
import platform
import numpy as np
import pgmpy

import sys
import os

import shutil

os.environ['CFLAGS'] = "-march=native"

USE_SIMD = False

if USE_SIMD:
    env_flags = {'SIMD': True}
else:
    env_flags = {'SIMD': False}


def check_rust_bitness():
    toolchain = subprocess.Popen(['rustup', 'show', 'active-toolchain'], stdout=subprocess.PIPE).communicate()[0]

    toolchain = str(toolchain)
    python_bitness = platform.architecture()[0]

    if python_bitness == "32bit":
        if not "i686" in toolchain:
            print("WARNING: Python interpreter is 32-bit, while default Rust target toolchain is not 32-bit.\n"
                            "\tUse \"rustup target list\" to list all the available Rust target toolchains.\n"
                            "\tUse \"rustup target add {toolchain_name}\" to include a compatible Rust target toolchain.\n"
                            "\tUse \"rustup default {toolchain_name}\" to set a default target toolchain.\n",
                   file=sys.stderr)

    if python_bitness == "64bit":
        if not "64" in toolchain:
            print("WARNING: Python interpreter is 64-bit, while default Rust target toolchain is not 64-bit.\n"
                            "\tUse \"rustup target list\" to list all the available Rust target toolchains.\n"
                            "\tUse \"rustup target add {toolchain_name}\" to include a compatible Rust target toolchain.\n"
                            "\tUse \"rustup default {toolchain_name}\" to set a default target toolchain.\n",
                   file=sys.stderr)

def build_native(spec):
    check_rust_bitness()

    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
        path='./pgmpy/rust'
    )

    spec.add_cffi_module(
        module_path='pgmpy.factors.continuous.CKDE_CPD._ffi',
        dylib=lambda: build.find_dylib('kde_ocl_sys', in_path='target/release'),
        header_filename=lambda: build.find_header('kde-ocl-sys.h', in_path='target'),
        rtld_flags=['NOW', 'NODELETE']
    )


class CleanCommand(distutils.cmd.Command):
    """
    Our custom command to clean out junk files.
    """
    description = "Cleans out all generated files and folders."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        files_to_delete = ["pgmpy/factors/continuous/CKDE_CPD/_ffi.py",
                           "pgmpy/factors/continuous/CKDE_CPD/_ffi__ffi.py",
                           "pgmpy/factors/continuous/CKDE_CPD/_ffi__lib.so",
                           "pgmpy/factors/continuous/CKDE_CPD/_ffi__lib.pyd",
                           "pgmpy/rust/src/open_cl_code.rs"]

        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

        current_dir = os.getcwd()

        os.chdir('pgmpy/rust')
        os.system("cargo clean")
        os.chdir(current_dir)

        folders_to_delete = [".eggs", "pgmpy.egg-info", "build"]

        for folder in folders_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)

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
    setup_requires=["milksnake"],
    milksnake_tasks=[
        build_native
    ],
    install_requires=["milksnake"],
    ext_modules=cythonize(["pgmpy/estimators/BGeScore.pyx",
                           "pgmpy/cython_backend/linear_algebra.pyx",
                           "pgmpy/cython_backend/covariance_simd.pyx",
                           "pgmpy/cython_backend/covariance.pyx",
                           "pgmpy/estimators/cython_hill_climbing.pyx"
                           ], annotate=True, compile_time_env=env_flags),
    include_dirs=[np.get_include()],
    cmdclass={
        "clean": CleanCommand,
    },
    zip_safe=False
)
