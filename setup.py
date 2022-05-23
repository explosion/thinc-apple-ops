#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy

from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize


PACKAGES = find_packages()
MOD_NAMES = ["thinc_apple_ops.blas"]


def clean(path):
    for name in MOD_NAMES:
        name = name.replace(".", "/")
        for ext in [".so", ".html", ".cpp", ".c"]:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


def setup_package():
    extensions = [
        Extension(
            "thinc_apple_ops.blas",
            ["thinc_apple_ops/blas.pyx"],
            include_dirs=[numpy.get_include()],
            libraries=["blas"],
        ),
        Extension(
            "thinc_apple_ops.ops",
            ["thinc_apple_ops/ops.pyx"],
            language="c++",
            include_dirs=[numpy.get_include()],
        ),
    ]

    setup(
        name="thinc_apple_ops",
        zip_safe=True,
        packages=PACKAGES,
        package_data={"": ["*.pyx", "*.pxd"]},
        ext_modules=cythonize(extensions),
    )


if __name__ == "__main__":
    setup_package()
