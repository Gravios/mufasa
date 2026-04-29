"""Minimal setup.py for compiling Cython extension modules.

The bulk of the Mufasa package configuration lives in pyproject.toml
(metadata, dependencies, entry points, package discovery). This file
exists only because setuptools' declarative-only configuration
doesn't currently support Cython extension compilation cleanly —
``cythonize()`` needs to run at build time and that requires a
build-time Python script.

If/when setuptools adds first-class Cython support (or we switch to
scikit-build / meson-python), this file can be deleted.

Behavior:
* Reads everything except ext_modules from pyproject.toml.
* Adds Cython-compiled extensions defined here.
* Numpy include path is added automatically — required for the
  ``cimport numpy`` in the .pyx files.

If Cython isn't installed at build time, the build fails with a
clear error pointing at the build-system requires in pyproject.toml.
"""
from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


# Keep the extension list short and explicit — one entry per .pyx.
# As more kernels port from numba to Cython, add them here.
_EXTENSIONS = [
    Extension(
        name="mufasa._native.inside_polygon",
        sources=["mufasa/_native/inside_polygon.pyx"],
        include_dirs=[np.get_include()],
        # -O3 enables auto-vectorization and inlining. -ffast-math
        # is intentionally NOT enabled — we want byte-equivalent
        # output to the numba version, and -ffast-math reorders
        # floating-point ops in ways that diverge.
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.inside_rectangle",
        sources=["mufasa/_native/inside_rectangle.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.inside_circle",
        sources=["mufasa/_native/inside_circle.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.euclidean_distance",
        sources=["mufasa/_native/euclidean_distance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.angle3pt",
        sources=["mufasa/_native/angle3pt.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.border_distances",
        sources=["mufasa/_native/border_distances.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-Wall"],
    ),
    Extension(
        name="mufasa._native.hull",
        sources=["mufasa/_native/hull.pyx"],
        include_dirs=[np.get_include()],
        # OpenMP for the prange in the per-frame loop.
        # On Linux, gcc/clang use -fopenmp for both compile and link.
        extra_compile_args=["-O3", "-Wall", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]


setup(
    ext_modules=cythonize(
        _EXTENSIONS,
        # language_level is also set per-file via the .pyx header
        # comment, but specifying it here is belt-and-suspenders.
        compiler_directives={"language_level": "3"},
        # Annotation HTML helps debugging — produces an .html file
        # next to each .pyx showing which lines compile to C and
        # which fall back to Python. Off by default; enable when
        # tuning a kernel.
        annotate=False,
    ),
)
