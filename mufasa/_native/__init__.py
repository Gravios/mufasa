"""mufasa._native — ahead-of-time compiled feature kernels.

Modules here are written in Cython and compiled to .so files at
pip-install time. They exist to:

1. Provide drop-in replacements for the numba-jitted feature
   kernels, with byte-equivalent output.
2. Eliminate the libgomp dependency from kernel imports (numba
   pulls in OpenMP at module load, which forces the parallel
   feature extractor to use the spawn start method instead of
   fork — slower worker startup).
3. Avoid the numba JIT warm-up cost on first call.

Status: proof of concept. Currently only `inside_polygon` is
ported. If the POC validates (byte-equivalent output, comparable
or faster runtime, build system stable), more kernels will
follow.

Usage:
    The Cython kernels are NOT yet wired into the production
    feature_subset_kernels chain. The numba versions remain the
    default. Code paths that want to test the Cython version
    import directly:

        from mufasa._native.inside_polygon import (
            framewise_inside_polygon_roi as cython_inside_polygon
        )

    The byte-equivalence test in
    ``tests/smoke_native_inside_polygon.py`` exercises this.

Build dependencies:
    * Cython (build-time)
    * numpy headers (build-time, already a runtime dep)
    * a C compiler (gcc/clang on Linux)

If pip install fails to compile, run ``pip install cython numpy``
then retry.
"""
