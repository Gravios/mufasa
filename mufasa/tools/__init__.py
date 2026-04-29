"""Standalone command-line tools that ship with mufasa.

Each module here is invokable both as ``python -m mufasa.tools.<name>``
and importable for programmatic use. They're separate from the main
package so they have minimal dependencies (typically just pandas /
numpy) and don't pull in the full Cython/numba stack.
"""
