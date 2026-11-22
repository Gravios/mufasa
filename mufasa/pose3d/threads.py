"""
mufasa.pose3d.threads
=====================

Patch 122lc: one thread per worker process.

3-D inference runs videos (or chunks of one video) in worker processes, one
per core. Each process loaded NumPy, SciPy and PyTorch with their own thread
pools: OpenBLAS, MKL, OpenMP and PyTorch's, each with a thread per core. The
solver held PyTorch to one thread, but every other step let the pools grow.
With 8 workers on a 16-thread machine that came to about 118 runnable threads
(load average 118), the CPU 98 % busy and spinning, and 10 videos in 14 hours
instead of about 8 every 5 minutes.

:func:`one_thread` is the initializer of every worker pool. It sets the
environment for libraries not loaded yet, limits the pools already loaded
(``threadpoolctl``, when installed), and PyTorch's own. A single process
(one worker) keeps the libraries' defaults.
"""
from __future__ import annotations

import os

ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def one_thread() -> None:
    """Limit this process's numeric libraries to one thread (a worker pool's initializer)."""
    for v in ENV_VARS:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)                 # OpenBLAS, MKL, OpenMP already loaded
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:                         # only before PyTorch's first parallel work
            pass
    except Exception:  # noqa: BLE001
        pass


def thread_counts() -> dict:
    """The thread pools of this process: {library: threads} (for checks)."""
    out = {}
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            out[f"{p.get('internal_api')}:{os.path.basename(p.get('filepath', ''))}"] = p.get("num_threads")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        out["torch"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return out


def worker_pool(n: int, mp_context=None):
    """A process pool whose workers each use one thread."""
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=int(n), mp_context=mp_context, initializer=one_thread)


__all__ = ["ENV_VARS", "one_thread", "thread_counts", "worker_pool"]
