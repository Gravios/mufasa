#!/usr/bin/env python3
"""
benchmark_circular_stats.py
===========================

Timing harness for the circular-statistics CPU vs GPU paths.

Runs each of the four dispatched functions over a synthetic angular
time-series at representative sizes, on both the CPU and GPU paths,
and prints a speedup matrix. Intended for calibration on nphy-069
(RTX 5070 Ti / Blackwell).

Usage:

    python scripts/benchmark_circular_stats.py
    python scripts/benchmark_circular_stats.py --sizes 100000 500000 2000000
    python scripts/benchmark_circular_stats.py --gpu-only
    python scripts/benchmark_circular_stats.py --cpu-only

By default runs N=[10k, 100k, 1M, 5M] with 4 sliding windows of
[0.5, 1.0, 2.0, 5.0] seconds at fps=30. For 5M frames that's roughly
the size of a 45-minute recording at 30 Hz — realistic upper bound
for SimBA/Mufasa workloads.

Expected order-of-magnitude speedups on RTX 5070 Ti (Blackwell, CC 12.0,
16 GB GDDR7):

    N=100k     : GPU ~3-5x    (transfer overhead dominates)
    N=1M       : GPU ~15-25x  (GPU threads saturated)
    N=5M       : GPU ~30-50x  (GPU memory bandwidth dominates)

The exact ratio varies with window count (more windows → more per-
window kernel launches → smaller speedup because launch overhead
amortises less).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable

import numpy as np


def _time_fn(fn: Callable, *args, n_iter: int = 3, warmup: int = 1) -> float:
    """Return the minimum wall time over ``n_iter`` runs, after ``warmup``
    untimed calls (to trigger numba / CUDA JIT compilation)."""
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return min(samples)  # min, not mean — avoids OS jitter / GC noise


def _bench_one(name: str, cpu_fn: Callable, gpu_fn: Callable, args: tuple,
               cpu_only: bool, gpu_only: bool) -> tuple[float, float, float]:
    """Return (t_cpu_s, t_gpu_s, speedup) or NaN-entries where skipped."""
    t_cpu = float("nan")
    t_gpu = float("nan")

    if not gpu_only:
        os.environ["MUFASA_CIRCULAR_BACKEND"] = "cpu"
        # Re-import dispatcher to pick up env var change.
        from mufasa.data_processors import circular_dispatch
        circular_dispatch.backend.cache_clear()
        circular_dispatch._env_backend_choice.cache_clear()
        t_cpu = _time_fn(cpu_fn, *args)

    if not cpu_only:
        os.environ["MUFASA_CIRCULAR_BACKEND"] = "gpu"
        from mufasa.data_processors import circular_dispatch
        circular_dispatch.backend.cache_clear()
        circular_dispatch._env_backend_choice.cache_clear()
        try:
            t_gpu = _time_fn(gpu_fn, *args)
        except Exception as e:
            print(f"    GPU failed: {type(e).__name__}: {e}")

    speedup = t_cpu / t_gpu if (t_gpu and not np.isnan(t_gpu) and t_gpu > 0 and not np.isnan(t_cpu)) else float("nan")
    return t_cpu, t_gpu, speedup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[10_000, 100_000, 1_000_000, 5_000_000])
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from mufasa.data_processors import circular_dispatch as cd
    from mufasa.ui_qt.linux_env import cuda_available, cuda_capability

    print("=" * 72)
    print(" Mufasa — circular statistics CPU vs GPU benchmark")
    print("=" * 72)
    print(f"  CUDA available: {cuda_available()}")
    cap = cuda_capability()
    if cap:
        print(f"  Compute capability: {cap[0]}.{cap[1]}")
    print(f"  CPU count (affinity): "
          f"{__import__('mufasa.ui_qt.linux_env', fromlist=['cpu_count']).cpu_count()}")
    print(f"  sizes:   {args.sizes}")
    print(f"  windows: {args.windows} s at {args.fps} fps")
    print()

    targets = [
        ("sliding_circular_mean",     cd.sliding_circular_mean),
        ("sliding_circular_std",      cd.sliding_circular_std),
        ("sliding_mean_resultant_vl", cd.sliding_mean_resultant_vector_length),
    ]

    rng = np.random.default_rng(42)
    header = f"{'fn':<30s} {'N':>10s} {'CPU (s)':>10s} {'GPU (s)':>10s} {'speedup':>9s}"
    print(header)
    print("-" * len(header))
    for N in args.sizes:
        data = rng.uniform(0, 360, size=N).astype(np.float64)
        windows = np.array(args.windows, dtype=np.float64)
        for name, fn in targets:
            t_cpu, t_gpu, sp = _bench_one(
                name, fn, fn, (data, windows, args.fps),
                cpu_only=args.cpu_only, gpu_only=args.gpu_only,
            )
            sp_str = f"{sp:6.1f}x" if not np.isnan(sp) else "  n/a"
            cpu_str = f"{t_cpu*1000:7.1f}ms" if not np.isnan(t_cpu) else "   skipped"
            gpu_str = f"{t_gpu*1000:7.1f}ms" if not np.isnan(t_gpu) else "   skipped"
            print(f"{name:<30s} {N:>10d} {cpu_str:>10s} {gpu_str:>10s} {sp_str:>9s}")
    print()
    # instantaneous_angular_velocity — different signature, benchmark separately
    print("instantaneous_angular_velocity (stride=1):")
    for N in args.sizes:
        data = rng.uniform(0, 360, size=N).astype(np.float64)
        t_cpu, t_gpu, sp = _bench_one(
            "iav", cd.instantaneous_angular_velocity, cd.instantaneous_angular_velocity,
            (data, 1), cpu_only=args.cpu_only, gpu_only=args.gpu_only,
        )
        sp_str = f"{sp:6.1f}x" if not np.isnan(sp) else "  n/a"
        cpu_str = f"{t_cpu*1000:7.1f}ms" if not np.isnan(t_cpu) else "   skipped"
        gpu_str = f"{t_gpu*1000:7.1f}ms" if not np.isnan(t_gpu) else "   skipped"
        print(f"{'iav':<30s} {N:>10d} {cpu_str:>10s} {gpu_str:>10s} {sp_str:>9s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
