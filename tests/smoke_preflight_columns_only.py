"""Test for the _read_columns_only helper used in preflight.

User report: preflight took several minutes for a 67-video CSV
project because the per-target loop called read_df on every
existing CSV file. read_df does a full pyarrow parse — minutes
for multi-MB files × 67. Header-only reads are sub-millisecond.

The helper takes the fast path for csv (pd.read_csv nrows=0) and
parquet (pyarrow schema metadata), with a fallback to read_df
for unknown formats or when the fast path raises.

Sandbox-runnable: pure pandas/pyarrow, no Cython or numba.

    PYTHONPATH=. python tests/smoke_preflight_columns_only.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


def _load_helper():
    """Extract _read_columns_only from feature_subsets.py and exec
    into a namespace. We can't import the module directly because
    of dependency chains."""
    src = Path("mufasa/feature_extractors/feature_subsets.py").read_text()
    tree = ast.parse(src)
    helper = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_read_columns_only"
    )
    helper_src = ast.unparse(helper)
    # Provide a stub for read_df since the helper falls back to it
    # on the unknown-format path. The fast paths we test below
    # never hit the fallback.
    def read_df_stub(file_path: str, file_type: str):
        raise RuntimeError(
            "read_df fallback hit — fast path should have succeeded"
        )
    ns = {"read_df": read_df_stub}
    exec(helper_src, ns)
    return ns["_read_columns_only"]


def main() -> int:
    helper = _load_helper()

    # ------------------------------------------------------------------ #
    # Case 1: CSV fast path returns the right column names
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
    df = pd.DataFrame({
        "Distance (mm) Nose-Tail": [1.0, 2.0, 3.0],
        "Animal_1 movement Nose (mm)": [4.0, 5.0, 6.0],
        "Animal_1 convex hull area (mm2)": [7.0, 8.0, 9.0],
    })
    df.to_csv(csv_path, index=False)
    cols = helper(csv_path, "csv")
    assert cols == set(df.columns), (
        f"CSV fast path wrong cols: {cols} != {set(df.columns)}"
    )
    os.unlink(csv_path)

    # ------------------------------------------------------------------ #
    # Case 2: Parquet fast path returns the right column names
    # (only run if pyarrow.parquet is available)
    # ------------------------------------------------------------------ #
    try:
        import pyarrow  # noqa
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            pq_path = f.name
        df.to_parquet(pq_path)
        cols = helper(pq_path, "parquet")
        assert cols == set(df.columns), (
            f"Parquet fast path wrong cols: {cols} != {set(df.columns)}"
        )
        os.unlink(pq_path)
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # Case 3: speed comparison — header-only beats full read by orders
    # of magnitude on a moderately large CSV. Build a synthetic 50K-row
    # × 100-col CSV and time both reads.
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        big_csv = f.name
    big_df = pd.DataFrame({
        f"col_{i}": list(range(50_000)) for i in range(100)
    })
    big_df.to_csv(big_csv, index=False)
    file_size_mb = os.path.getsize(big_csv) / (1024 * 1024)

    t0 = time.perf_counter()
    cols_fast = helper(big_csv, "csv")
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_df = pd.read_csv(big_csv)
    t_full = time.perf_counter() - t0
    cols_full = set(full_df.columns)

    speedup = t_full / max(t_fast, 1e-9)
    print(
        f"  Header-only: {t_fast*1000:.2f}ms"
        f" | Full read: {t_full*1000:.2f}ms"
        f" | Speedup: {speedup:.0f}× on {file_size_mb:.1f}MB CSV"
    )
    assert cols_fast == cols_full, "Header-only got different cols than full read"
    # Header-only should be at least 3× faster on a 50K-row file. In
    # practice it's usually 50-500× faster, but CI/sandbox VMs vary.
    assert speedup >= 3, (
        f"Header-only should be substantially faster; got {speedup:.1f}×"
    )
    os.unlink(big_csv)

    # ------------------------------------------------------------------ #
    # Case 4: feature_subsets.py uses _read_columns_only in its
    # preflight per-target loop (not read_df)
    # ------------------------------------------------------------------ #
    src = Path("mufasa/feature_extractors/feature_subsets.py").read_text()
    # Find the preflight_check method
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FeatureSubsetsCalculator"
    )
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    pf_src = ast.unparse(methods["preflight_check"])
    assert "_read_columns_only" in pf_src, (
        "preflight_check must use _read_columns_only for per-target "
        "header reads (was using read_df, which is full file parse)"
    )
    # Confirm it's the OLD slow path that's been removed from per-target
    # loop — read_df should NOT appear in the per-target reading.
    # (read_df CAN still appear in the probe-output read; that's a
    # single small file that just got written, no perf concern.)
    # Find the per-target loop region specifically
    target_loop_marker = "for label, dir_path in targets:"
    assert target_loop_marker in pf_src
    target_loop_idx = pf_src.index(target_loop_marker)
    target_loop_src = pf_src[target_loop_idx:target_loop_idx + 2000]
    assert "read_df" not in target_loop_src, (
        "read_df should not appear in the per-target loop — that's "
        "the slow path we replaced"
    )

    print("smoke_preflight_columns_only: 4/4 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
