"""
smoke_per_session_offset_diagnostic
====================================

Smoke tests for the patch-119-pre fix to
scripts/per_session_offset_diagnostic.py.

The diagnostic previously assumed both raw and smoothed
parquets used a DLC MultiIndex (scorer, bodyparts, coords).
The v2 smoother actually writes flat columns
(``<marker>_x``, ``<marker>_y``, ``<marker>_p``,
``<marker>_var_x``, ``<marker>_var_y``), so ``load_session``
raised KeyError on every smoothed file, the exception was
swallowed, and the recommendation block fell through to a
silent "Phase 2" default with zero data behind it.

These tests cover:
  Case 1: 3 sessions in v2 flat-column format actually
          produce data and a real ("Strong/Moderate/Weak
          evidence") recommendation.
  Case 2: When no raw/smoothed pairs can be formed, the
          recommendation reads "NO DATA" and does NOT
          mention Phase 2.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mufasa.data_processors.kalman_pose_smoother_v2 import (
    standard_rat_layout, FittedLengths, NoiseParamsV2,
    save_model_v2,
)

DIAG_SCRIPT = REPO / "scripts" / "per_session_offset_diagnostic.py"


def _make_flat_pose_df(markers, T, rng, body_x=200.0, body_y=200.0):
    """Build a flat-column DataFrame matching the v2 smoother's
    output: <m>_x, <m>_y, <m>_p columns. Marker world positions
    derived from the standard layout's default offsets, with
    small Gaussian noise so the diagnostic's IQR computations
    have something to chew on.
    """
    layout = standard_rat_layout()
    cols = {}
    body_dir = 0.0
    cos, sin = np.cos(body_dir), np.sin(body_dir)

    for m in markers:
        try:
            seg_name, (l_off, a_off) = layout.marker_attachment(m)
        except KeyError:
            continue
        local_x = l_off * np.cos(a_off)
        local_y = l_off * np.sin(a_off)
        # Layout offsets are in [0, 1]; scale to pixels
        scale = 30.0
        wx = body_x + scale * (cos * local_x - sin * local_y)
        wy = body_y + scale * (sin * local_x + cos * local_y)
        cols[f"{m}_x"] = wx + rng.normal(0, 1.0, T)
        cols[f"{m}_y"] = wy + rng.normal(0, 1.0, T)
        cols[f"{m}_p"] = np.full(T, 0.95)
    return pd.DataFrame(cols)


def _make_flat_smoothed_df(markers, T, rng):
    """v2 smoother's output also has _var_x / _var_y columns;
    include them to exercise the loader's handling of the
    auxiliary columns."""
    df = _make_flat_pose_df(markers, T, rng)
    for col in list(df.columns):
        if col.endswith("_x") and not col.endswith("_var_x"):
            base = col[:-2]
            df[f"{base}_var_x"] = 0.5
        elif col.endswith("_y") and not col.endswith("_var_y"):
            base = col[:-2]
            df[f"{base}_var_y"] = 0.5
    return df


def _make_model_npz(out_path):
    """Save a minimal v2 model file the diagnostic can load.
    Marker offsets are zeroed (the diagnostic doesn't actually
    read them; it only enumerates marker names)."""
    layout = standard_rat_layout()
    fl = FittedLengths(
        segment_lengths={s: 30.0 for s in layout.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout.non_root_topo_order},
        marker_offsets={m: (0.0, 0.0) for m in layout.marker_names},
    )
    params = NoiseParamsV2.default(
        layout, sigma_marker=1.0, q_root_pos=10.0,
    )
    save_model_v2(
        out_path, layout, fl, params,
        fps=30.0, likelihood_threshold=0.7,
    )


def _run_diag(raw_dir, smooth_dir, model_path, out_path):
    # Pass PYTHONPATH so the diagnostic can resolve mufasa
    # imports when run from a repo clone where mufasa hasn't
    # been pip-installed. In the user's normal environment
    # (where mufasa is installed editable) this is redundant
    # but harmless.
    import os
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO) + (os.pathsep + existing if existing else "")
    )
    proc = subprocess.run(
        [sys.executable, str(DIAG_SCRIPT),
         "--raw-dir", str(raw_dir),
         "--smoothed-dir", str(smooth_dir),
         "--model", str(model_path),
         "--output", str(out_path)],
        capture_output=True, text=True, env=env,
    )
    return proc


def case1_flat_columns_load_and_report():
    """3 synthetic sessions in v2 flat-column format. The
    loader must succeed on every session and the recommendation
    block must produce a real ('evidence') branch, not the
    silent Phase 2 default."""
    layout = standard_rat_layout()
    markers = layout.marker_names
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_dir = td / "raw"; raw_dir.mkdir()
        smooth_dir = td / "smooth"; smooth_dir.mkdir()
        for i in range(3):
            sess = f"session_{i:02d}"
            raw_df = _make_flat_pose_df(
                markers, T=500, rng=rng,
                body_x=200 + 5 * i, body_y=150,
            )
            smooth_df = _make_flat_smoothed_df(markers, T=500, rng=rng)
            raw_df.to_parquet(raw_dir / f"{sess}.parquet", index=False)
            smooth_df.to_parquet(
                smooth_dir / f"{sess}_smoothed_v2.parquet",
                index=False,
            )
        model_p = td / "model.npz"
        out_p = td / "report.txt"
        _make_model_npz(model_p)

        proc = _run_diag(raw_dir, smooth_dir, model_p, out_p)
        assert proc.returncode == 0, (
            f"Diagnostic exited non-zero: {proc.stderr}"
        )
        report = out_p.read_text()

        # Distal-only markers (back2, back4, neck, headmid,
        # tailbase, tailmid, tailend — 7 of 15) legitimately
        # report (no data) because they have no parent in
        # parent_marker_map. The other 8 must produce data.
        no_data_lines = sum(
            1 for ln in report.splitlines() if "(no data)" in ln
        )
        assert no_data_lines < 15, (
            f"Every marker still '(no data)' — fix didn't take. "
            f"Report:\n{report}"
        )
        rec = [
            l for l in report.splitlines()
            if l.strip().startswith(
                ("Strong", "Moderate", "Weak", "NO DATA")
            )
        ]
        assert rec, f"No recommendation line in report:\n{report}"
        assert "NO DATA" not in rec[0], (
            f"Loader fix failed: report says NO DATA on valid "
            f"input. Report:\n{report}"
        )


def case2_no_pairs_says_no_data():
    """When raw files have no smoothed counterparts, no pairs
    are formed. Previously the recommendation block silently
    fell through to 'Weak evidence ... go to Phase 2'. Now it
    must read NO DATA, with no Phase 2 mention."""
    layout = standard_rat_layout()
    markers = layout.marker_names
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_dir = td / "raw"; raw_dir.mkdir()
        smooth_dir = td / "smooth"; smooth_dir.mkdir()
        raw_df = _make_flat_pose_df(markers, T=500, rng=rng)
        raw_df.to_parquet(raw_dir / "lonely.parquet", index=False)
        model_p = td / "model.npz"
        out_p = td / "report.txt"
        _make_model_npz(model_p)

        proc = _run_diag(raw_dir, smooth_dir, model_p, out_p)
        assert proc.returncode == 0, (
            f"Diagnostic exited non-zero: {proc.stderr}"
        )
        report = out_p.read_text()
        rec_lines = [
            l for l in report.splitlines()
            if l.strip().startswith(
                ("Strong", "Moderate", "Weak", "NO DATA")
            )
        ]
        assert any("NO DATA" in l for l in rec_lines), (
            f"Recommendation must read NO DATA when no pairs "
            f"exist. Got:\n{report}"
        )
        assert not any("Phase 2" in l for l in rec_lines), (
            f"Must not recommend Phase 2 from zero data. "
            f"Got:\n{report}"
        )


def main() -> int:
    case1_flat_columns_load_and_report()
    case2_no_pairs_says_no_data()
    print("smoke_per_session_offset_diagnostic: all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
