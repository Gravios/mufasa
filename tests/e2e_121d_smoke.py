"""End-to-end smoke for patch 121d.

Verifies that --const-accel-segments produces different smoothed
output than baseline. With const-accel disabled the model is
constant-velocity (CV); enabling it adds an acceleration block
that lets the predictor extrapolate with curvature.

The synthetic session has fast accelerating motion (a hard turn
mid-trajectory) — exactly what the CV model under-fits. The CA
model should produce a noticeably different (and tighter) fit
during the turn.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/claude/fresh")
sys.path.insert(0, str(REPO))


def make_session(out_path, T=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    markers = [
        "back2", "back1", "back3", "lateral_left", "lateral_right",
        "center", "back4", "neck", "headmid",
        "nose", "ear_left", "ear_right",
        "tailbase", "tailmid", "tailend",
    ]
    rest = {
        "back2": (0, 0), "back1": (5, 0), "back3": (-5, 0),
        "lateral_left": (0, 5), "lateral_right": (0, -5),
        "center": (2.5, 0), "back4": (-15, 0), "neck": (15, 0),
        "headmid": (30, 0), "nose": (40, 0),
        "ear_left": (33, 5), "ear_right": (33, -5),
        "tailbase": (-30, 0), "tailmid": (-45, 0), "tailend": (-60, 0),
    }
    # Trajectory: rat sits still for 100 frames, then makes a
    # hard turn (centripetal accel ~ 800 px/s²) over 100 frames,
    # then sits still again. Tests the CA predictor's ability to
    # extrapolate curvature.
    t = np.arange(T) / 30.0
    cx = np.full(T, 200.0)
    cy = np.full(T, 200.0)
    heading = np.zeros(T)
    for i in range(T):
        if i < 100:
            heading[i] = 0.0
        elif i < 200:
            # Smooth turn: heading varies as sin(...), giving
            # nonzero second derivative throughout
            phase = (i - 100) / 100 * np.pi
            heading[i] = 1.5 * np.sin(phase)  # rad
        else:
            heading[i] = 0.0
    # Marker positions: rotate rest offsets by heading
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    for m in markers:
        ox, oy = rest[m]
        rx = ox * cos_h - oy * sin_h
        ry = ox * sin_h + oy * cos_h
        cols[f"{m}_x"] = cx + rx + rng.normal(0, 0.5, T)
        cols[f"{m}_y"] = cy + ry + rng.normal(0, 0.5, T)
        cols[f"{m}_p"] = np.full(T, 0.95)
    pd.DataFrame(cols).to_parquet(out_path, index=False)


def run(in_dir, out_dir, extra_flags):
    cmd = [
        sys.executable, "-m",
        "mufasa.data_processors.kalman_pose_smoother_v2",
        str(in_dir),
        "--output-dir", str(out_dir),
        "--likelihood-threshold", "0.5", "--fps", "30",
        "--em-max-iter", "2", "--workers", "1",
        "--no-validate", "--no-warm-start-sigma", "--no-perspective",
    ] + extra_flags
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=180,
    )


def diff_stats(df_a, df_b):
    common = [
        c for c in df_a.columns
        if c in df_b.columns
        and df_a[c].dtype.kind in "fiu"
        and not c.endswith("_var_x")
        and not c.endswith("_var_y")
    ]
    n_diff = 0
    max_d = 0.0
    for c in common:
        a = df_a[c].values
        b = df_b[c].values
        if a.shape != b.shape:
            continue
        d = np.max(np.abs(a - b))
        if d > 1e-8:
            n_diff += 1
            max_d = max(max_d, d)
    return n_diff, max_d, len(common)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    in_dir = td / "in"
    in_dir.mkdir()
    make_session(in_dir / "s0.parquet")

    results = {}
    for label, flags in [
        ("baseline", []),
        ("ca_body", ["--const-accel-segments", "body"]),
        ("ca_body_head", ["--const-accel-segments", "body,head"]),
    ]:
        out_dir = td / f"out_{label}"
        proc = run(in_dir, out_dir, flags)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1500:])
            raise SystemExit(f"failed: {label}")
        files = list(out_dir.glob("*_smoothed_v2.*"))
        if not files:
            raise SystemExit(f"no output: {label}")
        df = pd.read_parquet(files[0])
        results[label] = df
        print(f"  {label}: {df.shape[0]} frames, {df.shape[1]} cols")

    df_base = results["baseline"]
    df_ca_b = results["ca_body"]
    df_ca_bh = results["ca_body_head"]

    n1, d1, n_total = diff_stats(df_base, df_ca_b)
    print(f"\nbaseline vs ca_body:")
    print(f"  columns differing: {n1} / {n_total}")
    print(f"  max |diff|: {d1:.4f}")

    n2, d2, _ = diff_stats(df_base, df_ca_bh)
    print(f"\nbaseline vs ca_body_head:")
    print(f"  columns differing: {n2} / {n_total}")
    print(f"  max |diff|: {d2:.4f}")

    if n1 == 0:
        raise SystemExit(
            "FAIL: --const-accel-segments body produced "
            "identical output to baseline. F/Q wiring may "
            "be broken."
        )
    if n2 == 0:
        raise SystemExit(
            "FAIL: --const-accel-segments body,head produced "
            "identical output to baseline."
        )

    print("\nE2E smoke passed: 121d's const-accel changes smoothed output")
