"""Smoke test for patch 122hm — parameter hash in the output filename.

The smoothed output filename gains a short hash of the smoother parameters:
<stem>_smoothed_v2.<hash>.parquet. The hash is a blake2b digest (8 hex chars)
of the effective smoothing configuration — the fitted noise params, the layout
flags and tree, the joint prior / perspective presence, and the smoothing-time
knobs (likelihood_threshold, fps, apply_constraints) — read from the objects
present at write time, so it is mode-independent (a trained model and the same
model reloaded hash identically).

This keeps parameter variants separate in one output dir and makes the
incremental skip (patch 122hl) parameter-aware: a different setting yields a
different filename, so the skip re-smooths instead of leaving a stale file.
"""
from __future__ import annotations

import ast
import dataclasses as dc
import pathlib
import sys
import tempfile
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import mufasa.data_processors.kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


lay = K.layout_from_segments([
    {"name": "back", "markers": ["b1", "b2"]},
    {"name": "head", "parent": "back", "markers": ["h1", "h2"]},
])
p = K.NoiseParamsV2(
    sigma_marker={"b1": 3.0, "b2": 3.0, "h1": 3.0, "h2": 3.0},
    q_root_pos=100.0, q_root_ori=10.0,
    q_seg_ori={"head": 2.0}, q_length={"head": 5.0}, tau_accel=0.5,
)
jp = K.JointPriorV2(mu={"head": 1.0}, kappa={"head": 5.0}, n_obs={"head": 100})


def h(layout=lay, params=p, jprior=jp, persp=None,
      lt=0.5, fps=30.0, ac=True):
    return K._smoother_param_hash(layout, params, jprior, persp, lt, fps, ac)


base = h()

# ---- shape & determinism ----
check("hash is 8 hex chars", len(base) == 8 and all(
    c in "0123456789abcdef" for c in base))
check("hash is deterministic", h() == base)

# ---- sensitive to every meaningful parameter ----
check("changes with likelihood_threshold", h(lt=0.2) != base)
check("changes with fps", h(fps=60.0) != base)
check("changes with apply_constraints", h(ac=False) != base)
check("changes with joint prior presence", h(jprior=None) != base)
check("changes with a fitted noise param (q_root_pos)",
      h(params=dc.replace(p, q_root_pos=200.0)) != base)
check("changes with accel_tau", h(params=dc.replace(p, tau_accel=0.1)) != base)
check("changes with high_angular_noise_segments",
      h(layout=dc.replace(lay, high_angular_noise_segments=["head"])) != base)
check("changes with const_accel_segments",
      h(layout=dc.replace(lay, const_accel_segments=["head"])) != base)
check("changes with orientation_drift_segments",
      h(layout=dc.replace(lay, orientation_drift_segments=["head"])) != base)
check("changes with joint prior kappa",
      h(jprior=K.JointPriorV2(mu={"head": 1.0}, kappa={"head": 9.9},
                              n_obs={"head": 100})) != base)

# ---- robust to meaningless float noise ----
check("7th-significant-figure noise leaves the hash unchanged",
      h(params=dc.replace(p, q_root_pos=100.0000001)) == base)

# ---- filename construction ----
with tempfile.TemporaryDirectory() as d:
    outdir = Path(d)
    T = 8
    s = {"path": Path("/data/sessA.parquet"), "markers": ["m1", "m2"],
         "likelihoods": np.ones((T, 2)) * 0.9}
    sp = np.zeros((T, 2, 2))
    sv = np.ones((T, 2, 2))
    d2l = {"m1": 0, "m2": 1}

    out, _, _ = K._build_and_write_session_output(
        s, sp, sv, d2l, outdir, param_hash="abcd1234")
    check("hashed filename is <stem>_smoothed_v2.<hash>.parquet",
          out.name == "sessA_smoothed_v2.abcd1234.parquet")

    out2, _, _ = K._build_and_write_session_output(
        s, sp, sv, d2l, outdir, param_hash=None)
    check("no hash -> legacy <stem>_smoothed_v2.parquet",
          out2.name == "sessA_smoothed_v2.parquet")

    # skip integration: current hash's file is found, a different one is not
    tag = ".abcd1234"
    pq = outdir / f"sessA_smoothed_v2{tag}.parquet"
    check("skip check finds the current-hash file", pq.exists())
    other = outdir / "sessA_smoothed_v2.ff00ff00.parquet"
    check("skip check misses a different-hash file (would re-smooth)",
          not other.exists())

# ---- wiring in smooth_pose_v2 ----
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)

# helper exists
check("_smoother_param_hash is defined",
      any(isinstance(n, ast.FunctionDef) and n.name == "_smoother_param_hash"
          for n in ast.walk(tree)))
# writer takes param_hash
writer = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_build_and_write_session_output"), None)
check("writer accepts a param_hash argument",
      writer is not None and "param_hash" in [a.arg for a in writer.args.args])
# hash computed once and threaded
check("smooth_pose_v2 computes the param hash",
      "param_hash = _smoother_param_hash(" in src)
check("both writer calls pass param_hash",
      src.count("param_hash=param_hash") == 2)
# skip check is hash-aware
check("the incremental skip uses the hash tag",
      'f"{_stem}_smoothed_v2{_tag}.parquet"' in src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hm_param_hash_filename: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
