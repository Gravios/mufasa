"""Smoke test for patch 122hn — longer parameter hash.

The smoother parameter hash (patch 122hm) was widened from a 4-byte digest
(8 hex chars) to an 8-byte digest (16 hex chars) for stronger collision
resistance across parameter variants, while remaining short enough to keep the
output filename readable. This locks in the new length and confirms the digest
is still deterministic, still sensitive to the parameters, and that the wider
hash flows through to the output filename.
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


EXPECTED_HEX = 16  # 8-byte blake2b digest

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

base = K._smoother_param_hash(lay, p, jp, None, 0.5, 30.0, True)

check(f"hash is {EXPECTED_HEX} hex chars", len(base) == EXPECTED_HEX)
check("hash is valid hex",
      all(c in "0123456789abcdef" for c in base))
check("hash is still deterministic at the new length",
      K._smoother_param_hash(lay, p, jp, None, 0.5, 30.0, True) == base)
# still parameter-sensitive, and the differing hash is also the new length
alt = K._smoother_param_hash(lay, p, jp, None, 0.2, 30.0, True)
check("still changes with a parameter (likelihood_threshold)", alt != base)
check("the differing hash is also the new length", len(alt) == EXPECTED_HEX)
alt2 = K._smoother_param_hash(
    lay, dc.replace(p, q_root_pos=200.0), jp, None, 0.5, 30.0, True)
check("still changes with a fitted param", alt2 != base
      and len(alt2) == EXPECTED_HEX)

# the widened hash flows into the filename
with tempfile.TemporaryDirectory() as d:
    outdir = Path(d)
    T = 6
    s = {"path": Path("/data/sessA.parquet"), "markers": ["m1", "m2"],
         "likelihoods": np.ones((T, 2)) * 0.9}
    out, _, _ = K._build_and_write_session_output(
        s, np.zeros((T, 2, 2)), np.ones((T, 2, 2)),
        {"m1": 0, "m2": 1}, outdir, param_hash=base)
    check("filename carries the full 16-char hash",
          out.name == f"sessA_smoothed_v2.{base}.parquet"
          and len(base) == EXPECTED_HEX)

# source: the digest_size is 8
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)
digest_size = None
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "blake2b":
        for kw in node.keywords:
            if kw.arg == "digest_size" and isinstance(kw.value, ast.Constant):
                digest_size = kw.value.value
check("blake2b digest_size is 8 bytes", digest_size == 8)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hn_hash_length: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
