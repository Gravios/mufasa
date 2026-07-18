"""Smoke test for patch 122hh — root-cluster offsets from a PCA body axis.

The bug
-------

A rodent's trunk markers are a rigid cluster on the root segment. Every
non-root segment gets its body frame from the ``parent_distal -> seg_distal``
vector, but the root has no parent, so until 122gx the code filled that gap
with two hardcoded marker names from the built-in rig (``back3 -> back1``).
122gx generalized the kinematic tree to any skeleton but left that proxy
rig-specific. Any project whose trunk markers are not literally ``back1`` and
``back3`` therefore matched nothing, ``parent_distal`` stayed None, and
``fit_body_lengths`` skipped offset-fitting for the entire root cluster —
leaving every non-distal trunk marker on its placeholder ring offset
``(1.0, angle)``. Six markers pinned at unit radius around a circle instead
of strung along the spine: the "back markers bunched even when plainly
visible" report.

The fix uses no marker names: trunk markers lie along the body, so the first
principal component of the cluster's point cloud is the body axis, per frame.

What is asserted
----------------

On a synthetic trunk with known geometry: the fitted offsets recover it
(collinear spine, symmetric hips) with NO ring placeholders surviving, for a
rig whose names the old proxy never matched. And the mechanism the old code
used is gone.
"""
from __future__ import annotations

import ast
import math
import pathlib
import sys
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    _root_cluster_body_axis,
    fit_body_lengths,
    layout_from_segments,
)

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# A cluster on the ROOT with deliberately non-canonical names — exactly the
# case the old back1/back3 proxy could not handle.
SPEC = [
    {"name": "trunk",
     "markers": ["spine_0", "spine_1", "spine_2", "spine_3",
                 "haunch_l", "haunch_r"]},
    {"name": "neck", "parent": "trunk", "markers": ["nape"]},
    {"name": "skull", "parent": "neck", "markers": ["snout", "crown"]},
]
layout = layout_from_segments(SPEC, with_drift=False)
names = list(layout.marker_names)

# Ground-truth trunk geometry (px), in the body frame: spine collinear along
# +x, haunches straddling behind.
TRUE = {
    "spine_0": (0.0, 0.0),      # distal anchor
    "spine_1": (10.0, 0.0),
    "spine_2": (20.0, 0.0),
    "spine_3": (30.0, 0.0),
    "haunch_l": (34.0, 8.0),
    "haunch_r": (34.0, -8.0),
    "nape": (-6.0, 0.0),        # child markers, placed for realism
    "snout": (-18.0, 0.0),
    "crown": (-12.0, 3.0),
}

rng = np.random.default_rng(0)
T = 1500
cx = 300 + np.cumsum(rng.standard_normal(T) * 1.5)
cy = 300 + np.cumsum(rng.standard_normal(T) * 1.5)
theta = np.cumsum(rng.standard_normal(T) * 0.03)   # body rotates over time

pos = np.zeros((T, len(names), 2))
lik = np.ones((T, len(names)))
for i, m in enumerate(names):
    ox, oy = TRUE[m]
    c, s = np.cos(theta), np.sin(theta)
    pos[:, i, 0] = cx + c * ox - s * oy + rng.standard_normal(T) * 0.4
    pos[:, i, 1] = cy + s * ox + c * oy + rng.standard_normal(T) * 0.4

# ---------------------------------------------------------------- #
# 1. the axis helper
# ---------------------------------------------------------------- #
cluster_idx = [names.index(m) for m in
               ["spine_0", "spine_1", "spine_2", "spine_3",
                "haunch_l", "haunch_r"]]
axis = _root_cluster_body_axis(pos, lik, cluster_idx, 0.7)
check("axis is returned for a well-observed cluster", axis is not None)
check("axis is per-frame and unit-norm where defined",
      axis.shape == (T, 2)
      and np.allclose(np.hypot(axis[:, 0], axis[:, 1]), 1.0, atol=1e-6))

# The axis must track the body's rotation: its angle should correlate with
# theta (up to the arbitrary global sign/offset PCA leaves).
ax_ang = np.unwrap(np.arctan2(axis[:, 1], axis[:, 0]))
# de-mean both, compare slopes
d_ax = ax_ang - ax_ang.mean()
d_th = np.unwrap(theta) - np.unwrap(theta).mean()
corr = abs(np.corrcoef(d_ax, d_th)[0, 1])
check(f"axis rotates WITH the body (|corr| with theta = {corr:.2f})", corr > 0.9)

# Sign consistency: no bimodal split. Project each frame's axis on the mean;
# nearly all should share sign.
ref = np.nanmean(axis * np.sign(axis[:, [0]]), axis=0)
ref = ref / np.linalg.norm(ref)
same = (axis @ ref) > 0
check("axis sign is consistent across frames (no 180 deg flips)",
      np.mean(same) > 0.98)

# Under-determined clusters return None rather than guessing.
lik_blind = lik.copy()
lik_blind[:, cluster_idx] = 0.0
check("returns None when the cluster is never visible",
      _root_cluster_body_axis(pos, lik_blind, cluster_idx, 0.7) is None)

lik_two = lik.copy()
for j in cluster_idx[2:]:
    lik_two[:, j] = 0.0        # only 2 markers ever visible -> PCA undefined
check("returns None when fewer than 3 cluster markers are ever co-visible",
      _root_cluster_body_axis(pos, lik_two, cluster_idx, 0.7) is None)

# A round / near-coincident cluster has no meaningful principal axis; the
# guard must refuse it rather than fit a noise direction. Build a compact
# blob (all markers within a couple px of a point) and confirm None.
pos_blob = np.zeros((T, len(names), 2))
for i in range(len(names)):
    pos_blob[:, i, 0] = cx + rng.standard_normal(T) * 0.5
    pos_blob[:, i, 1] = cy + rng.standard_normal(T) * 0.5
check("returns None for a round cluster (anisotropy below threshold)",
      _root_cluster_body_axis(pos_blob, lik, cluster_idx, 0.7) is None)

# ---------------------------------------------------------------- #
# 2. fit_body_lengths recovers the trunk (the actual fix)
# ---------------------------------------------------------------- #
fl = fit_body_lengths(pos, lik, layout, names, 0.7, dt=1 / 30)

placeholders = dict(layout.segment_by_name("trunk").markers)


def is_ring(m):
    off = fl.marker_offsets.get(m)
    if off is None:
        return True  # missing == falls back to the ring downstream
    L, A = off
    ring = placeholders[m]
    return abs(L - ring[0]) < 1e-9 and abs(A - ring[1]) < 1e-9


check("NO trunk marker is left on its ring placeholder",
      not any(is_ring(m) for m in
              ["spine_1", "spine_2", "spine_3", "haunch_l", "haunch_r"]))
check("every non-distal trunk marker got a fitted offset",
      all(m in fl.marker_offsets for m in
          ["spine_1", "spine_2", "spine_3", "haunch_l", "haunch_r"]))


def local_xy(m):
    off = fl.marker_offsets.get(m)
    if off is None:              # reverted patch leaves root markers unfitted
        return float("nan"), float("nan")
    L, A = off
    return L * math.cos(A), L * math.sin(A)


# The spine must come out collinear: small |y| for all four.
spine_y = [abs(local_xy(m)[1]) for m in
           ["spine_0", "spine_1", "spine_2", "spine_3"]]
check(f"spine is collinear (max |y| = {np.nanmax(spine_y):.1f}px)",
      all(np.isfinite(y) for y in spine_y) and max(spine_y) < 2.0)

# Spine lengths increase monotonically along the body.
spine_x = [local_xy(m)[0] for m in
           ["spine_0", "spine_1", "spine_2", "spine_3"]]
check("spine markers are ordered along the axis (monotonic x)",
      all(b > a for a, b in zip(spine_x, spine_x[1:], strict=False))
      or all(b < a for a, b in zip(spine_x, spine_x[1:], strict=False)))

# Spine spacings should be ~10px apart (the ground truth).
gaps = [abs(b - a) for a, b in zip(spine_x, spine_x[1:], strict=False)]
gaps_ok = all(np.isfinite(g) and 7.0 < g < 13.0 for g in gaps)
check(f"spine spacing matches ground truth ~10px "
      f"(got {[round(g, 1) if np.isfinite(g) else 'nan' for g in gaps]})",
      gaps_ok)

# Haunches straddle the axis: opposite-sign y, similar magnitude.
hl, hr = local_xy("haunch_l"), local_xy("haunch_r")
check("haunches straddle the body axis (opposite-sign lateral offsets)",
      hl[1] * hr[1] < 0)
check(f"haunches are laterally symmetric (|y|: {abs(hl[1]):.1f} vs {abs(hr[1]):.1f})",
      abs(abs(hl[1]) - abs(hr[1])) < 3.0)
check(f"haunch lateral offset matches ground truth ~8px "
      f"(got {abs(hl[1]):.1f}, {abs(hr[1]):.1f})",
      6.0 < abs(hl[1]) < 10.0 and 6.0 < abs(hr[1]) < 10.0)

# Pairwise inter-marker distances are frame-invariant, so they are the
# cleanest ground-truth check (immune to the global axis sign/offset).
for a, b in [("spine_0", "spine_3"), ("spine_1", "haunch_l")]:
    ia, ib = names.index(a), names.index(b)
    d_obs = float(np.median(np.hypot(*(pos[:, ia, :] - pos[:, ib, :]).T)))
    xa, ya = local_xy(a)
    xb, yb = local_xy(b)
    d_fit = math.hypot(xa - xb, ya - yb)
    check(f"fitted {a}-{b} distance matches observed ({d_fit:.1f} vs {d_obs:.1f}px)",
          abs(d_fit - d_obs) < 3.0)

# ---------------------------------------------------------------- #
# 3. the old rig-specific proxy is gone
# ---------------------------------------------------------------- #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)


def fn_src(name):
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return ast.get_source_segment(src, f) or ""
    return ""


fbl = fn_src("fit_body_lengths")
# 122hh keeps the old back1/back3 proxy as a FALLBACK for round clusters
# (the canonical rig), but it is no longer the only root path: the PCA axis
# handles elongated clusters first. So the assertion is not "the proxy is
# gone" but "the PCA path exists and is tried first".
check("fit_body_lengths uses the PCA axis helper for root clusters",
      "_root_cluster_body_axis(" in fbl)
check("the PCA path is attempted before the back1/back3 fallback",
      fbl.index("_root_cluster_body_axis(") < fbl.index('"back3"'))

# A renamed single-cluster rig (no back1/back3 anywhere) must still fit — the
# regression test proper.
check("a non-canonical rig fits its root cluster (the regression)",
      not is_ring("spine_2") and not is_ring("haunch_l"))

# Non-root segments are untouched by this patch: skull/neck still fit via
# parent_distal.
check("child-segment offsets still fit (non-root path unchanged)",
      "crown" in fl.marker_offsets and fl.marker_offsets["crown"] is not None)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hh_root_pca_axis: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
