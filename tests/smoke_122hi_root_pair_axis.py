"""Smoke test for patch 122hi — root-cluster offsets from the widest chord.

122hh derived the root cluster's body frame from per-frame PCA gated on
anisotropy, assuming a trunk is elongated like a spine. On real top-down mouse
data it is not: the six dorsal markers project into a roughly ROUND 2D patch
(~40px across, eigenvalue ratio ~1.75). PCA of a round patch returns a noise
direction, the gate rejected ~90% of frames, the cluster fell back to the
placeholder ring, and the trunk kept bunching. 122hh fixed the (elongated)
tail but not the trunk.

122hi: a round patch still has a well-defined frame — the two markers farthest
apart on average define its widest chord, whose per-frame direction is stable
even when the cluster is round. No marker names, no elongation assumption. The
chord is signed, so there is no PCA sign ambiguity to repair.

Assertions are frame-INVARIANT (pairwise distances between fitted offsets match
observed and ground-truth inter-marker distances), because the absolute
orientation of the fitted frame is arbitrary and the filter absorbs a global
rotation into the segment's orientation state.
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


SPEC = [
    {"name": "trunk",
     "markers": ["m_a", "m_b", "m_c", "m_d", "haunch_l", "haunch_r"]},
    {"name": "neck", "parent": "trunk", "markers": ["nape"]},
    {"name": "skull", "parent": "neck", "markers": ["snout", "crown"]},
]
layout = layout_from_segments(SPEC, with_drift=False)
names = list(layout.marker_names)

TRUE = {
    "m_a": (0.0, 0.0),
    "m_b": (14.0, -3.0),
    "m_c": (10.0, 12.0),
    "m_d": (24.0, 8.0),
    "haunch_l": (30.0, -14.0),
    "haunch_r": (34.0, 16.0),
    "nape": (-10.0, 2.0),
    "snout": (-22.0, 1.0),
    "crown": (-16.0, 5.0),
}

rng = np.random.default_rng(0)
T = 1500
cx = 300 + np.cumsum(rng.standard_normal(T) * 1.5)
cy = 300 + np.cumsum(rng.standard_normal(T) * 1.5)
theta = np.cumsum(rng.standard_normal(T) * 0.03)

pos = np.zeros((T, len(names), 2))
lik = np.ones((T, len(names)))
for i, m in enumerate(names):
    ox, oy = TRUE[m]
    c, s = np.cos(theta), np.sin(theta)
    pos[:, i, 0] = cx + c * ox - s * oy + rng.standard_normal(T) * 0.4
    pos[:, i, 1] = cy + s * ox + c * oy + rng.standard_normal(T) * 0.4

cluster_idx = [names.index(m) for m in
               ["m_a", "m_b", "m_c", "m_d", "haunch_l", "haunch_r"]]

pts_mean = pos[:, cluster_idx, :].mean(axis=0)
c0 = pts_mean - pts_mean.mean(axis=0)
evals = np.linalg.eigvalsh(c0.T @ c0)
aniso = evals[-1] / max(evals[0], 1e-9)
check(f"the synthetic trunk is genuinely round (anisotropy {aniso:.1f} < 4)",
      aniso < 4.0)

axis = _root_cluster_body_axis(pos, lik, cluster_idx, 0.7)
check("a round patch still yields a frame (PCA could not)", axis is not None)

# The frame must come from the WIDEST chord, not just any pair. Compute the
# widest and narrowest pairs by mean separation; the axis must align with the
# widest (a degenerate 'closest pair' choice would align with the narrowest
# and be far noisier).
_mean = pos[:, cluster_idx, :].mean(axis=0)
_wide = _narrow = None
_wd, _nd = -1.0, 1e18
for _i in range(len(cluster_idx)):
    for _j in range(_i + 1, len(cluster_idx)):
        _d = math.hypot(_mean[_i, 0] - _mean[_j, 0], _mean[_i, 1] - _mean[_j, 1])
        if _d > _wd:
            _wd, _wide = _d, (_i, _j)
        if _d < _nd:
            _nd, _narrow = _d, (_i, _j)
_wv = _mean[_wide[1]] - _mean[_wide[0]]
_wv = _wv / np.linalg.norm(_wv)
_median_axis = np.array([np.nanmedian(axis[np.isfinite(axis[:, 0]), 0]),
                         np.nanmedian(axis[np.isfinite(axis[:, 1]), 1])])
_median_axis = _median_axis / (np.linalg.norm(_median_axis) + 1e-12)
_align_wide = abs(float(_median_axis @ _wv))
check(f"frame is derived from the WIDEST chord, not a short one "
      f"(|align| {_align_wide:.2f}, widest {_wd:.0f}px vs narrowest {_nd:.0f}px)",
      _align_wide > 0.9)
check("axis is per-frame, unit-norm where defined",
      axis.shape == (T, 2)
      and np.allclose(np.hypot(axis[np.isfinite(axis[:, 0]), 0],
                               axis[np.isfinite(axis[:, 1]), 1]), 1.0, atol=1e-6))

defined = np.isfinite(axis[:, 0])
# Sign stability for a SIGNED chord is the absence of sudden 180 deg jumps
# frame to frame — NOT confinement to a half-plane, since the body may rotate
# through a wide arc over the session (a PCA eigenvector, by contrast, would
# flip). Assert no large discontinuities in the unwrapped axis angle.
_ang = np.unwrap(np.arctan2(axis[defined, 1], axis[defined, 0]))
_jumps = np.abs(np.diff(_ang))
check("axis sign is stable frame-to-frame (no 180 deg chord flips)",
      _jumps.size > 0 and np.max(_jumps) < math.radians(90))

ax_ang = np.unwrap(np.arctan2(axis[defined, 1], axis[defined, 0]))
th_def = np.unwrap(theta)[defined]
corr = abs(np.corrcoef(ax_ang - ax_ang.mean(), th_def - th_def.mean())[0, 1])
check(f"axis rotates with the body (|corr| = {corr:.2f})", corr > 0.9)

lik_blind = lik.copy()
lik_blind[:, cluster_idx] = 0.0
check("returns None when the cluster is never visible",
      _root_cluster_body_axis(pos, lik_blind, cluster_idx, 0.7) is None)


# ---------------------------------------------------------------- #
# 1b. widest-chord selection, controlled so wrong choice is detectable
# ---------------------------------------------------------------- #
# Four markers: a long horizontal pair (0,0)-(40,0) and a short vertical pair
# (20,-3)-(20,3). Farthest = horizontal (axis ~ +x); closest = vertical
# (axis ~ +y). Choosing wrong rotates the frame 90 deg.
lay2 = layout_from_segments(
    [{"name": "t2", "markers": ["w0", "w1", "n0", "n1"]}], with_drift=False)
nm2 = list(lay2.marker_names)
G = {"w0": (0.0, 0.0), "w1": (40.0, 0.0), "n0": (20.0, -3.0), "n1": (20.0, 3.0)}
T2 = 400
p2 = np.zeros((T2, len(nm2), 2))
l2 = np.ones((T2, len(nm2)))
for i, m in enumerate(nm2):
    p2[:, i, 0] = 100 + G[m][0] + rng.standard_normal(T2) * 0.2
    p2[:, i, 1] = 100 + G[m][1] + rng.standard_normal(T2) * 0.2
ci2 = [nm2.index(m) for m in ["w0", "w1", "n0", "n1"]]
ax2 = _root_cluster_body_axis(p2, l2, ci2, 0.7)
med2 = np.array([np.nanmedian(ax2[np.isfinite(ax2[:, 0]), 0]),
                 np.nanmedian(ax2[np.isfinite(ax2[:, 1]), 1])])
med2 = med2 / (np.linalg.norm(med2) + 1e-12)
# widest chord is horizontal -> axis should be ~ (1,0), |x|>>|y|
check(f"picks the widest (horizontal) chord, not the short vertical one "
      f"(axis {med2[0]:.2f},{med2[1]:.2f})",
      abs(med2[0]) > 0.9 and abs(med2[1]) < 0.4)

fl = fit_body_lengths(pos, lik, layout, names, 0.7, dt=1 / 30)
placeholders = dict(layout.segment_by_name("trunk").markers)


def is_ring(m):
    off = fl.marker_offsets.get(m)
    if off is None:
        return True
    L, A = off
    r = placeholders[m]
    return abs(L - r[0]) < 1e-9 and abs(A - r[1]) < 1e-9


check("NO trunk marker is left on its ring placeholder",
      not any(is_ring(m) for m in
              ["m_b", "m_c", "m_d", "haunch_l", "haunch_r"]))
check("every non-distal trunk marker got a fitted offset",
      all(m in fl.marker_offsets for m in
          ["m_b", "m_c", "m_d", "haunch_l", "haunch_r"]))


def local_xy(m):
    off = fl.marker_offsets.get(m)
    if off is None:
        return float("nan"), float("nan")
    L, A = off
    return L * math.cos(A), L * math.sin(A)


pairs = [("m_a", "m_d"), ("m_b", "m_c"), ("haunch_l", "haunch_r"),
         ("m_a", "haunch_l"), ("m_b", "haunch_r"), ("m_c", "m_d")]
worst = 0.0
for a, b in pairs:
    ia, ib = names.index(a), names.index(b)
    d_obs = float(np.median(np.hypot(pos[:, ia, 0] - pos[:, ib, 0],
                                     pos[:, ia, 1] - pos[:, ib, 1])))
    xa, ya = local_xy(a)
    xb, yb = local_xy(b)
    worst = max(worst, abs(math.hypot(xa - xb, ya - yb) - d_obs))
check(f"all fitted pairwise distances match observed (worst {worst:.1f}px)",
      worst < 3.0)

gt_worst = 0.0
for a, b in pairs:
    gt = math.hypot(TRUE[a][0] - TRUE[b][0], TRUE[a][1] - TRUE[b][1])
    xa, ya = local_xy(a)
    xb, yb = local_xy(b)
    gt_worst = max(gt_worst, abs(math.hypot(xa - xb, ya - yb) - gt))
check(f"fitted shape matches ground-truth distances (worst {gt_worst:.1f}px)",
      gt_worst < 3.0)

src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)


def fn_src(name):
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return ast.get_source_segment(src, f) or ""
    return ""


helper = fn_src("_root_cluster_body_axis")
check("the anisotropy gate is gone (it rejected real round trunks)",
      "_ROOT_AXIS_MIN_ANISOTROPY" not in src)
check("the helper no longer relies on per-frame PCA eigenvectors",
      "np.linalg.eigh" not in helper)
check("the helper uses the most-separated marker pair",
      "best_pair" in helper or "best_d" in helper)

fbl = fn_src("fit_body_lengths")
check("fit_body_lengths still calls the root-cluster axis helper",
      "_root_cluster_body_axis(" in fbl)
check("the back1/back3 fallback remains for the canonical round cluster",
      '"back3"' in fbl)

check("child-segment offsets still fit via parent_distal (non-root path)",
      "crown" in fl.marker_offsets and fl.marker_offsets["crown"] is not None)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hi_root_pair_axis: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
