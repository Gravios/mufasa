"""Smoke test for patch 122hc — find the project; stop hand-writing whitelists.

122hb stopped `smooth_pose_v2` from assuming the built-in rat rig when
`layout is None`, and gave the CLI a `--config` so the layout could come from
the project. It was not enough: `main()` still *eagerly built the rig* when
`--config` was absent, so it never reached the new guard and the next real run
failed identically. A flag you have to know about is still a default for
everyone who doesn't.

122hc: the CLI walks up from the input path to the enclosing project.toml, and
the rig-fallback error stops blaming file discovery for what is a
"no project was found" problem.

Two bugs surfaced underneath, both masked until the layout resolved:
  * `--orient-drift-segments body,head` against a [skeleton]-derived tree,
    whose segments are named after distal markers and include no "body";
  * the warm-start rebuild of NoiseParamsV2, a hand-written whitelist that
    rotted three times and silently dropped const-accel's q_jerk_* fields.

Real tests: actual files, actual CLI, actual project trees.
"""
from __future__ import annotations

import dataclasses as _dc
import pathlib
import subprocess
import sys
import tempfile
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    CANONICAL_LAYOUT_ROLES,
    NoiseParamsV2,
    _validate_layout_against_data,
    layout_from_segments,
    segments_from_skeleton,
    standard_rat_layout,
)
from mufasa.project_layout import find_project_config  # noqa: E402

MARKERS = [
    "head_nose", "head_mid", "head_left", "head_right", "head_back",
    "back_T4", "back_T8", "back_L2", "back_L6", "back_V2",
    "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32",
]
EDGES = [
    ("head_nose", "head_mid"), ("head_left", "head_mid"),
    ("head_right", "head_mid"), ("head_mid", "head_back"),
    ("head_left", "head_nose"), ("head_right", "head_nose"),
    ("head_left", "head_right"), ("head_back", "back_T4"),
    ("head_left", "head_back"), ("head_right", "head_back"),
    ("back_T4", "back_T8"), ("back_T8", "back_L2"), ("back_L2", "back_L6"),
    ("back_L6", "back_V2"), ("back_V2", "tail_V6"), ("tail_V6", "tail_V18"),
    ("tail_V18", "tail_V32"), ("hip_left", "back_L6"),
    ("hip_right", "back_L6"), ("hip_left", "hip_right"),
    ("hip_left", "back_V2"), ("hip_right", "back_V2"),
    ("back_T4", "back_L2"), ("back_T8", "back_L6"),
]
# The rigid-cluster tree from docs — D=44, and the only source of segments
# actually named "body" and "head".
SEGMENTS_TOML = """
[[pose.segments]]
name    = "body"
markers = ["back_T8", "back_T4", "back_L2", "back_L6", "hip_left", "hip_right"]
[[pose.segments]]
name = "back_rear"
parent = "body"
markers = ["back_V2"]
[[pose.segments]]
name = "neck"
parent = "body"
markers = ["head_back"]
[[pose.segments]]
name = "head"
parent = "neck"
markers = ["head_mid", "head_nose", "head_left", "head_right"]
[[pose.segments]]
name = "tail_1"
parent = "back_rear"
markers = ["tail_V6"]
[[pose.segments]]
name = "tail_2"
parent = "tail_1"
markers = ["tail_V18"]
[[pose.segments]]
name = "tail_3"
parent = "tail_2"
markers = ["tail_V32"]
"""
SPEC = [
    {"name": "body", "markers": ["back_T8", "back_T4", "back_L2", "back_L6",
                                 "hip_left", "hip_right"]},
    {"name": "back_rear", "parent": "body", "markers": ["back_V2"]},
    {"name": "neck", "parent": "body", "markers": ["head_back"]},
    {"name": "head", "parent": "neck",
     "markers": ["head_mid", "head_nose", "head_left", "head_right"]},
    {"name": "tail_1", "parent": "back_rear", "markers": ["tail_V6"]},
    {"name": "tail_2", "parent": "tail_1", "markers": ["tail_V18"]},
    {"name": "tail_3", "parent": "tail_2", "markers": ["tail_V32"]},
]

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tmp = pathlib.Path(tempfile.mkdtemp())


def make_project(name: str, *, with_segments: bool) -> tuple[pathlib.Path, pathlib.Path]:
    root = tmp / name
    pose = root / "sources" / "pose"
    pose.mkdir(parents=True)
    ml = str(MARKERS).replace("'", '"')
    edges = ",\n  ".join(f'["{a}", "{b}"]' for a, b in EDGES)
    (root / "project.toml").write_text(
        f'project_layout_version = 1\n[project]\nproject_name = "{name}"\n'
        f"[pose_settings]\nbody_parts = {ml}\n"
        + (SEGMENTS_TOML if with_segments else "")
        + f"\n[skeleton]\nnodes = {ml}\nedges = [\n  {edges}\n]\n"
    )
    for stem in ("Cacna_87_f_wt_cort_25d_post", "Cacna_87_f_wt_cort_5d_post"):
        cols = {}
        for m in MARKERS:
            cols[f"{m}_x"] = np.cumsum(np.random.rand(50)) * 3
            cols[f"{m}_y"] = np.cumsum(np.random.rand(50)) * 3
            cols[f"{m}_p"] = np.ones(50)
        pd.DataFrame(cols).to_parquet(pose / f"{stem}.parquet")
    return root, pose


# ---------------------------------------------------------------- #
# 1. find_project_config
# ---------------------------------------------------------------- #
root_a, pose_a = make_project("strohA-kj", with_segments=True)
root_b, pose_b = make_project("otherproj", with_segments=True)

check("walks up from a data directory",
      find_project_config(str(pose_a)) == root_a / "project.toml")
check("walks up from a data file",
      find_project_config(str(pose_a / "Cacna_87_f_wt_cort_25d_post.parquet"))
      == root_a / "project.toml")
check("given the project.toml itself, returns it",
      find_project_config(str(root_a / "project.toml")) == root_a / "project.toml")
check("returns None outside any project", find_project_config(str(tmp)) is None)

deep = root_a / "a" / "b" / "c" / "d" / "e" / "f" / "g"
deep.mkdir(parents=True)
check("is depth-bounded, so a stray path can't adopt a distant project",
      find_project_config(str(deep)) is None)

# ---------------------------------------------------------------- #
# 2. the CLI, run for real
# ---------------------------------------------------------------- #
def run_cli(*extra: str, inputs: list[str] | None = None) -> tuple[int, str]:
    argv = [
        sys.executable, "-m", "mufasa.data_processors.kalman_pose_smoother_v2",
        *(inputs or [str(pose_a) + "/"]),
        "--output-dir", str(tmp / "out"), "--fps", "30",
        "--em-max-iter", "1", "--workers", "2", *extra,
    ]
    p = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(REPO), timeout=900,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
             "HOME": str(tmp), "QT_QPA_PLATFORM": "offscreen"},
    )
    return p.returncode, p.stdout + p.stderr


rc, out = run_cli("--verbose")
check("CLI announces the project it found", "[smoother-v2] Project:" in out
      and str(root_a / "project.toml") in out)
check("CLI never mentions the rig's marker names for this project",
      "back1" not in out and "tailbase" not in out)
check("CLI run without --config now succeeds", rc == 0)
check("pre-flight reports the matched fraction", "markers  : 15/15 matched" in out)
check("pre-flight reports state_dim", "state_dim D=" in out)
check("pre-flight reports projected memory", "GiB/worker peak" in out)

rc_two, out_two = run_cli(inputs=[str(pose_a) + "/", str(pose_b) + "/"])
check("CLI refuses inputs spanning two projects rather than guessing",
      rc_two == 1 and "more than one" in out_two)

# The user's real invocation: drift + orient-drift + const-accel on body,head.
rc_full, out_full = run_cli(
    "--with-drift", "--orient-drift-segments", "body,head",
    "--const-accel-segments", "body,head", "--verbose",
)
check("--with-drift + --orient-drift-segments + --const-accel-segments runs",
      rc_full == 0)
check("...and completes smoothing", "final smoothing done" in out_full)

# Same flags against a project with only [skeleton]: the segments genuinely
# don't exist, and the error must say what to do about it.
root_c, pose_c = make_project("skeleton-only", with_segments=False)
rc_sk, out_sk = run_cli(
    "--orient-drift-segments", "body,head", inputs=[str(pose_c) + "/"],
)
check("segment flags naming a nonexistent segment fail cleanly, not by traceback",
      rc_sk == 1 and "Traceback" not in out_sk)
check("...the error lists the segments that DO exist", "This layout's segments:" in out_sk)
check("...and points at [[pose.segments]] as the fix", "[[pose.segments]]" in out_sk)

# ---------------------------------------------------------------- #
# 3. the rig-fallback hint stops blaming file discovery
# ---------------------------------------------------------------- #
rig = standard_rat_layout()
full = list(rig.marker_names)
check("the rig is what CANONICAL_LAYOUT_ROLES names",
      set(full) == set(CANONICAL_LAYOUT_ROLES))

rig_hint = ""
try:
    _validate_layout_against_data(rig, MARKERS)
except ValueError as exc:
    rig_hint = str(exc)
check("rig-vs-project mismatch names the rig as the culprit",
      "BUILT-IN RAT RIG" in rig_hint)
check("...points at --config", "--config" in rig_hint)
check("...and does NOT send the user to backups/", "backups/" not in rig_hint)

partial_msg = ""
try:
    _validate_layout_against_data(rig, full[:-3])
except ValueError as exc:
    partial_msg = str(exc)
check("a genuine partial mismatch still points at file discovery",
      "backups/" in partial_msg and "BUILT-IN RAT RIG" not in partial_msg)

# ---------------------------------------------------------------- #
# 4. the warm-start NoiseParamsV2 rebuild
# ---------------------------------------------------------------- #
seg_layout = layout_from_segments(SPEC)
check("the docs' [[pose.segments]] block covers all 15 markers",
      sorted(seg_layout.marker_names) == sorted(MARKERS))
check("...yields D=44", seg_layout.state_dim == 44)
check("...and D=74 with drift",
      _dc.replace(seg_layout, with_drift=True).state_dim == 74)
check("a [skeleton]-derived tree names segments after distal markers, "
      "so it has no 'body' or 'head' to point --orient-drift-segments at",
      {s.name for s in segments_from_skeleton(MARKERS, EDGES)}
      .isdisjoint({"body", "head"}))

ca_layout = _dc.replace(
    seg_layout, with_drift=True,
    orientation_drift_segments=["body", "head"],
    const_accel_segments=["body", "head"],
)
params = NoiseParamsV2.default(ca_layout)
check("const-accel layout produces populated q_jerk_* fields",
      params.q_jerk_root_pos is not None
      and params.q_jerk_root_ori is not None
      and params.q_jerk_seg_ori is not None)

# Exactly the rebuild smooth_pose_v2 performs after warm-start.
_changes = {
    f.name: dict(value)
    for f in _dc.fields(params)
    if isinstance(value := getattr(params, f.name), dict)
}
_changes["sigma_marker"] = {m: 1.0 for m in ca_layout.marker_names}
rebuilt = _dc.replace(params, **_changes)

dropped = [
    f.name for f in _dc.fields(params)
    if getattr(params, f.name) is not None and getattr(rebuilt, f.name) is None
]
check("warm-start rebuild drops no populated field (q_jerk_* included)",
      dropped == [])
check("warm-start rebuild copies dicts rather than aliasing them",
      rebuilt.q_seg_ori is not params.q_seg_ori)
check("warm-start rebuild applies the new sigma",
      rebuilt.sigma_marker != params.sigma_marker)

src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("the whitelist rebuild is gone from smooth_pose_v2's EM",
      "q_theta_drift=(\n                dict(initial_params.q_theta_drift)" not in src)
check("...replaced by a copy-every-field replace",
      "_changes[\"sigma_marker\"] = sigma_warm" in src)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hc_project_discovery: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
