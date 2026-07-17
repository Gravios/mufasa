"""Smoke test for patch 122hd — the segment-flag error stops guessing.

122hc turned a raw ``BodyLayout.__post_init__`` traceback into a readable
message. The message was readable and wrong. Against a project that declared
its own ``[[pose.segments]]`` and typed ``--orient-drift-segments body,head``,
it said:

    ... segment(s) that don't exist in this layout: ['body'].
    This layout's segments: ['back', 'back_rear', 'head', 'neck', ...]
    A layout derived from [skeleton] names every segment after its distal
    marker, so there is no 'body' or 'head' to point at. Declare the tree you
    mean with a [[pose.segments]] block in project.toml ...

Four false statements in three lines: the layout was not derived from
[skeleton]; there plainly *was* a 'head' (it is in the list the message itself
printed); the fix suggested was a thing the user had already done; and the
flag was printed as ``--orientation-drift-segments``, which is not the flag.
The single real fault — 'body' should be 'back' — was the one thing it never
said.

122hd: state what is missing, list what exists, and explain only where there
is a signal that can be checked against the object in hand.

Real tests: project trees on disk, the CLI run as a subprocess, assertions on
its actual output.
"""
from __future__ import annotations

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
    standard_rat_layout,
)

MARKERS = [
    "head_nose", "head_mid", "head_left", "head_right", "head_back",
    "back_T4", "back_T8", "back_L2", "back_L6", "back_V2",
    "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32",
]
EDGES = [
    ("head_nose", "head_mid"), ("head_left", "head_mid"),
    ("head_right", "head_mid"), ("head_mid", "head_back"),
    ("head_back", "back_T4"), ("back_T4", "back_T8"), ("back_T8", "back_L2"),
    ("back_L2", "back_L6"), ("back_L6", "back_V2"), ("back_V2", "tail_V6"),
    ("tail_V6", "tail_V18"), ("tail_V18", "tail_V32"),
    ("hip_left", "back_L6"), ("hip_right", "back_L6"),
]
# Root deliberately named "back", not "body" — this is the user's tree, and
# the mismatch with the rig's "body" is the whole subject of the patch.
BLOCK = """
[[pose.segments]]
name    = "back"
markers = ["back_T8", "back_T4", "back_L2", "back_L6", "hip_left", "hip_right"]
[[pose.segments]]
name    = "back_rear"
parent  = "back"
markers = ["back_V2"]
[[pose.segments]]
name    = "neck"
parent  = "back"
markers = ["head_back"]
[[pose.segments]]
name    = "head"
parent  = "neck"
markers = ["head_mid", "head_nose", "head_left", "head_right"]
[[pose.segments]]
name    = "tail_1"
parent  = "back_rear"
markers = ["tail_V6"]
[[pose.segments]]
name    = "tail_2"
parent  = "tail_1"
markers = ["tail_V18"]
[[pose.segments]]
name    = "tail_3"
parent  = "tail_2"
markers = ["tail_V32"]
"""

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tmp = pathlib.Path(tempfile.mkdtemp())


def make_project(name: str, block: str) -> pathlib.Path:
    root = tmp / name
    pose = root / "sources" / "pose"
    pose.mkdir(parents=True)
    ml = str(MARKERS).replace("'", '"')
    edges = ",\n  ".join(f'["{a}", "{b}"]' for a, b in EDGES)
    (root / "project.toml").write_text(
        f'project_layout_version = 1\n[project]\nproject_name = "{name}"\n'
        f"[pose_settings]\nbody_parts = {ml}\n" + block
        + f"\n[skeleton]\nnodes = {ml}\nedges = [\n  {edges}\n]\n"
    )
    cols = {}
    for m in MARKERS:
        cols[f"{m}_x"] = np.cumsum(np.random.rand(40)) * 3
        cols[f"{m}_y"] = np.cumsum(np.random.rand(40)) * 3
        cols[f"{m}_p"] = np.ones(40)
    pd.DataFrame(cols).to_parquet(pose / "s1.parquet")
    return pose


declared = make_project("declared", BLOCK)
derived = make_project("derived", "")


def run(pose: pathlib.Path, *extra: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m",
         "mufasa.data_processors.kalman_pose_smoother_v2", str(pose) + "/",
         "--output-dir", str(tmp / "out"), "--fps", "30",
         "--em-max-iter", "1", "--workers", "2", *extra],
        capture_output=True, text=True, cwd=str(REPO), timeout=900,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
             "HOME": str(tmp), "QT_QPA_PLATFORM": "offscreen"},
    )
    return p.returncode, p.stdout + p.stderr


# ---------------------------------------------------------------- #
# 1. The reported case: [[pose.segments]] declared, "body" typed.
# ---------------------------------------------------------------- #
rc, out = run(declared, "--with-drift", "--orient-drift-segments", "body,head",
              "--const-accel-segments", "body,head")
check("declared-tree + bad segment name still fails cleanly", rc == 1)
check("no traceback", "Traceback" not in out)

check("only the ACTUALLY missing segment is named",
      "don't exist in this layout: ['body']" in out)
check("...'head' is not claimed missing when 'head' exists",
      "'body', 'head'" not in out)
check("the message prints the real flag name",
      "--orient-drift-segments names" in out
      and "--orientation-drift-segments" not in out)
check("the layout's segments are listed", "'back', 'back_rear', 'head'" in out)
check("the root segment is named outright",
      "Its root segment is: 'back'" in out)
check("'body' is identified as the built-in rig's root name",
      "built-in rat rig's name for its ROOT segment" in out)
check("...and the actual root is offered as the fix",
      "This layout's root is 'back' — try that instead." in out)
check("it does NOT claim a [skeleton]-derived tree",
      "derived from [skeleton]" not in out)
check("it does NOT tell the user to add a block they already have",
      "Declare the tree you mean" not in out)
check("it points at the declared names instead",
      "declared by [[pose.segments]] in project.toml" in out)

# ---------------------------------------------------------------- #
# 2. The contrasting case: no [[pose.segments]], tree from [skeleton].
# ---------------------------------------------------------------- #
rc_d, out_d = run(derived, "--orient-drift-segments", "body,head")
check("skeleton-derived tree + rig names still fails", rc_d == 1)
check("...both names are reported missing there",
      "don't exist in this layout: ['body', 'head']" in out_d)
check("...the [skeleton] explanation IS given when it's true",
      "derived from [skeleton]" in out_d)
check("...and [[pose.segments]] IS recommended when absent",
      "Declare the tree you mean" in out_d)
check("...'head' gets a near-match suggestion against marker-named segments",
      "did you mean 'head_mid'" in out_d)

# ---------------------------------------------------------------- #
# 3. The corrected command runs.
# ---------------------------------------------------------------- #
rc_ok, out_ok = run(declared, "--with-drift",
                    "--orient-drift-segments", "back,head",
                    "--const-accel-segments", "back,head", "--verbose")
check("the corrected invocation (back,head) runs to completion", rc_ok == 0)
check("...and reports D=82 for drift + orient-drift + const-accel",
      "state_dim D=82" in out_ok)
check("...with all 15 markers matched", "markers  : 15/15 matched" in out_ok)

# ---------------------------------------------------------------- #
# 4. Suggestions must not be invented.
# ---------------------------------------------------------------- #
rc_t, out_t = run(declared, "--orient-drift-segments", "trunk")
check("an unrelated name fails", rc_t == 1)
check("...without a nonsense near-match ('trunk' is not 'neck')",
      "did you mean" not in out_t)
check("...but still names the root as a starting point",
      "Its root segment is: 'back'" in out_t)

# ---------------------------------------------------------------- #
# 5. The rig is the source of 'body' — pin it, since the message says so.
# ---------------------------------------------------------------- #
rig = standard_rat_layout()
check("the built-in rig's root really is named 'body'",
      rig.root_segment.name == "body")
check("...and 'head' really is one of its segments",
      "head" in {s.name for s in rig.segments})

src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("the [skeleton] claim is gated on a checkable property, not provenance",
      "if set(available) <= set(lay.marker_names):" in src)
check("near-match cutoff stays strict enough to stay useful",
      "cutoff=0.6" in src)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hd_segment_flag_errors: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
