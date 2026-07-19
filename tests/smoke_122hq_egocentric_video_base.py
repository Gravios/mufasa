"""Smoke test for patch 122hq — egocentric alignment video lookup vs pose suffix.

Egocentric alignment finds the source video for each smoothed pose file by name.
The Kalman v2 smoother writes <video>_smoothed_v2.parquet and, since the
parameter hash landed (122hm/hn), <video>_smoothed_v2.<hash>.parquet — so the
pose stem carries _smoothed_v2 and a trailing .<hash> that the source video
filename does not. The aligner used the raw pose stem as the video name, so the
lookup failed with "could not find a video file representing
Cacna_..._smoothed_v2.<hash>".

122hq adds _video_base_from_pose_name(), which strips those smoother suffixes
to recover the video base, used ONLY for the video lookup (outputs keep the
full pose-derived name so they trace to the specific smoothed input). Critically
it must not truncate a legitimately dotted video name that lacks the smoother
marker.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ------------------------------------------------------------------ #
# extract the helper in isolation (the module imports heavy deps —
# h5py etc. — absent in the sandbox, so compile just the function)
# ------------------------------------------------------------------ #
mod = REPO / "mufasa/data_processors/egocentric_aligner.py"
src = mod.read_text()
tree = ast.parse(src)
fn = next((n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)
           and n.name == "_video_base_from_pose_name"), None)
check("_video_base_from_pose_name is defined", fn is not None)

ns: dict = {"os": os}
exec(compile(ast.get_source_segment(src, fn), "<fn>", "exec"), ns)
base = ns["_video_base_from_pose_name"]

# ------------------------------------------------------------------ #
# behaviour
# ------------------------------------------------------------------ #
# the user's exact failing case: _smoothed_v2 + 16-char hash
check("strips _smoothed_v2 and a 16-char hash (the reported case)",
      base("Cacna_51_f_wt_cort_16d_post_smoothed_v2.5e9edbb387cfed83")
      == "Cacna_51_f_wt_cort_16d_post")
# 8-char hash (122hm's original width)
check("strips _smoothed_v2 and an 8-char hash",
      base("Cacna_51_f_wt_cort_16d_post_smoothed_v2.cbf66f37")
      == "Cacna_51_f_wt_cort_16d_post")
# legacy: _smoothed_v2 with no hash
check("strips a bare _smoothed_v2 (legacy, no hash)",
      base("Cacna_51_f_wt_cort_16d_post_smoothed_v2")
      == "Cacna_51_f_wt_cort_16d_post")
# a plain video stem is unchanged
check("leaves a plain video stem unchanged",
      base("Cacna_51_f_wt_cort_16d_post")
      == "Cacna_51_f_wt_cort_16d_post")
check("leaves a simple name unchanged", base("Video1") == "Video1")

# CRITICAL false-positive guards: don't truncate legitimately dotted names
check("does NOT strip a dotted video name lacking the smoother marker",
      base("Session_2026.01.15_animalA") == "Session_2026.01.15_animalA")
check("does NOT strip a hex-looking extension without _smoothed_v2",
      base("recording.abcdef") == "recording.abcdef")
check("does NOT strip a non-hex suffix after _smoothed_v2",
      base("Cacna_51_smoothed_v2.mp4bak") == "Cacna_51_smoothed_v2.mp4bak")
check("does NOT strip a hash-like token that isn't a dotted extension",
      base("animal_deadbeef") == "animal_deadbeef")
# idempotent: stripping an already-clean base leaves it
check("idempotent on an already-clean base",
      base(base("Cacna_51_f_wt_cort_16d_post_smoothed_v2.cbf66f37"))
      == "Cacna_51_f_wt_cort_16d_post")

# ------------------------------------------------------------------ #
# wiring: the helper is used for the video lookups, not the outputs
# ------------------------------------------------------------------ #
# all three find_video_of_file calls must use the stripped base
check("pre-check lookup uses the stripped base",
      "_video_base_from_pose_name(\n                        get_fn_ext(file_path)[1]\n                    )" in src
      or "_video_base_from_pose_name(get_fn_ext(file_path)[1])" in src)
check("run() derives self.video_base from the pose name",
      "self.video_base = _video_base_from_pose_name(self.video_name)" in src)
check("run() video lookups use self.video_base",
      src.count("filename=self.video_base") == 2)
# outputs keep the full pose-derived name (video_name), so they still trace to
# the specific smoothed input
check("output save paths still use the full video_name",
      "f'{self.video_name}.{file_type}'" in src
      and "f'{self.video_name}.mp4'" in src)
# no remaining lookup uses the un-stripped video_name
check("no find_video_of_file still passes the raw video_name",
      "filename=self.video_name" not in src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hq_egocentric_video_base: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
