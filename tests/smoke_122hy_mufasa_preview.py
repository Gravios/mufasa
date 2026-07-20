"""Smoke test for patch 122hy — the mufasa-preview command.

mufasa-preview is a front-end over pose_video_overlay that pairs a video with
its pose files — from explicit paths or by searching a folder — and opens the
overlay viewer. Verifies the session-stem pairing (video <-> raw .fdlc.parquet
<-> smoothed .fdlc_smoothed_v2.<hash>.parquet), folder discovery, the overlay
argv construction, both CLI modes, the error/ambiguity paths, and that the
viewer is invoked in-process with the built argv. The entry point is registered
in pyproject.toml.
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import tempfile
import types
from pathlib import Path

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mufasa.tools import pose_preview as P  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ---- stem reduction (the pairing key) ----
check("video stem strips .mp4",
      P._video_stem("rec.mp4") == "rec")
check("video stem strips .avi/.mov/.mkv",
      P._video_stem("a.avi") == "a" and P._video_stem("b.mov") == "b"
      and P._video_stem("c.mkv") == "c")
check("raw .fdlc.parquet reduces to the video stem",
      P._pose_stem("rec.fdlc.parquet") == "rec")
check("plain .parquet reduces to the stem",
      P._pose_stem("rec.parquet") == "rec")
check("smoothed with a hash reduces to the stem",
      P._pose_stem("rec.fdlc_smoothed_v2.e62141d8fc4e0ba3.parquet") == "rec")
check("smoothed without a hash reduces to the stem",
      P._pose_stem("rec_smoothed_v2.parquet") == "rec")
check("the reported real filename pairs to its video stem",
      P._pose_stem(
          "K7_cam_20251124_1510_A10-0-051_reduced_negate.fdlc_smoothed_v2."
          "e62141d8fc4e0ba3.parquet")
      == P._video_stem("K7_cam_20251124_1510_A10-0-051_reduced_negate.mp4"))
check("_is_smoothed distinguishes smoothed from raw",
      P._is_smoothed("rec_smoothed_v2.parquet")
      and not P._is_smoothed("rec.fdlc.parquet"))


def _touch(folder, name):
    Path(os.path.join(folder, name)).write_bytes(b"")


# ---- folder discovery ----
with tempfile.TemporaryDirectory() as d:
    for n in ("rec.mp4", "rec.fdlc.parquet",
              "rec.fdlc_smoothed_v2.deadbeef.parquet"):
        _touch(d, n)
    groups = P._discover(d)
    check("discovery finds one group for a video with raw+smoothed",
          len(groups) == 1)
    g = groups[0]
    check("discovery pairs the video, raw, and smoothed by stem",
          g["stem"] == "rec"
          and g["video"].endswith("rec.mp4")
          and g["raw"].endswith("rec.fdlc.parquet")
          and g["smoothed"].endswith("rec.fdlc_smoothed_v2.deadbeef.parquet"))

# a video with no pose file is not returned
with tempfile.TemporaryDirectory() as d:
    _touch(d, "lonely.mp4")
    check("a video with no pose file is skipped", P._discover(d) == [])

# newest smoothed wins among several hashes
with tempfile.TemporaryDirectory() as d:
    _touch(d, "rec.mp4")
    old = os.path.join(d, "rec_smoothed_v2.aaaa.parquet")
    new = os.path.join(d, "rec_smoothed_v2.bbbb.parquet")
    Path(old).write_bytes(b"")
    Path(new).write_bytes(b"")
    os.utime(old, (1, 1))
    os.utime(new, (10 ** 9, 10 ** 9))
    picked = P._discover(d)[0]["smoothed"]
    check("newest smoothed file wins among parameter-hash variants",
          picked == new)

# ---- overlay argv construction ----
argv = P._build_overlay_argv("v.mp4", "s.parquet", "r.parquet", 0.2, 3, 100)
check("argv puts the video first",
      argv[0] == "v.mp4")
check("argv includes --smoothed / --raw and numeric opts",
      "--smoothed" in argv and "--raw" in argv
      and "--likelihood-threshold" in argv and "0.2" in argv
      and "--pose-offset" in argv and "--start-frame" in argv)
argv_min = P._build_overlay_argv("v.mp4", None, "r.parquet", 0.0, 0, 0)
check("argv omits absent options (no --smoothed, no zero opts)",
      "--smoothed" not in argv_min and "--likelihood-threshold" not in argv_min
      and "--pose-offset" not in argv_min)


def _run(argv, capture_out=False):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = P.main(argv)
    return rc, out.getvalue(), err.getvalue()


# ---- CLI: --print modes ----
with tempfile.TemporaryDirectory() as d:
    for n in ("rec.mp4", "rec.fdlc.parquet",
              "rec.fdlc_smoothed_v2.deadbeef.parquet"):
        _touch(d, n)
    v = os.path.join(d, "rec.mp4")

    rc, out, _ = _run(["--video", v, "--print"])
    check("--video --print emits an overlay command with the paired pose",
          rc == 0 and "pose_video_overlay" in out
          and "--smoothed" in out and "--raw" in out)

    rc, out, _ = _run(["--folder", d, "--print"])
    check("--folder --print emits the overlay command",
          rc == 0 and "pose_video_overlay" in out and "rec.mp4" in out)

# ---- CLI: error / ambiguity paths ----
with tempfile.TemporaryDirectory() as d:
    for n in ("A.mp4", "A.fdlc.parquet", "B.mp4", "B.fdlc.parquet"):
        _touch(d, n)
    rc, _, err = _run(["--folder", d, "--print"])
    check("ambiguous folder errors and lists matches",
          rc == 2 and "matched" in err and "A" in err and "B" in err)
    rc, out, _ = _run(["--folder", d, "--all", "--print"])
    check("--all previews every match",
          rc == 0 and out.count("pose_video_overlay") == 2)

with tempfile.TemporaryDirectory() as d:
    _touch(d, "novideo_pose.fdlc.parquet")
    rc, _, err = _run(["--folder", d, "--print"])
    check("folder with no matchable video errors", rc == 2)

with tempfile.TemporaryDirectory() as d:
    v = os.path.join(d, "x.mp4")
    Path(v).write_bytes(b"")
    rc, _, err = _run(["--video", v, "--print"])
    check("explicit video with no pose beside it errors clearly",
          rc == 2 and "no pose file" in err)

rc, _, err = _run(["--video", "/no/such/x.mp4", "--print"])
check("a missing video errors", rc == 2)

# --video and --folder are mutually exclusive
try:
    P._build_parser().parse_args(["--video", "a.mp4", "--folder", "/tmp"])
    check("--video and --folder are mutually exclusive", False)
except SystemExit:
    check("--video and --folder are mutually exclusive", True)

# ---- in-process overlay invocation (not --print) ----
_stub = types.ModuleType("mufasa.tools.pose_video_overlay")
_calls: list[list[str]] = []


def _fake_overlay_main(argv):
    _calls.append(argv)
    return 0


_stub.main = _fake_overlay_main
sys.modules["mufasa.tools.pose_video_overlay"] = _stub
try:
    with tempfile.TemporaryDirectory() as d:
        for n in ("rec.mp4", "rec.fdlc.parquet"):
            _touch(d, n)
        v = os.path.join(d, "rec.mp4")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = P.main(["--video", v])
        check("without --print, the overlay is invoked in-process",
              rc == 0 and len(_calls) == 1)
        check("the overlay receives the built argv (video + paired pose)",
              _calls and _calls[0][0] == v and "--raw" in _calls[0])
finally:
    del sys.modules["mufasa.tools.pose_video_overlay"]

# ---- entry point registered ----
pyproject = (REPO / "pyproject.toml").read_text()
check("mufasa-preview entry point is registered",
      "mufasa-preview" in pyproject
      and "mufasa.tools.pose_preview:main" in pyproject)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hy_mufasa_preview: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
