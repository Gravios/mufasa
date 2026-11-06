"""Smoke test for patch 122jm — animation: number of points, sampling
method, and a session-coloured intro.

User request: "For the animated tsne I would like to control the number of
samples and the sampling method. The first 40 frames should show the tsne
colored by session and then transition to the normal color map previously
used."

mufasa-landscape-animate: --sampling random | balanced | even (with
--n-points); --intro-frames N (default 40) frames of the t-SNE map coloured
by session, then --intro-fade frames easing to --color. The Animate… popup
gets Points in animation, Sampling and Intro frames.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ls", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"])
fam = fd / "frame_by_frame_body_part_movements_mm"
for d in (vd, fam):
    d.mkdir(parents=True, exist_ok=True)
A = "Animal_1 movement a (mm)_FEATURE_SUBSET"
B = "Animal_1 movement b (mm)_FEATURE_SUBSET"
C = "Animal_1 movement c (mm)_FEATURE_SUBSET"
ANG = "Animal_1 heading angle (degrees)_FEATURE_SUBSET"
OTHER = "Animal_1 unselected (mm)_FEATURE_SUBSET"
N = 1500
r = np.random.default_rng(3)
for v in ("v1", "v2"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    g = r.integers(0, 2, N)
    pd.DataFrame({
        A: r.normal(g * 4, 0.6),
        B: r.normal(g * -3, 0.7),
        C: r.normal(0, 1, N),
        ANG: r.uniform(0, 360, N),
        OTHER: r.normal(0, 1, N),
    }).to_parquet(fam / f"{v}.parquet")
from mufasa.tools import landscape_animate as la  # noqa: E402

# ---------------------------------------------------------------- sampling
rng = np.random.default_rng(0)
sess = np.array(["s1"] * 900 + ["s2"] * 100)
xy = np.r_[rng.normal(0, 0.1, (900, 2)), rng.uniform(-5, 5, (100, 2))]
cand = np.arange(1000)
r_ = la.sample_points(200, cand, sess, xy, "random", np.random.default_rng(1))
b_ = la.sample_points(200, cand, sess, xy, "balanced", np.random.default_rng(1))
e_ = la.sample_points(200, cand, sess, xy, "even", np.random.default_rng(1))
check("sampling methods", la.SAMPLING == ("random", "balanced", "even"))
check("each returns n distinct sorted indices",
      all(len(set(v)) == 200 and (np.diff(v) > 0).all() for v in (r_, b_, e_)))
check("random follows the data (≈ 90 % from the big session)",
      150 <= (sess[r_] == "s1").sum() <= 195)
check("balanced: the same number per session", (sess[b_] == "s1").sum() == 100)
dense = np.hypot(*xy.T) < 0.5
check("even: sparse regions of the map kept (dense cluster no longer dominates)",
      dense[e_].mean() < 0.55 < 0.8 < dense[r_].mean()
      and dense[la.sample_points(120, cand, sess, xy, "even",
                                 np.random.default_rng(1))].mean() < 0.25)
check("balanced: a small session gives all it has, rest shared",
      (sess[la.sample_points(300, cand, sess, xy, "balanced",
                             np.random.default_rng(2))] == "s2").sum() == 100)
check("asking for more than available returns all",
      len(la.sample_points(5000, cand, sess, xy, "even", rng)) == 1000)
check("restricted to the candidates",
      set(la.sample_points(50, cand[::2], sess, xy, "balanced", rng)) <= set(cand[::2]))

# ---------------------------------------------------------------- intro sequence
s0, s1 = np.zeros((3, 3)), np.ones((3, 3))
seq = la.intro(40, 20, s0, s1, "by session")
check("intro: 40 session-coloured frames, then 20 to change",
      len(seq) == 60 and all(np.array_equal(c, s0) and al == 1.0 for c, _, al in seq[:40]))
check("intro: colours ease to the main colouring, subtitle fades",
      np.allclose(seq[-1][0], s1) and seq[-1][2] == 0.0
      and 0 < seq[50][0][0, 0] < 1)
sc = la.session_colors(pd.Series([f"s{i}" for i in range(30)]))
check("more than 20 sessions: still distinct colours",
      len({tuple(np.round(c, 3)) for c in sc}) == 30)

# ---------------------------------------------------------------- end to end
from mufasa.tools import feature_landscape as fl  # noqa: E402

out = tmp / "run"
cf = tmp / "c.txt"
cf.write_text(f"{A}\n{B}\n{C}\n")
fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
         "--out", str(out), "--tsne-sample", "600", "--frames-per-session", "800",
         "--n-jobs", "1", "--perplexity", "20"])
calls = []
orig = la.Renderer.frame


def spy(self, i, j, t, colors=None, subtitle="", subtitle_alpha=0.0):  # noqa: ANN001
    calls.append((i, j, None if colors is None else colors.copy(), subtitle, subtitle_alpha))
    return orig(self, i, j, t, colors=colors, subtitle=subtitle, subtitle_alpha=subtitle_alpha)


la.Renderer.frame = spy
import contextlib  # noqa: E402
import io  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    la.main([str(out), "--pair", A, B, "--through-tsne", "--size", "120", "--hold", "0.2",
             "--transition", "0.2", "--fps", "5", "--n-points", "240", "--sampling",
             "balanced", "--intro-frames", "6", "--intro-fade", "4", "--out", str(out / "a.gif")])
log = buf.getvalue()
emb = pd.read_parquet(out / "tsne.parquet")
check("CLI: n points and sampling used", "240 point(s), sampling: balanced" in log)
check("CLI: intro reported", "intro: 6 frame(s) by session, 4 to change colour" in log)
intro_calls = [c for c in calls if c[2] is not None]
check("first 6 frames: t-SNE map coloured by session, labelled",
      len(intro_calls) == 10 and all(c[0] == 0 and c[1] == 0 for c in intro_calls)
      and len({tuple(r) for r in np.round(intro_calls[0][2], 4)}) == 2
      and "session" in intro_calls[0][3])
check("then the colours change to the t-SNE colouring",
      not np.allclose(intro_calls[0][2], intro_calls[-1][2])
      and calls[len(intro_calls)][2] is None)
calls.clear()
with contextlib.redirect_stdout(io.StringIO()):
    la.main([str(out), "--pair", A, B, "--size", "120", "--hold", "0.2",
             "--transition", "0.2", "--fps", "5", "--n-points", "200", "--color", "session",
             "--out", str(out / "b.gif")])
check("no intro when already coloured by session", all(c[2] is None for c in calls))
la.Renderer.frame = orig

# ---------------------------------------------------------------- popup and argv
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402

d = lr.FeatureNamesDialog(None, cfg, [A, B, C], "animate")
o = d.options()
check("popup: points, sampling and intro frames (defaults 4000 / random / 40)",
      o["n_points"] == 4000 and o["sampling"] == "random" and o["intro_frames"] == 40
      and [d.sampling.itemText(i) for i in range(3)] == list(la.SAMPLING))
d.n_points.setValue(1500)
d.sampling.setCurrentText("even")
d.intro_frames.setValue(0)
o = d.options()
argv = lr.animate_argv(out, [(A, B)], out / "x.gif", None, theme="dark", mp4=False,
                       color="tsne", n_points=o["n_points"], sampling=o["sampling"],
                       intro_frames=o["intro_frames"])
check("argv carries them",
      argv[argv.index("--n-points") + 1] == "1500" and argv[argv.index("--sampling") + 1]
      == "even" and argv[argv.index("--intro-frames") + 1] == "0")
check("landscape popup has none of them", "sampling" not in
      lr.FeatureNamesDialog(None, cfg, [A, B], "landscape").options())

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jm_animate_sampling_and_session_intro: {len(checks) - len(failed)}/"
      f"{len(checks)} checks passed")
sys.exit(1 if failed else 0)
