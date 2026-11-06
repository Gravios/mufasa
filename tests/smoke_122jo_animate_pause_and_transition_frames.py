"""Smoke test for patch 122jo — animation pause and transition in frames.

User request: "I would also like to set the transition frame count and the
pause count at each feature".

mufasa-landscape-animate: --transition-frames (frames per move),
--hold-frames (pause on each feature space; 0 = none) and
--tsne-hold-frames (pause on the t-SNE map; default as --hold-frames); the
seconds options stay for the rest. The Animate… popup has the three spins
(defaults 30 / 30 / 40 = the previous 1.5 s / 2 s at 20 fps).
"""
from __future__ import annotations

import contextlib
import io
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

# ---------------------------------------------------------------- timeline
tl = la.timeline(3, 20, 1.5, 2.0, holds=[5, 7, 5], n_transition=4)
holds = [sum(1 for i, j, _ in tl if i == j == k) for k in range(3)]
moves = [sum(1 for i, j, _ in tl if (i, j) == (k, k + 1)) for k in range(2)]
check("timeline: pause per space in frames", holds == [5, 7, 5])
check("timeline: transition frames per move", moves == [4, 4] and len(tl) == 17 + 8)
check("timeline: pause 0 allowed",
      sum(1 for i, j, _ in la.timeline(2, 20, 1, 1, holds=[0, 0], n_transition=3)
          if i == j) == 0)
check("timeline: seconds as before without overrides",
      len(la.timeline(2, 20, 1.5, 2.0)) == 30 * 2 + 40)
check("timeline: transition eases from 0 to 1",
      [round(t, 3) for i, j, t in tl if (i, j) == (0, 1)][::3] == [0.0, 1.0])

# ---------------------------------------------------------------- CLI
from mufasa.tools import feature_landscape as fl  # noqa: E402

out = tmp / "run"
cf = tmp / "c.txt"
cf.write_text(f"{A}\n{B}\n{C}\n")
with contextlib.redirect_stdout(io.StringIO()):
    fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
             "--out", str(out), "--tsne-sample", "400", "--frames-per-session", "600",
             "--n-jobs", "1", "--perplexity", "20"])
calls = []
orig = la.Renderer.frame


def spy(self, i, j, t, colors=None, subtitle="", subtitle_alpha=0.0):  # noqa: ANN001
    calls.append((i, j, t))
    return orig(self, i, j, t, colors=colors, subtitle=subtitle, subtitle_alpha=subtitle_alpha)


la.Renderer.frame = spy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    la.main([str(out), "--pair", A, B, "--pair", B, C, "--through-tsne", "--size", "100",
             "--n-points", "150", "--intro-frames", "0", "--hold-frames", "6",
             "--tsne-hold-frames", "3", "--transition-frames", "5",
             "--out", str(out / "a.gif")])
la.Renderer.frame = orig
# spaces: tsne, p1, tsne, p2, tsne
per = [sum(1 for i, j, _ in calls if i == j == k) for k in range(5)]
mv = [sum(1 for i, j, _ in calls if (i, j) == (k, k + 1)) for k in range(4)]
check("CLI: 6 frames on each feature space, 3 on t-SNE", per == [3, 6, 3, 6, 3])
check("CLI: 5 frames per transition", mv == [5, 5, 5, 5])
check("CLI: timing reported",
      "pause 6 frame(s) on each feature space, 3 on t-SNE; 5 frame(s) per transition"
      in buf.getvalue())
calls.clear()
la.Renderer.frame = spy
with contextlib.redirect_stdout(io.StringIO()):
    la.main([str(out), "--pair", A, B, "--size", "100", "--n-points", "150",
             "--intro-frames", "0", "--hold-frames", "4", "--fps", "10", "--hold", "2",
             "--out", str(out / "b.gif")])
la.Renderer.frame = orig
check("CLI: --tsne-hold-frames defaults to --hold-frames; transition in seconds",
      [sum(1 for i, j, _ in calls if i == j == k) for k in range(3)] == [4, 4, 4]
      and sum(1 for i, j, _ in calls if (i, j) == (0, 1)) == 20)

# ---------------------------------------------------------------- popup and argv
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402

d = lr.FeatureNamesDialog(None, cfg, [A, B, C], "animate")
o = d.options()
check("popup: pause / t-SNE pause / transition, defaults 30 / 30 / 40",
      (o["hold_frames"], o["tsne_hold_frames"], o["transition_frames"]) == (30, 30, 40))
d.hold_frames.setValue(12)
d.tsne_hold_frames.setValue(0)
d.transition_frames.setValue(25)
o = d.options()
argv = lr.animate_argv(out, [(A, B)], out / "x.gif", None, theme="dark", mp4=False,
                       color="tsne", hold_frames=o["hold_frames"],
                       tsne_hold_frames=o["tsne_hold_frames"],
                       transition_frames=o["transition_frames"])
val = lambda f: argv[argv.index(f) + 1]  # noqa: E731
check("argv carries them", (val("--hold-frames"), val("--tsne-hold-frames"),
                            val("--transition-frames")) == ("12", "0", "25"))
check("argv without them is unchanged",
      "--hold-frames" not in lr.animate_argv(out, [(A, B)], out / "x.gif", None,
                                             theme="dark", mp4=False, color="tsne"))

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jo_animate_pause_and_transition_frames: {len(checks) - len(failed)}/"
      f"{len(checks)} checks passed")
sys.exit(1 if failed else 0)
