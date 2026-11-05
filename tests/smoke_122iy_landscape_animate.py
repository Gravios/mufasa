"""Smoke test for patch 122iy — mufasa-landscape-animate.

Generates a project with planted structure, runs mufasa-landscape (which now
also writes tsne_samples.parquet), then renders animations from the run and
checks them: frame counts and timing, that the first/last frames show the
t-SNE map, that points really move to the feature-pair coordinates, pair
selection from pairs.csv and by name, --through-tsne ordering, colour modes,
MP4 output, and the error paths.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import tomllib
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


from mufasa.tools import feature_landscape as FL  # noqa: E402
from mufasa.tools import landscape_animate as LA  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

r = np.random.default_rng(5)
tmp = Path(tempfile.mkdtemp())
bps = ["nose", "tail"]
cfg = Path(ProjectConfigCreator(
    project_path=str(tmp), project_name="A", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=bps).config_path)
P = cfg.parent
run = P / "derived/smoothed/kalman_v2/20261105-120000-abcdef"
run.mkdir(parents=True)
T = 1500
for s in range(3):
    v = f"s{s}"
    pd.DataFrame({f"{b}_{c}": r.normal(200, 30, T) for b in bps for c in "xy"}
                 ).to_parquet(run / f"{v}_smoothed_v2.11aa22bb.parquet")
    w = r.normal(0, .25, T) + np.where(r.integers(0, 2, T), 1.3, -1.3)
    u = r.normal(0, 2.5, T)
    fam = P / "derived/features/f"
    fam.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ea": (u + w) / np.sqrt(2), "eb": (u - w) / np.sqrt(2),
                  "bc": np.where(r.random(T) < .4, r.normal(-3, .5, T), r.normal(2, .6, T)),
                  "sp": np.abs(r.normal(0, 1, T))}).to_parquet(fam / f"{v}.parquet")
with contextlib.redirect_stdout(io.StringIO()):
    FL.main(["--project", str(P), "--frames-per-session", "1000", "--tsne-sample",
             "1200", "--perplexity", "15", "--n-jobs", "1", "--n-plot", "2"])
rd = next((P / "derived/plots/feature_landscape").iterdir())
emb = pd.read_parquet(rd / "tsne.parquet")
smp = pd.read_parquet(rd / "tsne_samples.parquet")
check("mufasa-landscape writes tsne_samples.parquet row-aligned with tsne.parquet",
      len(smp) == len(emb) == 1200 and {"ea", "eb", "kfv2.nose_x"} <= set(smp.columns))

# ---- timeline / easing ---------------------------------------------------
tl = LA.timeline(3, fps=10, hold=1.0, transition=2.0)
check("timeline: hold + move per space, last space held (10+20+10+20+10)",
      len(tl) == 70 and tl[0] == (0, 0, 0.0) and tl[-1] == (2, 2, 0.0))
moves = [t for i, j, t in tl if i != j]
check("easing starts at 0, ends at 1, is monotone and symmetric",
      moves[0] == 0.0 and moves[19] == 1.0 and all(np.diff(moves[:20]) >= 0)
      and abs(LA.ease(0.25) + LA.ease(0.75) - 1) < 1e-12)

# ---- render from the run directory --------------------------------------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = LA.main([str(rd), "--pair", "ea", "eb", "--pair", "sp", "bc", "--n-points", "600",
                  "--fps", "10", "--hold", "0.5", "--transition", "1", "--size", "320",
                  "--mp4"])
gif = rd / "animation.gif"
check("renders a GIF and an MP4 into the run directory",
      rc == 0 and gif.is_file() and (rd / "animation.mp4").stat().st_size > 1000)
log = buf.getvalue()
# t-SNE → ea×eb → sp×bc → t-SNE: 4 holds of 5 + 3 moves of 10 = 50 frames
check("frame count follows the timeline (2 pairs → 4 spaces, 50 frames)",
      "2 pair(s), 4 spaces, 50 frames" in log)
im = Image.open(gif)
check("GIF is 320 px square, loops, and has frames", im.size == (320, 320)
      and im.info.get("loop") == 0 and im.n_frames > 10)
dur = 0
for k in range(im.n_frames):
    im.seek(k)
    dur += im.info.get("duration", 0)
check(f"total GIF duration ≈ 5 s (got {dur} ms; identical hold frames may merge)",
      4800 <= dur <= 5200)
im.seek(0)
first = np.asarray(im.convert("RGB"), float)
im.seek(im.n_frames - 1)
last = np.asarray(im.convert("RGB"), float)
check("the animation starts and ends on the same t-SNE view",
      np.abs(first - last).mean() < 1.0)

# ---- geometry: points go exactly to each space's coordinates -------------
sub = smp.iloc[:300]
S1 = LA.feature_space(sub.ea.to_numpy(), sub.eb.to_numpy(), "ea", "eb")
lo, hi = np.percentile(sub.ea, [0.5, 99.5])
check("feature space maps the 0.5-99.5 percentile range onto the unit square",
      np.allclose(S1.xy[:, 0], np.clip((sub.ea - lo) / (hi - lo), -LA.PAD, 1 + LA.PAD)))
St = LA.tsne_space(emb[["tsne_1", "tsne_2"]].to_numpy()[:300])
def _span(v):
    lo_, hi_ = np.percentile(v, [0.5, 99.5])
    return hi_ - lo_


d = _span(St.xy[:, 0]) / _span(St.xy[:, 1])
e = _span(emb.tsne_1[:300]) / _span(emb.tsne_2[:300])
check("t-SNE space keeps its aspect ratio", abs(d - e) / e < 0.05)
R = LA.Renderer([St, S1], LA.tsne_colors(St.xy), size=200, theme="light",
                point_size=4, caption="")
R.frame(0, 1, 1.0)
at_end = np.asarray(R.sc.get_offsets())
R.frame(0, 1, 0.5)
mid = np.asarray(R.sc.get_offsets())
R.close()
check("at the end of a move every point sits at its feature-space position",
      np.allclose(at_end, S1.xy))
check("half-way, points are half-way", np.allclose(mid, (St.xy + S1.xy) / 2))

# ---- colours ------------------------------------------------------------
c = LA.tsne_colors(St.xy)
check("t-SNE colouring: one RGB per point, distinct across the map",
      c.shape == (300, 3) and len({tuple(np.round(x, 2)) for x in c}) > 100)
cs = LA.point_colors("session", emb.iloc[:300], sub, St.xy)
check("session colouring: one colour per session",
      len({tuple(x) for x in cs}) == emb.iloc[:300].session.nunique())
cv = LA.point_colors("sp", emb.iloc[:300], sub, St.xy)
check("column colouring follows the column's value",
      np.corrcoef(cv[:, 1], sub.sp.clip(*np.percentile(sub.sp, [1, 99])))[0, 1] > 0.8)

# ---- --top from pairs.csv, --through-tsne, run discovery -----------------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = LA.main([str(P), "--top", "2", "--through-tsne", "--n-points", "300",
                  "--fps", "5", "--hold", "0.2", "--transition", "0.4", "--size", "200"])
check("finds the latest run from the project path; --top picks pairs from pairs.csv; "
      "--through-tsne returns between pairs (2 pairs → 5 spaces)",
      rc == 0 and "2 pair(s), 5 spaces" in buf.getvalue()
      and (rd / "animation_1.gif").is_file())

# ---- errors -----------------------------------------------------------------
for args, why in [([str(rd), "--pair", "ea", "nope"], "unknown column"),
                  ([str(rd), "--color", "nope"], "unknown colour")]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            LA.main(args + ["--n-points", "50", "--fps", "2", "--hold", "0.5",
                            "--transition", "1", "--size", "120"])
        ok = False
    except SystemExit as exc:
        ok = isinstance(exc.code, str)
    check(f"{why} → clear error", ok)
(rd / "tsne_samples.parquet").rename(rd / "x.parquet")
try:
    LA.main([str(rd)])
    ok = False
except SystemExit as exc:
    ok = "re-run mufasa-landscape" in str(exc.code)
check("a run without tsne_samples.parquet → asks to re-run mufasa-landscape", ok)

py = tomllib.loads((REPO / "pyproject.toml").read_text())
check("console script declared",
      py["project"]["scripts"].get("mufasa-landscape-animate")
      == "mufasa.tools.landscape_animate:main")

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iy_landscape_animate: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
