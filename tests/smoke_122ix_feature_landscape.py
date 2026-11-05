"""Smoke test for patch 122ix — mufasa-landscape (dipsplit JPDFs + t-SNE).

Runs for real on a generated project: four sessions with Kalman v2 output
(``derived/smoothed/kalman_v2/<run>/<video>_smoothed_v2.<hash>.parquet``) and
per-family features, with planted structure:

* ``emerg_a`` / ``emerg_b`` — each unimodal, but the pair splits into two
  parallel bands (bimodal only along the anti-diagonal): *emergent*;
* ``bimodal_c`` — bimodal on its own;
* ``inside_roi`` — a 0/1 flag, which must be excluded (trivially "bimodal");
* ``orphan`` — a features-only session, which must be skipped;
* one session whose features are 10 frames shorter than its pose.

Checks dipsplit's mode counts on known distributions (including skewed and
heavy-tailed unimodal ones at large n, where the dip test alone over-calls),
the pair ranking, the per-column plotting cap, and every output file.
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

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


from mufasa.tools import feature_landscape as FL  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- dipsplit on known distributions --------------------------------------
r = np.random.default_rng(3)
n = 20000
cases = {
    "normal": (r.normal(0, 1, n), 1),
    "lognormal (skewed)": (r.lognormal(0, 0.8, n), 1),
    "student-t df=3 (heavy tails)": (r.standard_t(3, n), 1),
    "uniform": (r.uniform(0, 1, n), 1),
    "two Gaussians 4 sd apart": (np.r_[r.normal(0, 1, n // 2), r.normal(4, 1, n // 2)], 2),
    "unequal mixture 30/70": (np.r_[r.normal(-3, .6, 6000), r.normal(1.5, .8, 14000)], 2),
    "three Gaussians": (np.r_[r.normal(0, .5, 7000), r.normal(4, .5, 7000),
                              r.normal(8, .5, 6000)], 3),
}
dp = dict(alpha=0.01, min_n=100, min_dip=0.01, max_valley=0.9)
for name, (x, k) in cases.items():
    got = len(FL.dipsplit(x, **dp))
    check(f"dipsplit: {name} → {k} mode(s) (got {got})", got == k)
cut = FL.split_points(FL.dipsplit(cases["two Gaussians 4 sd apart"][0], **dp))
check("the split of two Gaussians lies between them", len(cut) == 1 and 1.2 < cut[0] < 2.8)

# ---- generated project ------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
bps = ["nose", "head", "back", "tail"]
cfg = Path(ProjectConfigCreator(
    project_path=str(tmp), project_name="L", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=bps).config_path)
P = cfg.parent
old_run = P / "derived/smoothed/kalman_v2/20250101-000000-000000"
run = P / "derived/smoothed/kalman_v2/20261105-120000-abcdef"
old_run.mkdir(parents=True)
run.mkdir(parents=True)
T = 4000
for s in range(4):
    v = f"sess{s:02d}"
    pose = {}
    for i, b in enumerate(bps):
        pose[f"{b}_x"] = r.normal(300 + 10 * i, 50, T)
        pose[f"{b}_y"] = r.normal(200, 40, T)
        pose[f"{b}_p"] = r.uniform(.5, 1, T)
        pose[f"{b}_var_x"] = pose[f"{b}_var_y"] = r.uniform(1, 3, T)
    pd.DataFrame(pose).to_parquet(run / f"{v}_smoothed_v2.4d4273a8.parquet")
    pd.DataFrame({"nose_x": np.zeros(5)}).to_parquet(old_run / f"{v}_smoothed_v2.0000aaaa.parquet")
    Tf = T - 10 if s == 3 else T
    state = r.integers(0, 2, Tf)
    u, w = r.normal(0, 2.5, Tf), r.normal(0, .25, Tf) + np.where(state, 1.3, -1.3)
    fam = P / "derived/features/fam_a"
    fam.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"emerg_a": (u + w) / np.sqrt(2), "emerg_b": (u - w) / np.sqrt(2),
                  "speed": np.abs(r.normal(0, 1, Tf)),
                  "bimodal_c": np.where(r.random(Tf) < .4, r.normal(-3, .5, Tf),
                                        r.normal(2, .6, Tf)),
                  "inside_roi": r.integers(0, 2, Tf).astype(float)}
                 ).to_parquet(fam / f"{v}.parquet")
pd.DataFrame({"emerg_a": np.zeros(10)}).to_parquet(P / "derived/features/fam_a/orphan.parquet")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = FL.main(["--project", str(P), "--frames-per-session", "2000",
                  "--tsne-sample", "1500", "--perplexity", "20", "--n-jobs", "1",
                  "--n-plot", "6", "--max-per-feature", "2"])
log = buf.getvalue()
check("runs to completion", rc == 0)
check("the features-only session is skipped and named",
      "skipping 1 session" in log and "orphan" in log)
check("the latest Kalman v2 run is used (older run's 5-frame files ignored)",
      "sess00: 4000 frames" in log)
check("a pose/features length mismatch is reported and truncated",
      "sess03: pose / features lengths differ (4000, 3990); using the first 3990" in log)
outs = list((P / "derived/plots/feature_landscape").iterdir())
check("one run directory under derived/plots/feature_landscape/", len(outs) == 1)
out = outs[0]
check("all outputs written",
      all((out / f).is_file() for f in ["pairs.csv", "jpdf_top.png", "jpdf_pairs.pdf",
                                        "tsne.png", "tsne.parquet", "run.toml"]))
pairs = pd.read_csv(out / "pairs.csv")
cols = set(pairs.feature_a) | set(pairs.feature_b)
check("pose columns are prefixed kfv2. and limited to x/y",
      {"kfv2.nose_x", "kfv2.tail_y"} <= cols and not any(c.endswith(("_p", "_var_x")) for c in cols))
check("the 0/1 inside_roi flag is excluded from the screen", "inside_roi" not in cols)
top = pairs.iloc[0]
check("the emergent pair ranks first and is flagged emergent",
      {top.feature_a, top.feature_b} == {"emerg_a", "emerg_b"} and bool(top.emergent)
      and top.modes_x == 1 and top.modes_y == 1 and top.modes_joint == 2)
check("it is the only emergent pair", int(pairs.emergent.sum()) == 1)
check("pairs with bimodal_c are multimodal but not emergent",
      pairs[(pairs.feature_a == "bimodal_c") | (pairs.feature_b == "bimodal_c")]
      .pipe(lambda d: d.multimodal.all() and not d.emergent.any()))
check("pairs of unstructured columns are not multimodal",
      not pairs[~pairs.feature_a.isin(["emerg_a", "emerg_b", "bimodal_c"])
                & ~pairs.feature_b.isin(["emerg_a", "emerg_b", "bimodal_c"])].multimodal.any())
picked = FL.pick_pairs(pairs, 6, 2)
cnt = pd.concat([picked.feature_a, picked.feature_b]).value_counts()
check("plot selection caps pairs per column", cnt.max() <= 2 and len(picked) >= 2)
E = pd.read_parquet(out / "tsne.parquet")
check("t-SNE embedding: requested size, all four sessions, finite",
      len(E) == 1500 and set(E.session) == {f"sess{s:02d}" for s in range(4)}
      and np.isfinite(E[["tsne_1", "tsne_2"]].to_numpy()).all())
rt = tomllib.loads((out / "run.toml").read_text())
check("run.toml records stage, parameters and results",
      rt["stage"] == "feature_landscape" and rt["results"]["sessions"] == 4
      and rt["results"]["emergent_pairs"] == 1 and rt["params"]["frames_per_session"] == 2000)

py = tomllib.loads((REPO / "pyproject.toml").read_text())
check("console script and analysis extra declared",
      py["project"]["scripts"].get("mufasa-landscape") == "mufasa.tools.feature_landscape:main"
      and any(d.startswith("diptest") for d in py["project"]["optional-dependencies"]["analysis"])
      and any(d.startswith("diptest") for d in py["project"]["optional-dependencies"]["all"]))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ix_feature_landscape: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
