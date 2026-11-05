"""Smoke test for patch 122jb — angles no longer take over mufasa-landscape.

User report (after 122iz): "still only angle pairs … their distributions are
circular and in a JPDF often have two modes as they wrap around".

With 15 markers the three-point-angle family has one column per marker
triple (455), and many of those angles are circular in a way rotation
cannot fix: angles that cover the whole circle (no empty direction to cut
at) and mirror-image pairs (bent left / bent right). Now:

* ``--angles unwrap`` (default) additionally drops full-circle angles;
  ``--angles fold`` maps a → min(a, 360 − a) so mirror images coincide;
  ``--angles drop`` leaves angles out;
* ``--include-families`` / ``--exclude-families`` restrict the screen;
* plotting caps pairs per feature family (``--max-per-family``, default 6),
  filling leftover slots in rank order when few families qualify;
* ``pairs.csv`` gains family_a / family_b; the log counts multimodal pairs
  per family.

Runs mufasa-landscape on a generated project whose angle family is large and
full of circular structure, next to distance / movement / head-velocity
families with real bimodality.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
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

r = np.random.default_rng(11)
dp = dict(alpha=0.01, min_n=100, min_dip=0.01, max_valley=0.9)
n = 20000

# ---- angle treatments on single columns --------------------------------------
mirror = np.where(r.random(n) < .5, r.normal(150, 8, n), r.normal(210, 8, n))
full = np.mod(r.uniform(0, 360, n) + r.normal(0, 5, n), 360)
wrap0 = np.mod(r.normal(0, 12, n), 360)
X = pd.DataFrame({"Angle (degrees) m_FEATURE_SUBSET": mirror,
                  "Angle (degrees) full_FEATURE_SUBSET": full,
                  "Angle (degrees) wrap_FEATURE_SUBSET": wrap0,
                  "dist (mm)": r.gamma(3, 4, n)})
with contextlib.redirect_stdout(io.StringIO()) as lg:
    U = FL.clean_columns(X, angles="unwrap")
check("unwrap: full-circle angle dropped, wrap-around angle kept and unimodal",
      "Angle (degrees) full_FEATURE_SUBSET" not in U.columns
      and len(FL.dipsplit(U["Angle (degrees) wrap_FEATURE_SUBSET"].to_numpy(), **dp)) == 1)
check("…and the log says so", "1 full-circle" in lg.getvalue()
      and "2 unwrapped" in lg.getvalue())
F = FL.clean_columns(X, angles="fold", log=lambda *a: None)
fm = F["Angle (degrees) m_FEATURE_SUBSET"].to_numpy()
check("fold: mirror-image angles (150° / 210°) become one mode near 150°",
      len(FL.dipsplit(mirror, **dp)) == 2 and len(FL.dipsplit(fm, **dp)) == 1
      and abs(np.median(fm) - 150) < 3 and fm.max() <= 180)
D = FL.clean_columns(X, angles="drop", log=lambda *a: None)
check("drop: angles gone, other columns kept", list(D.columns) == ["dist (mm)"])

# ---- generated project ------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = Path(ProjectConfigCreator(
    project_path=str(tmp), project_name="ang", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path)
P = cfg.parent
F_ANG = "within_animal_three_point_body_part_angles_degrees"
F_DIST = "two_point_body_part_distances_mm"
F_MOV = "frame_by_frame_body_part_movements_mm"
F_HV = "head_velocity_in_body_frame_mm_s"
T = 4000
for s in range(3):
    v = f"s{s}"
    moving = r.random(T) < 0.45                      # two locomotor states
    cols = {}
    for k in range(40):                              # a big, circular angle family
        kind = k % 3
        if kind == 0:
            cols[f"Angle (degrees) t{k}_FEATURE_SUBSET"] = np.where(
                r.random(T) < .5, r.normal(140 + k, 6, T), r.normal(220 - k, 6, T))
        elif kind == 1:
            cols[f"Angle (degrees) t{k}_FEATURE_SUBSET"] = r.uniform(0, 360, T)
        else:
            cols[f"Angle (degrees) t{k}_FEATURE_SUBSET"] = np.mod(r.normal(0, 20, T), 360)
    for fam, df in {
        F_ANG: pd.DataFrame(cols),
        F_DIST: pd.DataFrame({f"Animal_1 distance (mm) d{k}_FEATURE_SUBSET":
                              np.where(moving, r.normal(60 + k, 3, T), r.normal(45 + k, 3, T))
                              for k in range(4)}),
        F_MOV: pd.DataFrame({f"Animal_1 movement m{k} (mm)_FEATURE_SUBSET":
                             np.abs(np.where(moving, r.normal(4, .7, T), r.normal(.3, .2, T))
                                    + r.normal(0, .05 * (k + 1), T)) for k in range(3)}),
        F_HV: pd.DataFrame({
            "Animal_1 head velocity forward along body-head axis (mm/s)_FEATURE_SUBSET":
                np.where(moving, r.normal(120, 15, T), r.normal(0, 10, T)),
            "Animal_1 head velocity lateral to body-head axis, left positive (mm/s)_FEATURE_SUBSET":
                r.normal(0, 25, T)}),
    }.items():
        (P / "derived/features" / fam).mkdir(parents=True, exist_ok=True)
        df.to_parquet(P / "derived/features" / fam / f"{v}.parquet")


def run(*extra) -> tuple[pd.DataFrame, str, Path]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FL.main(["--project", str(P), "--sources", "features", "--no-tsne", "--n-jobs", "1",
                 "--frames-per-session", "3000", "--max-corr", "0.999", *extra])
    out = sorted((P / "derived/plots/feature_landscape").iterdir())[-1]
    return pd.read_csv(out / "pairs.csv"), buf.getvalue(), out


pairs, log, out = run()
check("pairs.csv has family_a / family_b",
      {"family_a", "family_b"} <= set(pairs.columns)
      and set(pairs.family_a) | set(pairs.family_b) == {F_ANG, F_DIST, F_MOV, F_HV})
check("the log counts multimodal pairs per family",
      "multimodal pairs per family:" in log and F_HV in log)
cols = set(pairs.feature_a) | set(pairs.feature_b)
check("full-circle angles are not screened",
      not any(f"t{k}_" in c for c in cols for k in range(1, 40, 3)))
top = FL.pick_pairs(pairs, 24, 3, 6)
fam_n = pd.Series({f: int(((top.family_a == f) | (top.family_b == f)).sum())
                   for f in (F_ANG, F_DIST, F_MOV, F_HV)})   # pairs touching each
check(f"plotted pairs span the families; angles touch ≤ 6 of {len(top)} "
      f"({dict(fam_n)})",
      fam_n[F_ANG] <= 6 and (fam_n[[F_DIST, F_MOV, F_HV]] > 0).all())
check("the head-velocity × movement/distance structure (two locomotor states) "
      "is among the plotted pairs",
      ((top.family_a == F_HV) | (top.family_b == F_HV)).any())
few = FL.pick_pairs(pairs[(pairs.family_a == F_ANG) & (pairs.family_b == F_ANG)], 5, 3, 2)
check("the family cap holds even when it leaves slots empty (and the run says so)",
      len(few) == 2 and "caps leave no more" in log)

pairs_f, log_f, _ = run("--angles", "fold")
mm = pairs_f[pairs_f.multimodal & (pairs_f.family_a == F_ANG) & (pairs_f.family_b == F_ANG)]
mm_u = pairs[pairs.multimodal & (pairs.family_a == F_ANG) & (pairs.family_b == F_ANG)]
check(f"--angles fold: far fewer multimodal angle×angle pairs "
      f"({len(mm)} vs {len(mm_u)} unwrapped)", len(mm) < len(mm_u) / 3)

pairs_x, log_x, _ = run("--exclude-families", "angles")
check("--exclude-families angles: no angle columns screened",
      F_ANG not in set(pairs_x.family_a) | set(pairs_x.family_b)
      and "family filter:" in log_x)
pairs_i, _, _ = run("--include-families", "head_velocity,movements")
check("--include-families head_velocity,movements: only those two",
      set(pairs_i.family_a) | set(pairs_i.family_b) == {F_HV, F_MOV})
pairs_d, _, _ = run("--angles", "drop")
check("--angles drop: no angle columns", F_ANG not in set(pairs_d.family_a) | set(pairs_d.family_b))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122jb_landscape_angles_and_families: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
