"""Smoke test for patch 122kx — gait bounds from uninjured rats on a CatWalk (Neckel et al. 2013),
speed-dependent, with a stride clock for the paws' order and timing.

User: attached the Neckel CatWalk data (Mendeley sg3d7dv345, split into parts) to calibrate the
gait bounds of patch 122kw.

* the reference (108 rats, 5780 strides) holds the laws in BOUNDS, and the laws reproduce its
  medians by speed; scale (dynamic similarity); TactileArena's fixed bounds kept;
* the calibration script: a MATLAB 7.3 file of a known gait, split into parts, read in place,
  gives back its laws and phases;
* synthetic walks at 15, 30 and 50 cm/s: strides, duty factors and the paws' order as the rats';
* the animal's size from nose to tail base, not the back's labels;
* the CLI's and the 3-D tab's choice of bounds; docs.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def synthetic_footfalls(rng, laws, offsets, n_rats=12, runs=3):
    """Footfalls of rats walking a lateral-sequence gait by ``laws`` at random speeds."""
    import pandas as pd
    rows = []
    for rat in range(n_rats):
        for run in range(runs):
            v = rng.uniform(150, 600)
            stride = laws["stride_at_rest_mm"] + laws["stride_per_speed_s"] * v
            period = stride / v
            for paw, off in offsets.items():
                fh = "fore" if paw[1] == "F" else "hind"
                sa, sb = laws[f"stance_law_{fh}"]
                stance = sa * v ** sb
                y = (-1 if paw[0] == "R" else 1) * (11.5 if fh == "fore" else 14.5)
                for k in range(6):
                    t_on = 0.3 + (k + off) * period + rng.normal(0, 0.004)
                    x = 900 - v * (t_on - 0.3) + rng.normal(0, 1.5)
                    rows.append((rat, run, paw, x, y, t_on, t_on + stance))
    return pd.DataFrame(rows, columns=["rat", "run", "paw", "x", "y", "t_on", "t_off"])


def write_mat73(path, ff, n_weeks=2):
    """A MATLAB 7.3 file shaped like Neckel et al.'s: CW{rat, week, limb, run} of 4×n footfalls."""
    import h5py
    limbs = ["RF", "RH", "LF", "LH"]
    n_rat, n_run = int(ff.rat.max()) + 1, int(ff.run.max()) + 1
    with h5py.File(path, "w", userblock_size=512) as f:
        refs = f.create_group("#refs#")
        cw = np.empty((n_run, 4, n_weeks, n_rat), dtype=h5py.ref_dtype)
        empty = refs.create_dataset("empty", data=np.zeros(2, np.uint64))
        empty.attrs["MATLAB_class"] = np.bytes_("double")
        empty.attrs["MATLAB_empty"] = np.uint8(1)
        i = 0
        for rat in range(n_rat):
            for w in range(n_weeks):
                for li, paw in enumerate(limbs):
                    for run in range(n_run):
                        r = ff[(ff.rat == rat) & (ff.run == run) & (ff.paw == paw)] if w == 0 else ff.iloc[:0]
                        if len(r):
                            d = refs.create_dataset(f"r{i}", data=r[["x", "y", "t_on", "t_off"]].to_numpy().T)
                            d.attrs["MATLAB_class"] = np.bytes_("double")
                            cw[run, li, w, rat] = d.ref
                            i += 1
                        else:
                            cw[run, li, w, rat] = empty.ref
        f.create_dataset("CW", data=cw)
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, synthetic".ljust(116) + b"\0" * 8 + b"\x00\x02IM")


def main() -> int:
    import pandas as pd

    from mufasa.pose3d import gait as G
    ref = json.loads((REPO / "mufasa/assets/gait/catwalk_rat_reference.json").read_text())
    B = G.BOUNDS
    same = all(np.allclose(ref["laws"][k], B[k], rtol=0.01)
               for k in ("stance_law_fore", "stance_law_hind", "swing_law_fore", "swing_law_hind",
                         "stride_at_rest_mm", "stride_per_speed_s"))
    check(f"reference: {ref['rats']} rats, {ref['strides']} strides; BOUNDS hold its laws",
          ref["rats"] == 108 and ref["strides"] == 5780 and same and "Neckel" in B["source"])
    worst = {"stance": 0.0, "swing": 0.0, "duty": 0.0, "stride": 0.0}
    for fh, rows in ref["by_speed"].items():
        for r in rows:
            L = G.gait_laws(r["speed"], fh[0].upper())
            worst["stance"] = max(worst["stance"], abs(L["stance_s"] / r["stance_s"] - 1))
            worst["swing"] = max(worst["swing"], abs(L["swing_s"] / r["swing_s"] - 1))
            worst["duty"] = max(worst["duty"], abs(L["duty"] - r["duty"]))
            worst["stride"] = max(worst["stride"], abs(L["stride_mm"] - r["stride_mm"]))
    check(f"the laws give the rats' medians by speed (worst: stance {worst['stance']:.0%}, swing "
          f"{worst['swing']:.0%}, duty {worst['duty']:.3f}, stride {worst['stride']:.1f} mm)",
          worst["stance"] < 0.15 and worst["swing"] < 0.15 and worst["duty"] < 0.05 and worst["stride"] < 10)
    a, b4 = G.gait_laws(300, "H"), G.gait_laws(300 * 2, "H", {"scale": 4.0})
    lo, hi = G.gait_laws(20, "F"), G.gait_laws(5000, "F")
    check("scale: lengths × k, times × √k at the dynamically similar speed; laws held outside 10–71 cm/s",
          math.isclose(b4["stride_mm"], 4 * a["stride_mm"]) and math.isclose(b4["stance_s"], 2 * a["stance_s"])
          and math.isclose(b4["duty"], a["duty"]) and math.isclose(lo["duty"], G.gait_laws(100, "F")["duty"])
          and hi["stride_mm"] == B["stride_max_mm"])
    ta = G.bounds_preset("tactilearena")
    t15 = G.gait_laws(150, "H", ta)
    check(f"TactileArena's fixed bounds kept (hind duty {t15['duty']}, stance {t15['stance_s']:.2f} s)",
          t15["duty"] == 0.78 and math.isclose(t15["stance_s"], 0.15 * 0.78 / 0.22)
          and "TactileArena" in ta["source"] and G.bounds_preset("catwalk", 2.0)["scale"] == 2.0)
    try:
        G.bounds_preset("nope")
        bad = False
    except ValueError:
        bad = True
    check("an unknown set of bounds is refused", bad)

    # the calibration script on a known gait, as a split MATLAB 7.3 file
    import calibrate_gait_catwalk as C
    laws = {"stance_law_fore": [12.0, -0.7], "stance_law_hind": [20.0, -0.78],
            "stride_at_rest_mm": 60.0, "stride_per_speed_s": 0.17}
    offs = {"LH": 0.0, "RH": 0.5, "LF": 0.4, "RF": 0.9}
    ff = synthetic_footfalls(np.random.default_rng(3), laws, offs)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_mat73(td / "cw.mat", ff)
        raw = (td / "cw.mat").read_bytes()
        cut = [0, len(raw) // 3, 2 * len(raw) // 3, len(raw)]
        for i, s in enumerate(("aa", "ab", "ac")):
            (td / f"cw_part.{s}").write_bytes(raw[cut[i]:cut[i + 1]])
        whole = C.footfalls(str(td / "cw.mat"))
        parts = C.footfalls(str(td / "cw_part.a?"))
        out = td / "ref.json"
        C.main([str(td / "cw_part.a?"), "-o", str(out)])
        got = json.loads(out.read_text())
    lw = got["laws"]
    ph = got["phases_25_50_75"]
    check(f"calibration: the split parts read in place as one ({len(parts)} footfalls, week 0 only); "
          f"laws back (fore stance {lw['stance_law_fore']}, stride {lw['stride_at_rest_mm']} + "
          f"{lw['stride_per_speed_s']} s × v); phases {ph['LF_RF'][1]}, {ph['LH_LF'][1]}",
          len(parts) == len(whole) == len(ff) and parts.equals(whole)
          and abs(lw["stance_law_fore"][1] + 0.7) < 0.05 and abs(lw["stance_law_hind"][1] + 0.78) < 0.05
          and abs(lw["stride_per_speed_s"] - 0.17) < 0.02 and abs(lw["stride_at_rest_mm"] - 60) < 6
          and abs(ph["LF_RF"][1] - 0.5) < 0.03 and abs(ph["LH_LF"][1] - 0.4) < 0.03
          and abs(got["percentile_table"]["fore"]["bos_mm"][2] - 23) < 1.5)

    # synthetic walks
    from smoke_122kw_gait_physics import synth

    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d.rodent_model import RodentModel
    model = RodentModel()
    q0 = stance_qpos(model)
    phys = G.BodyPhysics.virtual_rodent()
    fps = 60.0
    rows = []
    for v in (150.0, 300.0, 500.0):
        T = int(5 * fps)
        names, xyz = synth(model, q0, fps, T, v=v)
        r = G.estimate_gait(names, xyz, np.arange(T), fps, phys=phys)
        w = G._walking(r.steps)
        w = w[w.on_s > 1.0]
        L = G.gait_laws(v, "F")
        s = r.summary
        rows.append((v, float(w.stride_mm.median()), L["stride_mm"], float(w.duty_factor.median()),
                     (G.gait_laws(v, "F")["duty"] + G.gait_laws(v, "H")["duty"]) / 2,
                     s["phase_LF_RF"], s["phase_LH_RH"], s["phase_LH_LF"], s["walking_strides"]))
    rp = pd.DataFrame(rows, columns=["v", "stride", "law", "duty", "duty_law", "lf_rf", "lh_rh", "lh_lf", "n"])
    check("walking at 15, 30, 50 cm/s: strides " + ", ".join(f"{a:.0f}/{b:.0f}" for a, b in zip(rp.stride, rp.law))
          + " mm (estimated/rats), duty " + ", ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(rp.duty, rp.duty_law)),
          ((rp.stride - rp.law).abs() < 12).all() and ((rp.duty - rp.duty_law).abs() < 0.09).all() and (rp.n > 8).all())
    check("walking: the paws' order as the rats' (left–right " + ", ".join(f"{a}/{b}" for a, b in zip(rp.lf_rf, rp.lh_rh))
          + "; hind→fore " + ", ".join(str(x) for x in rp.lh_lf) + ", rats 0.5 and 0.39)",
          ((rp.lf_rf - 0.5).abs() < 0.08).all() and ((rp.lh_rh - 0.5).abs() < 0.08).all()
          and ((rp.lh_lf - 0.394).abs() < 0.1).all())

    # size from nose to tail base, whatever the back's labels
    names, xyz = synth(model, q0, fps, 30, size=1.2)
    moved = xyz.copy()
    for n in ("back_T4", "back_T8", "back_L2", "back_L6"):
        j = names.index(n)
        moved[:, j, 1] += np.array([12.0, -10.0, 9.0, -11.0])[["back_T4", "back_T8", "back_L2", "back_L6"].index(n)]
    sz, sz_moved = G.animal_size(phys, names, xyz), G.animal_size(phys, names, moved)
    old_path = sum(np.linalg.norm(moved[:, names.index(b), :2] - moved[:, names.index(a), :2], axis=1)
                   for a, b in zip(["head_back", "back_T4", "back_T8", "back_L2", "back_L6"],
                                   ["back_T4", "back_T8", "back_L2", "back_L6", "back_V2"])).mean() / phys.back_path
    check(f"size from nose to tail base: {sz:.3f} (true 1.2), {sz_moved:.3f} with the back's labels off the "
          f"midline (the path along the back: {old_path:.2f})",
          abs(sz - 1.2) < 0.01 and abs(sz_moved - 1.2) < 0.01 and old_path > 1.3)

    # the CLI and the 3-D tab
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names, xyz = synth(model, q0, 30.0, 120, v=200.0)
        cols = {"frame": np.arange(120)}
        for j, n in enumerate(names):
            for k, ax in enumerate("xyz"):
                cols[f"{n}_{ax}"] = xyz[:, j, k]
        from mufasa.pose3d.io import write_pose3d
        write_pose3d(pd.DataFrame(cols), td / "v1_3d.parquet", {"units": "mm", "fps": 30.0})
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "gait", str(td / "v1_3d.parquet"),
                            "--out", str(td / "ta_gait.parquet"), "--bounds", "tactilearena", "--gait-scale", "1.1"],
                           capture_output=True, text=True, cwd=REPO, timeout=300)
        ok = r.returncode == 0
        meta = G.read_gait(td / "ta_gait.parquet")[1]["summary"]["bounds"] if ok else {}
        check(f"CLI: --bounds tactilearena --gait-scale 1.1 ({r.returncode}; recorded in the summary)",
              ok and "TactileArena" in meta.get("source", "") and meta.get("scale") == 1.1 and meta.get("duty_hind") == 0.78)
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
        root = td / "proj"
        root.mkdir()
        ProjectPaths(root).ensure_skeleton()
        write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "g"})
        run = root / "derived" / "pose3d" / "20261120-090000-abcdef"
        run.mkdir(parents=True)
        (run / "v1_3d.parquet").write_bytes((td / "v1_3d.parquet").read_bytes())
        from mufasa.ui_qt.forms.gait import GaitForm
        f = GaitForm(config_path=str(root / PROJECT_CONFIG_FILENAME))
        items = [f.bounds.itemData(i) for i in range(f.bounds.count())]
        f.bounds.setCurrentIndex(items.index("tactilearena"))
        args = f.collect_args()
        f.target(**args)
        outs = list((root / "derived" / "gait").glob("*/v1_gait.parquet"))
        toml = (outs[0].parent / "run.toml").read_text() if outs else ""
        check(f"3-D tab: a Bounds choice {items} (CatWalk first), passed to the run and recorded in run.toml",
              items == ["catwalk", "tactilearena"] and args["bounds"] == "tactilearena" and len(outs) == 1
              and 'bounds = "tactilearena"' in toml
              and "TactileArena" in G.read_gait(outs[0])[1]["summary"]["bounds"]["source"])
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: the CatWalk bounds, the stride clock, the size and the calibration script",
          "Neckel et al. (2013)" in doc and "stride clock" in doc and "nose to the tail's base" in doc
          and "calibrate_gait_catwalk.py" in doc and "--gait-scale" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kx_gait_catwalk: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
