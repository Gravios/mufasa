"""Smoke test for patch 122kw — paw placement, weight distribution and gait from 3-D body markers
with a physics model of the body and limbs.

User: "Could you add a physics model of the body and limbs with which we can use to estimate paw
placement and gait? Weight distribution and momentum could be used to minimize paw placement with
physiological bounds." (Mendes et al. 2015; Neckel et al.; TactileArena.)

* the body: the virtual rodent's segment masses (338 g) carried by the markers; centre of mass and
  girdles follow sampled spine postures within a few mm; legs' reach;
* support margin (polygon, segment, point; widened by the paws) and load sharing (centre of
  pressure at the ZMP, most even, non-negative, summing to 1);
* a body standing still keeps all four paws down, its weight shared as its centre of mass lies;
* walking: every paw steps, strides and stances within the bounds, left and right alternate,
  the ZMP stays over the support;
* rearing: the forelimbs cannot reach the floor, the hind paws carry the weight;
* the animal's size (nose to tail base since 122kx), mass as size³ or weighed; files; the 3-D viewer's overlay;
  the 3-D tab's form and run; the CLI.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def synth(model, q0, fps, T, *, v=0.0, size=1.0, rear_deg=None, turn=0.0):
    """Markers of the virtual rodent (standing pose) moved along x at ``v`` mm/s; optionally
    rearing about the hind paws for the middle third."""
    from mufasa.pose3d.rodent_model import LABELS
    T0 = model.fk(q0)
    L = {k: v_[0] for k, v_ in model.labels(q0, T0).items()}
    toe = T0[0, model.body_index["toe_L"], :3, 3]
    floor = min(T0[0, model.body_index[b], 2, 3] for b in ("toe_L", "toe_R"))
    names = list(LABELS)
    X0 = np.array([L[k] for k in names]) - np.array([0, 0, floor])
    xyz = np.zeros((T, len(names), 3))
    for t in range(T):
        X = X0.copy()
        if rear_deg is not None and T // 3 <= t < 2 * T // 3:
            a = math.radians(rear_deg)
            d = X - np.array([toe[0], 0, 0])
            X = np.array([toe[0], 0, 0]) + np.c_[math.cos(a) * d[:, 0] - math.sin(a) * d[:, 2], d[:, 1],
                                                  math.sin(a) * d[:, 0] + math.cos(a) * d[:, 2]]
        X = X * size
        X[:, 0] += v * t / fps
        xyz[t] = X
    return names, xyz


def main() -> int:
    import pandas as pd

    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d import gait as G
    from mufasa.pose3d.rodent_model import RodentModel
    phys = G.BodyPhysics.virtual_rodent()
    check(f"body: {phys.mass_kg * 1000:.0f} g in {len(phys.segments)} segments carried by the markers",
          330 < phys.mass_kg * 1000 < 345 and len(phys.segments) > 40)
    check(f"centre of mass and girdles follow sampled spine postures ({phys.check_mm['com']} mm, "
          f"{phys.check_mm['girdles']} mm)", phys.check_mm["com"] < 5 and phys.check_mm["girdles"] < 5)
    check(f"legs' reach: hind {phys.reach_max['LH']:.0f} mm, fore {phys.reach_max['LF']:.0f} mm",
          phys.reach_max["LH"] > phys.reach_max["LF"] > 40 and phys.reach_min["LH"] < phys.reach_max["LH"])
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]])
    check("support margin: inside +, outside −, a two-paw segment, the paws' radius",
          abs(G.support_margin(np.array([5, 5.0]), sq) - 5) < 1e-9
          and abs(G.support_margin(np.array([13, 5.0]), sq) + 3) < 1e-9
          and abs(G.support_margin(np.array([5, 2.0]), sq[:2]) + 2) < 1e-9
          and abs(G.support_margin(np.array([5, 2.0]), sq[:2], 3.0) - 1) < 1e-9
          and G.support_margin(np.zeros(2), np.zeros((0, 2))) == -np.inf)
    w_mid = G.share_load(sq, np.array([5, 5.0]))
    w_front = G.share_load(sq, np.array([8, 5.0]))
    cop = w_front @ sq
    check(f"loads: even under the middle {np.round(w_mid, 2)}, shifted forward with the ZMP "
          f"(centre of pressure {np.round(cop, 1)}), never negative, summing to 1",
          np.allclose(w_mid, 0.25, atol=0.02) and abs(cop[0] - 8) < 0.3 and (w_front >= 0).all()
          and abs(w_front.sum() - 1) < 1e-9 and np.allclose(G.share_load(sq[:1], np.zeros(2)), 1))

    model = RodentModel()
    q0 = stance_qpos(model)
    fps = 60.0
    names, xyz = synth(model, q0, fps, 120)
    r = G.estimate_gait(names, xyz, np.arange(120), fps, phys=phys)
    s = r.summary
    still = r.frames[r.frames.valid]
    fore = still[["LF_load", "RF_load"]].sum(1).median()
    check(f"standing still: all four paws down throughout, no step; size {s['size_vs_model']} (the model)",
          (still.n_support == 4).all() and len(r.steps) == 4 and abs(s["size_vs_model"] - 1) < 0.03)
    gx = {p: still[f"{p}_girdle_x"].median() for p in G.PAWS}
    expect = (still.com_x.median() - (gx["LH"] + gx["RH"]) / 2) / ((gx["LF"] + gx["RF"]) / 2 - (gx["LH"] + gx["RH"]) / 2)
    check(f"its weight shared as its centre of mass lies: forelimbs {100 * fore:.0f}% (centre of mass "
          f"{100 * expect:.0f}% of the way from the hips to the shoulders); the ZMP inside the support",
          abs(fore - expect) < 0.12 and (still.zmp_margin_mm > 0).all()
          and np.allclose(still[[f"{p}_load" for p in G.PAWS]].sum(1), 1))
    names, xyz = synth(model, q0, fps, 360, v=150.0)
    r = G.estimate_gait(names, xyz, np.arange(360), fps, phys=phys)
    s, st = r.summary, r.steps.dropna(subset=["stride_s"])
    per = st.groupby("paw").size()
    b = G.BOUNDS
    check(f"walking at 15 cm/s: every paw steps ({dict(per)}), strides {st.stride_mm.min():.0f}–"
          f"{st.stride_mm.max():.0f} mm, stances ≥ {st.stance_s.min():.2f} s",
          (per >= 3).all() and st.stride_mm.min() >= 0.5 * b["stride_at_rest_mm"]
          and st.stride_mm.max() <= 1.1 * b["stride_max_mm"] and st.stance_s.min() >= b["stance_min_s"] - 1e-9)
    check(f"walking: duty factors {[s[f'{p}_duty_factor'] for p in G.PAWS]}, left–right phases "
          f"{s['phase_LF_RF']} / {s['phase_LH_RH']}",
          all(0.45 <= s[f"{p}_duty_factor"] <= 0.97 for p in G.PAWS)
          and 0.25 <= s["phase_LF_RF"] <= 0.75 and 0.25 <= s["phase_LH_RH"] <= 0.75)
    ff = r.frames.iloc[int(fps):]                       # the first steps start from a standstill
    check(f"walking: the ZMP over the support in {100 * (1 - s['zmp_outside_support']):.0f}% of frames; "
          f"after the first second, left and right of a girdle never both in the air",
          s["zmp_outside_support"] <= 0.25
          and not ((ff.LF_contact == 0) & (ff.RF_contact == 0)).any()
          and not ((ff.LH_contact == 0) & (ff.RH_contact == 0)).any())
    names, xyz = synth(model, q0, fps, 180, rear_deg=45)
    r = G.estimate_gait(names, xyz, np.arange(180), fps, phys=phys)
    mid = r.frames.iloc[75:105]
    check(f"rearing: forelimbs cannot reach the floor ({int((mid.LF_contact == -1).mean() * 100)}% of frames), "
          f"the hind paws carry {100 * mid[['LH_load', 'RH_load']].sum(1).median():.0f}% of the weight",
          (mid.LF_contact == -1).all() and (mid.RF_contact == -1).all()
          and (mid[["LH_load", "RH_load"]].sum(1) > 0.99).all())
    names, xyz = synth(model, q0, fps, 60, size=1.3)
    r = G.estimate_gait(names, xyz, np.arange(60), fps, phys=phys)
    r2 = G.estimate_gait(names, xyz, np.arange(60), fps, phys=phys, mass_g=500)
    check(f"size from nose to tail base {r.summary['size_vs_model']} (true 1.3); mass {r.summary['mass_g']} g = model × size³; "
          f"weighed mass used when given (forces in grams-force)",
          abs(r.summary["size_vs_model"] - 1.3) < 0.04
          and abs(r.summary["mass_g"] - phys.mass_kg * 1000 * r.summary["size_vs_model"] ** 3) < 1
          and r2.summary["mass_g"] == 500
          and abs(r2.frames[[f"{p}_force_g" for p in G.PAWS]].sum(1).median() - 500) < 5)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names, xyz = synth(model, q0, 30.0, 180, v=150.0)
        cols = {"frame": np.arange(180)}
        for j, n in enumerate(names):
            for k, a in enumerate("xyz"):
                cols[f"{n}_{a}"] = xyz[:, j, k]
        from mufasa.pose3d.io import write_pose3d
        write_pose3d(pd.DataFrame(cols), td / "v1_3d.parquet", {"units": "mm", "fps": 30.0})
        res = G.gait_from_pose3d(td / "v1_3d.parquet")
        g, sp = G.write_gait(res, td / "v1_gait.parquet", {"pose3d": "v1"})
        df, meta = G.read_gait(g)
        check("files: per frame (gait.parquet, summary in its metadata) and per stance (steps.csv)",
              sp.name == "v1_steps.csv" and len(df) == 180 and meta["summary"]["steps"] == res.summary["steps"]
              and "LH_force_g" in df and len(pd.read_csv(sp)) == len(res.steps))
        # the 3-D viewer's overlay
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.pose3d.animation import _qimage_bgr
        from mufasa.pose3d.scene_models import load_model
        from mufasa.ui_qt.pose3d_viewer import Volume3DWidget, load_pose3d
        pose = load_pose3d(td / "v1_3d.parquet")
        ov = load_model(str(g), pose)
        vol = Volume3DWidget(pose)
        vol.setAttribute(Qt.WA_DontShowOnScreen, True)
        vol.resize(400, 300)
        vol.set_frame(90)

        def shot():
            img = QImage(400, 300, QImage.Format_RGB32)
            vol.render(img)
            return _qimage_bgr(img).astype(int)
        a = shot()
        vol.add_model(ov)
        bimg = shot()
        check(f"3-D viewer: paws, support polygon, centre of mass and ZMP drawn ({ov.info(90)})",
              ov.kind == "gait" and int((np.abs(bimg - a).sum(2) > 30).sum()) > 100 and "on the floor" in ov.info(90))
        # the 3-D tab
        from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
        from mufasa.section_provenance import SECTIONS
        root = td / "proj"
        root.mkdir()
        ProjectPaths(root).ensure_skeleton()
        write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "g"})
        run = root / "derived" / "pose3d" / "20261120-070000-abcdef"
        run.mkdir(parents=True)
        (run / "v1_3d.parquet").write_bytes((td / "v1_3d.parquet").read_bytes())
        from mufasa.ui_qt.forms.gait import GaitForm
        f = GaitForm(config_path=str(root / PROJECT_CONFIG_FILENAME))
        f.mass.setValue(420)
        args = f.collect_args()
        f.target(**args)
        outs = list((root / "derived" / "gait").glob("*/v1_gait.parquet"))
        check("3-D tab: 'Gait (paws from a body model)' runs on a 3-D run into derived/gait/<run>/ (mass given)",
              args["mass_g"] == 420 and len(outs) == 1 and (outs[0].parent / "v1_steps.csv").is_file()
              and (outs[0].parent / "run.toml").is_file() and G.read_gait(outs[0])[1]["summary"]["mass_g"] == 420
              and SECTIONS["pose3d_gait"].page == "3-D" and "pose3d" in SECTIONS["pose3d_gait"].depends_on)
        src = (REPO / "mufasa" / "ui_qt" / "pages" / "pose3d_page.py").read_text()
        check("the 3-D page lists the form", "GaitForm" in src)
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "gait", str(td / "v1_3d.parquet"),
                            "--out", str(td / "cli_gait.parquet"), "--mass-g", "400"], capture_output=True, text=True,
                           cwd=REPO, timeout=300)
        check(f"CLI: mufasa-pose3d gait ({r.returncode})", r.returncode == 0 and (td / "cli_gait.parquet").is_file()
              and (td / "cli_steps.csv").is_file())
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: the gait section (physics, bounds, sources)", "## Gait" in doc and "TactileArena" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kw_gait_physics: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
