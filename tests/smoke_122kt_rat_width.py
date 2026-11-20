"""Smoke test for patch 122kt — the rat's width: a heavier base shape (girth) and per-frame widening.

User: "Continue" (after 122ks, whose K7 fits showed rats curled up or sitting covering more floor
than the slim virtual rodent).

* ``rodent_rig.widen``: the trunk wider about the midline from the thorax to the pelvis, tapering
  at the neck and the tail base; head, tail, paws and the back's midline unchanged; linear in the
  girth, so one displacement per unit (``D``) serves every width;
* ``export_rodent_rig(girth=…)``: a heavier base shape; ``Rig.verts`` applies a frame's
  ``widen`` (0…0.5) on top; rigs without a widening shape ignore it, the mouse's posture does not
  include it, old fit files read as unwidened;
* a synthetic rat curled wide is fitted with its width (the slim fit overlaps it less);
* after a fit: the animal's area against the model's, and a girth suggestion;
* the import form's *Girth* (virtual rodent only) and ``body-rig --girth``.
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


W_IMG, H_IMG, S_TRUE = 800, 450, 1.1


def main() -> int:
    import json

    import cv2
    import pandas as pd

    from mufasa.bodymodel import fit as BF
    from mufasa.bodymodel import rodent_rig as RR
    from mufasa.bodymodel.render import OverlayStyle, draw_model
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rig = BF.Rig(RR.export_rodent_rig(td / "rat.rig.npz", say=lambda s: None))
        V, W, ix = rig.V, rig.W, rig.ix
        same = RR.widen(V, W, rig.names, rig.heads, rig.tails, 1.0)
        V13 = RR.widen(V, W, rig.names, rig.heads, rig.tails, 1.3)
        dom = W.argmax(1)
        y_mid = (rig.heads[ix["Spine6"]][1] + rig.tails[ix["Pelvis"]][1]) / 2
        trunk = (np.abs(V[:, 1] - y_mid) < 10) & (V[:, 2] > 15) & np.isin(dom, [ix[f"Spine{i}"] for i in range(1, 7)])
        fixed = np.isin(dom, [ix["Head"]] + [ix[f"Tail{i}"] for i in range(1, 31)]) | (V[:, 2] < 1)
        check("widen: girth 1 changes nothing", np.array_equal(same, V))
        check(f"widen 1.3: the mid-trunk 1.3x wider ({np.median(V13[trunk, 0] / V[trunk, 0]):.2f}); head, tail, "
              f"paws and the midline unchanged; only sideways",
              np.allclose(V13[trunk, 0], 1.3 * V[trunk, 0]) and np.allclose(V13[fixed], V[fixed])
              and np.allclose(V13[:, 1:], V[:, 1:]) and np.allclose(V13[np.abs(V[:, 0]) < 1e-9], V[np.abs(V[:, 0]) < 1e-9]))
        check("the rig carries the widening per unit (D, and Dd for the fitting mesh); widen 0.3 = girth 1.3",
              rig.D.shape == V.shape and rig.Dd.shape == rig.Vd.shape
              and np.allclose(rig.verts(BF.neutral(widen=0.3)), V13) and rig.verts(BF.neutral()) is rig.V)
        rig13 = BF.Rig(RR.export_rodent_rig(td / "rat13.rig.npz", girth=1.3, say=lambda s: None))
        a1, a13 = BF.silhouette_area(rig, 1.0), BF.silhouette_area(rig13, 1.0)
        check(f"girth 1.3 export: heavier base shape ({a13 / a1:.2f}x the area from above), named, in the metadata",
              np.allclose(rig13.V, V13) and 1.12 < a13 / a1 < 1.3 and "girth 1.3" in rig13.profile["name"]
              and json.loads(str(np.load(td / "rat13.rig.npz")["meta"]))["girth"] == 1.3)
        moved = max(np.abs(rig13.V[m[1]] - V[m[1]]).max() for m in rig.profile["markers"].values() if m[0] == "vert")
        check(f"the ear and back labels stay put (head and midline; at most {moved:.2f} mm, the back labels "
              f"lie within 2.5 mm of the midline)", moved < 1.0)
        check(f"suggested girth from the area ratio ({BF.suggest_girth(a13 / a1):.2f} for {a13 / a1:.2f}x)",
              abs(BF.suggest_girth(a13 / a1) - 1.3) <= 0.05 and BF.suggest_girth(0.9) == 1.0)
        check("'widen' is a posture value of the rat (0…0.5), not of the mouse",
              ("widen", 0, 0.5) in rig.posture and all(p[0] != "widen" for p in BF.POSTURE) and "widen" in BF.PN)
        z = dict(np.load(td / "rat.rig.npz"))
        for k in ("D", "Dd"):
            z.pop(k)
        np.savez(td / "nod.rig.npz", **z)
        r0 = BF.Rig(td / "nod.rig.npz")
        check("a rig without a widening shape ignores 'widen'", np.allclose(r0.verts(BF.neutral(widen=0.4)), r0.V))
        old = pd.DataFrame([{**{k: 0.0 for k in BF.PN if k != "widen"}, "frame": f, "tx": 10.0 * f} for f in (0, 4)])
        p2 = BF.params_at(old, 2)
        check("old fit files (no 'widen' column) read as unwidened", p2["widen"] == 0.0 and abs(p2["tx"] - 20) < 1e-9)

        # ---- a heavy rat, curled wide, on darker bedding
        n = 10
        poses = [BF.neutral(tx=240 + 300 * k / (n - 1), ty=225 + 30 * math.sin(k / 3), phi=0.3 + 0.4 * k / (n - 1),
                            s=S_TRUE, lum_yaw=3 * math.sin(k / 2), lum_pitch=4, widen=0.4) for k in range(n)]
        extra = [BF.neutral(tx=float(rng.uniform(150, 650)), ty=float(rng.uniform(150, 300)),
                            phi=float(rng.uniform(-3, 3)), s=S_TRUE, widen=0.4) for _ in range(24)]
        bg = np.full((H_IMG, W_IMG, 3), 40, np.uint8)
        for _ in range(600):
            x, y = rng.integers(0, W_IMG), rng.integers(0, H_IMG)
            g = int(rng.integers(20, 70))
            cv2.rectangle(bg, (int(x), int(y)), (int(x + rng.integers(4, 12)), int(y + rng.integers(4, 12))), (g, g, g), -1)
        style = OverlayStyle(alpha=1.0, outline=False, labels=False, info=False)
        wr = cv2.VideoWriter(str(td / "fat.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W_IMG, H_IMG))
        for p in poses + extra:
            wr.write(draw_model(bg, rig, p, style))
        wr.release()
        mk = BF.Markers.build(rig, None)
        cols = {}
        for p in poses + extra:
            for nm, X in mk.positions(rig, rig.skinning_matrices(p)).items():
                uv = BF.to_image(X[None], p)[0] + rng.normal(0, 1.5, 2)
                cols.setdefault(nm + "_x", []).append(uv[0])
                cols.setdefault(nm + "_y", []).append(uv[1])
                cols.setdefault(nm + "_p", []).append(0.97)
        pd.DataFrame(cols).to_parquet(td / "fat.parquet", index=False)
        labels = BF.load_labels(td / "fat.parquet")
        bgm = BF.background(td / "fat.avi", list(range(n + len(extra))))

        def run(r):
            sf = BF.SequenceFitter(r, td / "fat.avi", labels, BF.FitConfig(scale=S_TRUE, evals_first=2500, evals_warm=500),
                                   BF.MaskConfig(min_grey_light=60), bg=bgm)
            return sf.fit(range(n), say=lambda s: None)
        res = run(rig)
        slim = BF.Rig(td / "rat.rig.npz")
        slim.posture = [p for p in slim.posture if p[0] != "widen"]
        res0 = run(slim)
        wid = np.median([r.params["widen"] for r in res])
        iou, iou0 = (np.median([1 - r.parts["sil"] for r in rr]) for rr in (res, res0))
        # the rendered flanks are shaded darker than the mask threshold, so the mask is a little narrow
        # and the width comes out a little low
        check(f"a rat curled wide (widen 0.40) is fitted with its width ({wid:.2f}); IoU {iou:.2f} against "
              f"{iou0:.2f} for the slim model",
              0.25 < wid < 0.5 and iou > 0.8 and iou > iou0 + 0.05)
        df = pd.DataFrame([r.row() for r in res])
        ratio = BF.body_area_ratio(rig, df)
        df0 = pd.DataFrame([r.row() for r in res0])
        ratio0 = BF.body_area_ratio(slim, df0)
        check(f"area ratio after the fit: {ratio:.2f} with the width, {ratio0:.2f} without (suggests girth "
              f"{BF.suggest_girth(ratio0):.2f})", 0.85 < ratio < 1.12 and ratio0 > ratio + 0.08
              and BF.suggest_girth(ratio0) > 1.1)

        # ---- the form and the CLI
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.utils.config_creator import ProjectConfigCreator
        cfgp = ProjectConfigCreator(project_path=str(td), project_name="p", target_list=[],
                                    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1,
                                    file_type="parquet", body_parts=["head_nose", "back_L2"]).config_path
        root = Path(cfgp).parent
        from mufasa.ui_qt.forms.bodymodel import RigImportForm
        rf = RigImportForm(config_path=cfgp)
        off = rf.girth.isEnabled()
        rf.source.setCurrentIndex(1)
        rf.girth.setValue(1.25)
        a = rf.collect_args()
        rf.target(**a)
        saved = BF.Rig(root / "sources" / "bodymodel" / "rat.rig.npz")
        check("import form: Girth only for the virtual rodent; the rig saved with it",
              not off and rf.girth.isEnabled() and a["girth"] == 1.25 and "girth 1.25" in saved.profile["name"])
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "body-rig", "--virtual-rodent",
                            str(td / "cli.rig.npz"), "--girth", "1.4"], capture_output=True, text=True, cwd=REPO,
                           timeout=300)
        check(f"CLI body-rig --virtual-rodent --girth 1.4 ({r.returncode})",
              r.returncode == 0 and "girth 1.4" in BF.Rig(td / "cli.rig.npz").profile["name"])
        doc = (REPO / "docs" / "bodymodel.md").read_text().lower()
        check("docs: girth and widening", "girth" in doc and "widen" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kt_rat_width: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
