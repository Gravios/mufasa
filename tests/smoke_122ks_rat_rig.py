"""Smoke test for patch 122ks — the virtual rodent (a rat) as a rig for the Body model tab.

User: "looks good, continue" (after 122kr; the plan's next step: the rat rig importer for the
Body model tab, with rig-specific profiles).

* the built-in virtual rodent exports as a .rig.npz without Blender: HBP conventions (facing
  -y, left +x, z up, floor at 0, mm, parents first, rooted at the pelvis), its own profile
  (spine regions, posture limits from the model's joint ranges, label positions);
* kinematics: rest = skin; flexion lowers the nose, extension raises it; rearing keeps the
  hind paws down; the thorax does not bend; the limits are the rat's;
* rigs without a profile are still the HBP mouse (defaults unchanged);
* masks: a light animal is recognised (auto), and the region holding its labels is taken
  even where the ground differs from the background by more;
* a synthetic white rat on darker bedding is followed (position, heading, silhouette);
* the forms (import the virtual rodent, the animal's polarity) and the CLI.
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
    import cv2
    import pandas as pd

    from mufasa.bodymodel import fit as BF
    from mufasa.bodymodel.render import OverlayStyle, draw_model
    from mufasa.bodymodel.rodent_rig import POSTURE, decimate, export_rodent_rig
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = export_rodent_rig(td / "rat.rig.npz", say=lambda s: None)
        rig = BF.Rig(path)
        check("virtual rodent exported without Blender: 63 bones, parents first, rooted at the pelvis",
              len(rig.names) == 63 and rig.names[0] == "Pelvis" and rig.parent[0] == -1
              and all(rig.parent[i] < i for i in range(1, rig.B)))
        V = rig.V
        nose = int(V[:, 1].argmin())
        check(f"HBP conventions: facing -y (nose at y {V[nose, 1]:.0f} mm), floor at z 0, mm",
              V[nose, 1] < -100 and abs(V[:, 2].min()) < 1e-6 and 250 < np.ptp(V[:, 1]) < 400)
        check("profile: its own name, spine regions (6 lumbar, 7 neck, no thoracic), limits, label positions",
              rig.profile["name"].startswith("virtual rodent") and len(rig.groups["lum"]) == 6
              and len(rig.groups["cer"]) == 7 and rig.groups["tho"] == [] and rig.posture == [tuple(p) for p in POSTURE]
              and "virtual rodent" in rig.limits["lumbar (Spine1-6)"] and "head_back" in rig.profile["markers"])
        check(f"fitting mesh decimated ({len(rig.Vd)} of {len(V)} vertices), weights normalised",
              len(rig.Vd) < len(V) / 2 and np.allclose(rig.Wd.sum(1), 1, atol=1e-5))
        Vd, Fd, Wd = decimate(V, rig.F, rig.W, 6.0)
        check("decimation: coarser grid, fewer faces, no collapsed faces",
              len(Fd) < len(rig.Fd) and (Fd[:, 0] != Fd[:, 1]).all() and (Fd[:, 1] != Fd[:, 2]).all())

        G0 = rig.skinning_matrices(BF.neutral())
        check("rest pose reproduces the skin", np.allclose(BF.Rig.lbs(G0, rig.V, rig.W), rig.V, atol=1e-6))

        def nose_z(**kw):
            return BF.Rig.lbs(rig.skinning_matrices(BF.neutral(**kw)), rig.V[[nose]], rig.W[[nose]])[0, 2]
        z0 = V[nose, 2]
        check(f"flexion lowers the nose ({nose_z(lum_pitch=8):.0f} mm), extension raises it "
              f"({nose_z(lum_pitch=-12):.0f} mm; rest {z0:.0f})",
              nose_z(lum_pitch=8) < z0 - 30 and nose_z(lum_pitch=-12) > z0 + 30)
        G = rig.skinning_matrices(BF.neutral(rear=45))
        paws = [rig.ix[n] for n in ("BPhalange1.L", "BPhalange1.R")]
        pz = [(G[i] @ np.r_[rig.heads[i], 1])[2] - rig.heads[i][2] for i in paws]
        check(f"rearing turns about the hind paws (paw height change {max(map(abs, pz)):.1f} mm; nose up to "
              f"{nose_z(rear=45):.0f} mm)", max(map(abs, pz)) < 6 and nose_z(rear=45) > 150)
        same = rig.skinning_matrices(BF.neutral(tho_yaw=5, tho_pitch=5))
        check("the thorax is rigid (thoracic bend has no effect)", np.allclose(same, G0))
        mk = BF.Markers.build(rig, None)
        P = mk.positions(rig, G0)
        check("labels where the model puts them: nose in front, back labels on top, tail behind",
              P["head_nose"][1] < P["head_back"][1] < P["back_T4"][1] < P["back_L2"][1] < P["back_V2"][1]
              < P["tail_V6"][1] < P["tail_V32"][1] and min(P[k][2] for k in ("back_T4", "back_L2", "back_L6")) > 75)

        # rigs without a profile: the HBP mouse defaults
        z = dict(np.load(path))
        z.pop("profile")
        np.savez(td / "noprof.rig.npz", **z)
        r0 = BF.Rig(td / "noprof.rig.npz")
        check("a rig without a profile is the HBP mouse (its groups, limits, the 17-value posture)",
              r0.profile["name"] == "HBP mouse" and r0.posture == BF.POSTURE and r0.limits == BF.LIMITS
              and r0.groups["tho"][0] == "Spine7")

        # ---- a white rat on darker bedding
        n = 14
        poses = [BF.neutral(tx=230 + 330 * k / (n - 1), ty=230 + 40 * math.sin(k / 3), phi=0.25 + 0.5 * k / (n - 1),
                            s=S_TRUE, lum_yaw=4 * math.sin(k / 2), head_yaw=15 * math.sin(k / 3),
                            tail_yaw_p=4, tail_yaw_d=-3) for k in range(n)]
        extra = [BF.neutral(tx=float(rng.uniform(150, 650)), ty=float(rng.uniform(150, 300)),
                            phi=float(rng.uniform(-3, 3)), s=S_TRUE) for _ in range(30)]
        bg = np.full((H_IMG, W_IMG, 3), 40, np.uint8)
        for _ in range(700):
            x, y = rng.integers(0, W_IMG), rng.integers(0, H_IMG)
            g = int(rng.integers(20, 70))
            cv2.rectangle(bg, (int(x), int(y)), (int(x + rng.integers(4, 12)), int(y + rng.integers(4, 12))), (g, g, g), -1)
        style = OverlayStyle(alpha=1.0, outline=False, labels=False, info=False)
        wr = cv2.VideoWriter(str(td / "rat.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W_IMG, H_IMG))
        for k, p in enumerate(poses + extra):
            img = draw_model(bg, rig, p, style)
            if k < 6:                                   # bedding pushed about: a bright patch, briefly
                cv2.rectangle(img, (560, 40), (720, 160), (235, 235, 235), -1)
            wr.write(img)
        wr.release()
        cols = {}
        for p in poses + extra:
            mp = mk.positions(rig, rig.skinning_matrices(p))
            for nm, X in mp.items():
                uv = BF.to_image(X[None], p)[0] + rng.normal(0, 1.5, 2)
                cols.setdefault(nm + "_x", []).append(uv[0])
                cols.setdefault(nm + "_y", []).append(uv[1])
                cols.setdefault(nm + "_p", []).append(0.97)
        pd.DataFrame(cols).to_parquet(td / "rat.parquet", index=False)
        labels = BF.load_labels(td / "rat.parquet")
        bgm = BF.background(td / "rat.avi", list(range(n + len(extra))))
        # the rendered rat's shaded flanks are darker than a real white rat's: a lower brightness floor
        mcfg = BF.MaskConfig(min_grey_light=60)
        sf = BF.SequenceFitter(rig, td / "rat.avi", labels, BF.FitConfig(scale=S_TRUE, evals_first=2500,
                                                                         evals_warm=500), mcfg, bg=bgm)
        imgs = sf.read(range(n))
        check("auto: a light animal (pixels under the back labels brighter than the background)",
              BF.animal_polarity(imgs, bgm, labels) == "light")
        exp = BF.silhouette_area(rig, S_TRUE)
        g0 = imgs[0]
        near = BF.body_points(BF.observed(labels, 0), mk.spec)
        m_any, _ = BF.animal_mask(g0, bgm, BF.MaskConfig(polarity="light", min_grey_light=60, near_labels=False,
                                                         area_range=(0, 1e9)), exp)
        m_lab, a_lab = BF.animal_mask(g0, bgm, BF.MaskConfig(polarity="light", min_grey_light=60), exp, near)
        truth0 = np.zeros((H_IMG, W_IMG), np.uint8)
        BF.fill_triangles(truth0, BF.to_image(BF.Rig.lbs(rig.skinning_matrices(poses[0]), rig.Vd, rig.Wd),
                                              poses[0])[rig.body_faces_d])
        iou = (m_lab & (truth0 > 0)).sum() / max(1, (m_lab | (truth0 > 0)).sum()) if m_lab is not None else 0
        check(f"mask: without the labels the bright patch wins; with them, the rat (IoU {iou:.2f} with the model)",
              m_any is not None and m_any[100, 700] and m_lab is not None and not m_lab[100, 700] and iou > 0.65)
        res = sf.fit(range(n), say=lambda s: None)
        def body_px(p):                                 # the labels' points on the model, without the tail tip
            q = mk.positions(rig, rig.skinning_matrices(p))
            return BF.to_image(np.array([q[k] for k in q if k not in ("tail_V18", "tail_V32")]), p)
        err = [np.linalg.norm(body_px(r.params) - body_px(t), axis=1).mean() for r, t in zip(res, poses)]
        def axis(p):                                    # the body's direction: back_V2 -> head_back, from above
            q = mk.positions(rig, rig.skinning_matrices(p))
            d = BF.to_image(np.array([q["head_back"], q["back_V2"]]), p)
            return math.atan2(*(d[0] - d[1])[::-1])
        dphi = [abs((axis(r.params) - axis(t) + math.pi) % (2 * math.pi) - math.pi) for r, t in zip(res, poses)]
        ious = [1 - r.parts.get("sil", 1) for r in res]
        check(f"fit follows the white rat (body points {np.median(err):.1f} px, body axis {np.degrees(np.median(dphi)):.1f}°, "
              f"IoU {np.median(ious):.2f}; mask used in {sum(r.mask_used for r in res)}/{n})",
              np.median(err) < 6 and np.degrees(np.median(dphi)) < 6 and np.median(ious) > 0.75
              and sum(r.mask_used for r in res) >= n - 1 and sf.mcfg.polarity == "light")
        check("all postures within the rat's limits",
              all(lo - 1e-6 <= r.params[nm] <= hi + 1e-6 for r in res for nm, lo, hi in rig.posture))

        # ---- the tab and the CLI
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.utils.config_creator import ProjectConfigCreator
        cfgp = ProjectConfigCreator(project_path=str(td), project_name="p", target_list=[],
                                    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1,
                                    file_type="parquet", body_parts=["head_nose", "back_L2"]).config_path
        root = Path(cfgp).parent
        for d in ("videos", "pose", "bodymodel"):
            (root / "sources" / d).mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(td / "rat.avi", root / "sources" / "videos" / "rat.avi")
        shutil.copy(td / "rat.parquet", root / "sources" / "pose" / "rat.parquet")
        from mufasa.ui_qt.forms.bodymodel import BodyFitForm, RigImportForm
        rf = RigImportForm(config_path=cfgp)
        rf.source.setCurrentIndex(1)
        a = rf.collect_args()
        rf.target(**a)
        check("rig import form: 'Virtual rodent (rat, built in)' needs no .blend, saved as rat.rig.npz",
              rf.name.text() == "rat" and not rf.blend.isEnabled()
              and (root / "sources" / "bodymodel" / "rat.rig.npz").is_file())
        f = BodyFitForm(config_path=cfgp)
        check("fit form: the rat rig listed; the animal darker/lighter (auto by default)",
              f.rig.findText("rat") >= 0 and f.polarity.currentData() == "auto")
        f.polarity.setCurrentIndex(2)
        args = f.collect_args()
        check("fit form: the polarity reaches the run", args["polarity"] == "light")
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "body-rig", "--virtual-rodent",
                            str(td / "cli.rig.npz")], capture_output=True, text=True, cwd=REPO, timeout=300)
        check(f"CLI body-rig --virtual-rodent ({r.returncode})", r.returncode == 0 and (td / "cli.rig.npz").is_file())
        check("docs: bodymodel.md covers the rat",
              "virtual rodent" in (REPO / "docs" / "bodymodel.md").read_text().lower())

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ks_rat_rig: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
