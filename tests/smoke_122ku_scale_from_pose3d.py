"""Smoke test for patch 122ku — the Body model's scale from a single-camera 3-D run.

User: "Continue" (after 122kt, where the scale estimated per stretch of K7 varied 0.59–0.81 px
per model mm, and width and scale traded off).

* ``fit.scale_from_pose3d``: the camera's pixels per mm at the back's height and place, times
  the animal's size (the back's length from above, frames with the head low) against the model;
  exact on a synthetic animal 1.25× the model seen by a known camera; rearing frames do not
  inflate it; errors when the camera file or the labels are missing;
* ``fit_video`` uses it (``RunConfig.scale_from``) and records it in the metadata;
* the fit form offers the project's 3-D runs (*Scale from*), the CLI ``--scale-from``;
* the virtual rodent's rest pose tucks the limbs under the body (seen from above they stuck
  out of a rat's outline), every angle within range, all four paws on the floor.
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


def main() -> int:
    import cv2
    import pandas as pd

    from mufasa.bodymodel import fit as BF
    from mufasa.bodymodel.render import OverlayStyle, draw_model
    from mufasa.bodymodel.rodent_rig import export_rodent_rig
    from mufasa.pose3d.camera import Camera
    from mufasa.pose3d.io import write_pose3d
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rig = BF.Rig(export_rodent_rig(td / "rat.rig.npz", say=lambda s: None))
        cam = Camera.straight_down((400, 225), 300.0, 450.0, (800, 450))
        cam.save(td / "cam.toml")
        mk = BF.Markers.build(rig, None)
        rest = mk.positions(rig, rig.skinning_matrices(BF.neutral()))
        SIZE, n = 1.25, 400
        rows = []
        for k in range(n):
            # rig frame (facing -y) -> world: the animal turned by phi, scaled, placed on the floor
            phi = rng.uniform(-math.pi, math.pi)
            c, s_ = math.cos(phi), math.sin(phi)
            Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            t = np.r_[rng.uniform(-120, 120, 2), 0]
            rear = k % 4 == 0                               # a quarter of the frames: head high
            row = {}
            for m, X in rest.items():
                P = Rz @ (SIZE * X) + t
                if rear and m.startswith("head"):
                    P[2] += 120
                P += rng.normal(0, 1.0, 3)
                row.update({f"{m}_x": P[0], f"{m}_y": P[1], f"{m}_z": P[2]})
            rows.append(row)
        df3 = pd.DataFrame(rows)
        write_pose3d(df3, td / "v_3d.parquet", {"units": "mm", "camera_file": str(td / "cam.toml")})
        s, info = BF.scale_from_pose3d(rig, td / "v_3d.parquet")
        backs = [m for m in BF.BACK_CHAIN if m.startswith("back_")]
        z = float(np.median(np.concatenate([df3[f"{m}_z"].to_numpy() for m in backs])))
        u = cam.project(np.array([[0, 0, z], [10, 0, z]]))
        ppmm = np.linalg.norm(u[1] - u[0]) / 10
        check(f"size against the model {info['size_vs_model']:.3f} (true {SIZE}); rearing frames left out",
              abs(info["size_vs_model"] - SIZE) < 0.02 * SIZE)
        check(f"scale {s:.4f} = {info['px_per_mm']:.4f} px/mm at {info['height_mm']:.0f} mm x size "
              f"(expected {ppmm * SIZE:.4f})", abs(s - ppmm * SIZE) < 0.02 * ppmm * SIZE
              and abs(info["px_per_mm"] - ppmm) < 0.01 * ppmm and info["labels"][0] == "head_back")
        write_pose3d(df3, td / "nocam_3d.parquet", {"units": "mm", "camera_file": str(td / "gone.toml")})
        try:
            BF.scale_from_pose3d(rig, td / "nocam_3d.parquet")
            err = ""
        except ValueError as e:
            err = str(e)
        s2, _ = BF.scale_from_pose3d(rig, td / "nocam_3d.parquet", camera=td / "cam.toml")
        check("camera file missing: a clear error; it can be given instead",
              "camera file" in err and abs(s2 - s) < 1e-9)
        few = df3[[c for c in df3 if c.startswith(("head_back", "back_T4"))]]
        write_pose3d(few, td / "few_3d.parquet", {"camera_file": str(td / "cam.toml")})
        try:
            BF.scale_from_pose3d(rig, td / "few_3d.parquet")
            ok = False
        except ValueError as e:
            ok = "at least three" in str(e)
        check("too few back labels in 3-D: a clear error", ok)

        # ---- the stance: limbs tucked under the body
        from mufasa.bodymodel.rodent_rig import STANCE, stance_qpos
        from mufasa.pose3d.rodent_model import RodentModel
        model = RodentModel()
        q = stance_qpos(model)
        check(f"stance: {len(STANCE)} limb joints per side set, all within their ranges",
              int((np.abs(q) > 0).sum()) == 2 * len(STANCE)
              and bool(np.all((q[0] >= model.lo - 1e-9) & (q[0] <= model.hi + 1e-9))))
        flat = BF.Rig(export_rodent_rig(td / "flat.rig.npz", stance=False, say=lambda s: None))

        def parts(r):
            dom = r.W.argmax(1)
            limb = np.isin(dom, [i for i, nm in enumerate(r.names)
                                 if nm.split(".")[0] in ("Femur", "Tibia", "BFoot", "BPhalange1", "Scapula", "Humerus",
                                                         "Radius", "FHand", "FPhalange1")])
            trunk = np.isin(dom, [r.ix[nm] for nm in ["Pelvis"] + [f"Spine{i}" for i in range(1, 7)]])
            return dom, limb, trunk
        out = []
        for r in (flat, rig):
            dom, limb, trunk = parts(r)
            out.append((np.abs(r.V[limb, 0]).max(), np.abs(r.V[trunk, 0]).max()))
        dom, limb, trunk = parts(rig)
        paw = lambda nms: rig.V[np.isin(dom, [rig.ix[n] for n in nms]), 2].min()  # noqa: E731
        check(f"stance: limbs within the trunk's width from above ({out[1][0]:.0f} mm, trunk {out[1][1]:.0f} mm; "
              f"splayed {out[0][0]:.0f} mm), fore and hind paws on the floor",
              out[1][0] <= out[1][1] + 1.0 and out[0][0] > out[0][1] + 5
              and paw(["BPhalange1.L", "BFoot.L"]) < 3 and paw(["FPhalange1.L", "FHand.L"]) < 3)

        # ---- fit_video with the scale from the 3-D run (a few frames of a rendered rat)
        S_IMG = s
        poses = [BF.neutral(tx=300 + 40 * k, ty=225, phi=0.4, s=S_IMG) for k in range(4)]
        extra = [BF.neutral(tx=float(rng.uniform(150, 650)), ty=float(rng.uniform(150, 300)),
                            phi=float(rng.uniform(-3, 3)), s=S_IMG) for _ in range(12)]
        bg = np.full((450, 800, 3), 40, np.uint8)
        style = OverlayStyle(alpha=1.0, outline=False, labels=False, info=False)
        wr = cv2.VideoWriter(str(td / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (800, 450))
        cols = {}
        for p in poses + extra:
            wr.write(draw_model(bg, rig, p, style))
            for nm, X in mk.positions(rig, rig.skinning_matrices(p)).items():
                uv = BF.to_image(X[None], p)[0]
                cols.setdefault(nm + "_x", []).append(uv[0])
                cols.setdefault(nm + "_y", []).append(uv[1])
                cols.setdefault(nm + "_p", []).append(0.97)
        wr.release()
        pd.DataFrame(cols).to_parquet(td / "v.parquet", index=False)
        run = BF.RunConfig(start=0, stop=4, step=1, workers=1, chunk=10,
                           fit=BF.FitConfig(evals_first=600, evals_warm=200),
                           mask=BF.MaskConfig(min_grey_light=60), scale_from=str(td / "v_3d.parquet"))
        log = []
        out = BF.fit_video(td / "rat.rig.npz", td / "v.avi", td / "v.parquet", run, td / "fit.parquet", say=log.append)
        _, meta = BF.read_fit(out)
        check("fit_video: the scale from the 3-D run, logged and in the metadata",
              abs(meta["scale_px_per_unit"] - s) < 1e-9 and meta["scale_from"]["size_vs_model"] == info["size_vs_model"]
              and any("from the 3-D run" in x for x in log))

        # ---- the form and the CLI
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.utils.config_creator import ProjectConfigCreator
        cfgp = ProjectConfigCreator(project_path=str(td), project_name="p", target_list=[],
                                    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1,
                                    file_type="parquet", body_parts=["head_nose", "back_L2"]).config_path
        root = Path(cfgp).parent
        import shutil
        for d in ("videos", "pose", "bodymodel"):
            (root / "sources" / d).mkdir(parents=True, exist_ok=True)
        shutil.copy(td / "v.avi", root / "sources" / "videos" / "v.avi")
        shutil.copy(td / "v.parquet", root / "sources" / "pose" / "v.parquet")
        shutil.copy(td / "rat.rig.npz", root / "sources" / "bodymodel" / "rat.rig.npz")
        r3 = root / "derived" / "pose3d" / "20261120-000000-abcdef"
        r3.mkdir(parents=True)
        (r3 / "run.toml").write_text('stage = "pose3d"\n')
        from mufasa.ui_qt.forms.bodymodel import BodyFitForm
        f = BodyFitForm(config_path=cfgp)
        idx = f.scale_from.findData(str(r3))
        check("fit form: 'Scale from' lists the project's 3-D runs (default: estimated from the video)",
              f.scale_from.currentData() is None and idx >= 0)
        f.scale_from.setCurrentIndex(idx)
        try:
            f.collect_args()
            msg = ""
        except ValueError as e:
            msg = str(e)
        shutil.copy(td / "v_3d.parquet", r3 / "v_3d.parquet")
        args = f.collect_args()
        f.scale.setValue(2.0)
        args2 = f.collect_args()
        check("fit form: the run's file for this video is used; missing: a clear error; a set scale wins",
              "v_3d.parquet" in msg and args["scale_from"] == str(r3 / "v_3d.parquet")
              and args2["scale_from"] is None and args2["scale"] == 2.0)
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "body-fit", "--help"],
                           capture_output=True, text=True, cwd=REPO, timeout=120)
        check("CLI: body-fit --scale-from", "--scale-from" in r.stdout)
        check("docs: Scale from a 3-D run", "--scale-from" in (REPO / "docs" / "bodymodel.md").read_text())

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ku_scale_from_pose3d: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
