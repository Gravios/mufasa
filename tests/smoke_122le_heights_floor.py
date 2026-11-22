"""Smoke test for patch 122le — heights the body can have, and the model on top of the floor.

User: "we need to adapt the model to prevent clipping through the floor, and override heights
which are not physiologically possible, since the height is sometimes poorly estimated near the
ground."

* ``enforce_heights``: every marker within its body-model height range (scaled to the animal)
  and above the floor, moved along its camera ray (it still reprojects onto its 2-D label), or
  straight up/down without a camera; ``<m>_z_fixed`` flags; project labels mapped to the model's;
* ``Anatomy.limit_head``: the rigid head is kept with every head label in range and the nose
  above the floor (before: the ears' mean only; K7's nose was under the floor in 5 % of frames);
* the 3-D inference run, the form's 3-D smoothing and ``mufasa-pose3d heights`` (for files
  already made) all apply it;
* the viewer's rig fit: floor probes and an exact lift keep the trunk above the floor (per frame
  and batched alike), with marker targets clipped to the body's heights; the limbs and the tail,
  which the rig cannot bend, are laid on the floor in the drawn mesh instead of lifting the body
  off its markers.
"""
from __future__ import annotations

import io
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def camera():
    from mufasa.pose3d.camera import Camera
    K = np.array([[700.0, 0, 640], [0, 700.0, 512], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0])
    C = np.array([20.0, -30.0, 700.0])
    return Camera(K=K, R=R, t=-R @ C, image_size=(1280, 1024))


def table(rng, n=200):
    """Markers of a rat on the floor, some under it and some too high."""
    base = {"head_nose": 25.0, "head_left": 40.0, "head_right": 40.0, "head_back": 45.0, "back_T8": 60.0,
            "tail_V18": 5.0}
    rows = {"frame": np.arange(n)}
    for k, (m, z) in enumerate(base.items()):
        rows[f"{m}_x"] = rng.uniform(-150, 150, n) + 10 * k
        rows[f"{m}_y"] = rng.uniform(-100, 100, n)
        rows[f"{m}_z"] = z + rng.normal(0, 8, n)
    df = pd.DataFrame(rows)
    df.loc[::10, "head_nose_z"] = -12.0               # under the floor
    df.loc[5::20, "back_T8_z"] = 600.0                # far above any back
    df.loc[::7, "tail_V18_z"] = -4.0
    return df


def main() -> int:
    from mufasa.pose3d import rodent_model
    from mufasa.pose3d.io import marker_names, read_pose3d, write_pose3d
    from mufasa.pose3d.mono import Anatomy, camera_center_from_meta, enforce_heights
    rng = np.random.default_rng(3)
    cam = camera()
    names = ["head_nose", "head_left", "head_right", "head_back", "back_T8", "tail_V18"]

    # -- enforce_heights, along the rays ------------------------------------ #
    df = table(rng)
    before = df.copy()
    fixed = enforce_heights(df, names, camera=cam, scale=1.0)
    anat = Anatomy(rodent_model.load_limits(), 1.0, {n: n for n in names})
    inside = all(((df[f"{m}_z"] >= anat.zrange_floor(m)[0] - 1e-9) & (df[f"{m}_z"] <= anat.zrange_floor(m)[1] + 1e-9)).all()
                 for m in names)
    uv0 = {m: cam.project(before[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy()) for m in names}
    uv1 = {m: cam.project(df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy()) for m in names}
    rep_px = max(float(np.abs(uv0[m] - uv1[m]).max()) for m in names)
    check(f"every marker within its range and above the floor ({fixed})", inside and fixed["head_nose"] >= 20
          and fixed["back_T8"] >= 10 and fixed["tail_V18"] >= 29)
    check(f"moved along the camera rays: reprojection changes by {rep_px:.2e} px", rep_px < 1e-6)
    flag_ok = all((df[f"{m}_z_fixed"].to_numpy() == (np.abs(df[f"{m}_z"] - before[f"{m}_z"]) > 1e-9).to_numpy()).all()
                  for m in names)
    check("<m>_z_fixed marks exactly the frames moved", flag_ok)

    # without a camera: straight up/down; with no model: the floor only
    d2 = table(rng)
    b2 = d2.copy()
    enforce_heights(d2, names, scale=1.0, model=None)
    xy_same = all(np.allclose(d2[f"{m}_x"], b2[f"{m}_x"]) and np.allclose(d2[f"{m}_y"], b2[f"{m}_y"]) for m in names)
    check("no camera: straight up; no model: the floor only (600 mm backs kept)",
          xy_same and (d2[[f"{m}_z" for m in names]].min().min() >= 0) and (d2["back_T8_z"] == 600.0).sum() == 10)

    # project labels named otherwise
    d3 = table(rng).rename(columns=lambda c: c.replace("back_T8", "spine2"))
    enforce_heights(d3, [n.replace("back_T8", "spine2") for n in names], camera=cam, scale=1.0,
                    label_map={"spine2": "back_T8"})
    check("project labels mapped to the model's (spine2 → back_T8: the 600 mm frames lowered)",
          d3["spine2_z"].max() <= anat.zrange("back_T8")[1] + 1e-9)

    # -- limit_head: rigid, every head label, the floor wins ----------------- #
    head = ["head_back", "head_left", "head_right", "head_nose"]
    N = 50
    X = {}
    for m, z in zip(head, (45.0, 40.0, 40.0, 25.0)):
        X[m] = np.c_[rng.uniform(-80, 80, N), rng.uniform(-80, 80, N), np.full(N, z)]
    for m in head:
        X[m][:20, 2] -= 35.0                              # sunk: the nose 10 mm under the floor
    X0 = {m: v.copy() for m, v in X.items()}
    raised, lowered = anat.limit_head(cam, X, head)
    C = cam.center
    k = [(X[m][:, 2] - C[2]) / (X0[m][:, 2] - C[2]) for m in head]
    rigid = all(np.allclose(k[0], kk) for kk in k) and all(
        np.allclose(X[m] - C, k[0][:, None] * (X0[m] - C)) for m in head)
    check(f"limit_head: the nose above the floor ({raised} frames raised), the head moved as one along "
          "the rays", raised >= 20 and min(X[m][:, 2].min() for m in head) >= -1e-9 and rigid)

    # -- the CLI, on a file made before this patch --------------------------- #
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d4 = table(rng)
        meta = {"units": "mm", "method": "single camera", "camera": cam.viewer_meta(),
                "report": {"anatomy": {"model": "virtual_rodent", "scale": 1.0}}}
        write_pose3d(d4, td / "a_3d.parquet", meta)
        from mufasa.tools import pose3d_cli
        out = io.StringIO()
        with redirect_stdout(out):
            rc = pose3d_cli.main(["heights", str(td / "a_3d.parquet"), "--out", str(td / "b_3d.parquet")])
        d5, m5 = read_pose3d(td / "b_3d.parquet")
        cc = camera_center_from_meta(m5)
        px = max(float(np.abs(cam.project(d4[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy())
                              - cam.project(d5[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy())).max()) for m in names)
        check(f"mufasa-pose3d heights: exit {rc}, {out.getvalue().strip()[:90]}…; camera centre from the file "
              f"({np.round(cc, 3).tolist()}), reprojection kept ({px:.1e} px)",
              rc == 0 and np.allclose(cc, cam.center) and px < 1e-6 and d5[[f"{m}_z" for m in names]].min().min() >= -1e-9
              and m5["report"]["heights_fixed"]["head_nose"] >= 20 and "above the floor" in m5["method"])

        # -- the form's 3-D smoothing ---------------------------------------- #
        from mufasa.pose3d.kalman3d import Kalman3DConfig
        from mufasa.ui_qt.forms.pose3d import smooth_run
        run = td / "run1"
        run.mkdir()
        write_pose3d(table(rng), run / "v_3d.parquet", meta)
        with redirect_stdout(io.StringIO()):
            res = smooth_run(run, Kalman3DConfig())
        d6, m6 = read_pose3d(res / "v_3d.parquet")
        check(f"3-D smoothing in the form: heights after the smoothing ({m6['report'].get('heights_fixed')})",
              d6[[f"{m}_z" for m in marker_names(d6)]].min().min() >= -1e-9 and "heights_fixed" in m6["report"]
              and d6["back_T8_z"].max() <= anat.zrange("back_T8")[1] + 1e-9)

        # -- the 3-D inference run ------------------------------------------- #
        from mufasa.pose3d import pipeline
        cam.save(td / "cam.toml")
        real = pipeline.load_pose2d, pipeline.reconstruct_mono
        d7 = table(rng).rename(columns=lambda c: c.replace("back_T8", "spine2"))
        try:
            pipeline.load_pose2d = lambda *a, **k: None
            pipeline.reconstruct_mono = lambda *a, **k: (d7.copy(), {"anatomy": {
                "model": "virtual_rodent", "scale": 1.0, "labels": {"spine2": "back_T8"}}})
            st = pipeline.RunSettings(kalman=None)
            r = pipeline.run_job(pipeline.Job(stem="v", pose=Path("x"), camera=td / "cam.toml"), st, td / "out")
        finally:
            pipeline.load_pose2d, pipeline.reconstruct_mono = real
        d8, m8 = read_pose3d(r["output"])
        check(f"3-D inference: the last step, with the run's label mapping ({m8['report']['heights_fixed']})",
              d8[[f"{m}_z" for m in marker_names(d8)]].min().min() >= -1e-9
              and d8["spine2_z"].max() <= anat.zrange("back_T8")[1] + 1e-9 and "above the floor" in m8["method"])

        # -- the viewer's rig: on top of the floor ---------------------------- #
        import mufasa.bodymodel.fit as BF
        from mufasa.pose3d import rig_batch
        from mufasa.pose3d import scene_models as SM
        from mufasa.pose3d.rig_floor import (
            clip_targets,
            floor_probes,
            target_z_range,
            trunk_vertices,
        )
        from mufasa.ui_qt.pose3d_viewer import load_pose3d
        rig = BF.Rig(SM.builtin_rig_path())
        mk = BF.Markers.build(rig, None)
        rows = []
        for i in range(24):
            p = BF.neutral(lum_pitch=3 + 2 * math.sin(i / 3), head_yaw=12 * math.sin(i / 4), rear=6 * max(0.0, math.sin(i / 5)))
            pos = mk.positions(rig, rig.skinning_matrices(p))
            P = SM._yaw(1.3 * np.array(list(pos.values())), 0.5 + 0.05 * i) + np.array([5.0 * i, 0.0, -30.0])
            row = {"frame": i}                                          # sunk 30 mm: as K7's fits went
            for nm, Xm in zip(pos, P):
                row.update({f"{nm}_x": Xm[0], f"{nm}_y": Xm[1], f"{nm}_z": Xm[2]})
            rows.append(row)
        pd.DataFrame(rows).to_parquet(td / "sunk_3d.parquet", index=False)
        pose = load_pose3d(td / "sunk_3d.parquet")
        cpu = SM.load_model("virtual_rodent", pose, size=1.3)

        def trunk_low(m, f):
            x = m.fit_frame(f)[0]
            _, G, pp = m._markers_world(x)
            return float(m._probe_z(G, pp, x[3], every=True).min())
        low_cpu = min(trunk_low(cpu, f) for f in range(0, 24, 3))
        drawn = min(float(cpu.world(f)[:, 2].min()) for f in range(0, 24, 3))
        check(f"per-frame rig fit: the trunk's lowest point {low_cpu:+.2e} mm (markers sunk 30 mm); the drawn "
              f"mesh's {drawn:+.1e} mm (limbs and tail on the floor)", low_cpu >= -1e-6 and drawn >= 0)
        if rig_batch.available():
            bat = SM.load_model("virtual_rodent", pose, size=1.3)
            bat.fit_frames(range(24), device="cpu")
            low_b = min(trunk_low(bat, f) for f in range(24))
            rms_c = np.median([cpu.fit_frame(f)[1] for f in range(0, 24, 3)])
            rms_b = np.median([bat.fit_frame(f)[1] for f in range(0, 24, 3)])
            prob = rig_batch.RigProblem(bat, "cpu")
            import torch
            x = torch.tensor(bat.fit_frame(5)[0])
            P, G, pp = bat._markers_world(x.numpy())
            par = float(np.abs(prob.probe_z(x).numpy() - bat._probe_z(G, pp, float(x[3]))).max())
            parm = float(np.abs(prob.mesh_z(x).numpy() - bat._probe_z(G, pp, float(x[3]), every=True)).max())
            check(f"batched rig fit: the trunk's lowest point {low_b:+.2e} mm; marker RMS {rms_b:.1f} vs per frame {rms_c:.1f} mm; "
                  f"probe / mesh heights equal the per-frame ones ({par:.0e}, {parm:.0e} mm)",
                  low_b >= -1e-6 and abs(rms_b - rms_c) < 3.0 and par < 1e-9 and parm < 1e-9)
        else:
            check("batched rig fit: PyTorch not installed (skipped)", True)
        pr = floor_probes(rig)
        own = set(rig.Wd[pr].argmax(1).tolist())
        tv = trunk_vertices(rig)
        trunk_own = set(rig.Wd[tv].argmax(1).tolist())
        lo, hi = target_z_range(["head_nose", "back_T8", "nothing"], 1.3)
        Yc = clip_targets(np.array([[0, 0, -20.0], [0, 0, 900.0], [0, 0, -5.0]]), lo, hi)
        check(f"{len(pr)} floor probes over all {len(own)} trunk bones that own vertices ({len(tv)} trunk vertices); "
              f"targets clipped: nose {Yc[0, 2]:.0f}, back "
              f"{Yc[1, 2]:.0f} (≤ {hi[1]:.0f}), unknown label {Yc[2, 2]:.0f} (the floor)",
              own == trunk_own and set(pr) <= set(tv) and Yc[0, 2] >= 0 and Yc[1, 2] == hi[1]
              and Yc[2, 2] == 0.0)

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: heights and the floor", "mufasa-pose3d heights" in doc and "floor" in doc and "_z_fixed" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122le_heights_floor: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
