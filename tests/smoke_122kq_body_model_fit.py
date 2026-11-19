"""Smoke test for patch 122kq — fitting a rigged body model to video (Body model tab).

User: "you generate a mufasa tab that uses the algorithm you used to fit, and if requested,
render the overlay? Can you use the frame history forwards and backwards to refine the
positioning?"

A synthetic rig with the mouse model's bone names (pelvis, 17 spine, 8 neck, head, 23 tail
bones, femurs, hind paws; a tube-shaped skin with ears) walks, turns and bends through a
synthetic top-down video; its 2-D labels carry noise, dropouts and a few gross head-label
errors. Checks:

* kinematics: rest pose = rest mesh; joint limits hold; rearing keeps the hind paws down;
  the local quaternions match the joint rotations;
* markers: placed along the model at the labels' own spacing (scale recovered);
* masks: the animal is segmented against the median background; implausible masks dropped;
* the fit follows the animal (position, heading, spine and head bends);
* forwards + backwards (the frame history) beats fitting frames on their own, most on
  frames with bad labels;
* fit_video in parallel chunks, params_at between fitted frames, the overlay video,
  run_body_fit (run folder and run.toml), the forms, the CLI, and (with bpy) the .blend
  export.
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


W_IMG, H_IMG, S_TRUE = 960, 540, 3.0


def frame_of(y_axis):
    y = y_axis / np.linalg.norm(y_axis)
    x = np.cross(y, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.c_[x, y, z]


def make_rig(path: Path):
    bones = []           # name, parent, head, tail

    def add(name, parent, h, t):
        bones.append((name, parent, np.array(h, float), np.array(t, float)))
    add("Pelvis", None, (0, 25, 10), (0, 17, 20))
    sp = np.linspace([0, 17, 20], [0, -25, 30], 18)
    for i in range(17):
        add(f"Spine{i + 1}", "Pelvis" if i == 0 else f"Spine{i}", sp[i], sp[i + 1])
    nk = np.linspace([0, -25, 30], [0, -34, 32], 9)
    for i in range(8):
        add(f"Neck{i + 1}", "Spine17" if i == 0 else f"Neck{i}", nk[i], nk[i + 1])
    add("Head", "Neck8", (0, -34, 32), (0, -56, 33))
    tl = np.c_[np.zeros(24), np.linspace(25, 90, 24), np.linspace(10, 1.5, 24)]
    for i in range(23):
        add(f"Tail{i + 1}", "Pelvis" if i == 0 else f"Tail{i}", tl[i], tl[i + 1])
    for side, sx in (("L", 1), ("R", -1)):
        add(f"Femur.{side}", "Pelvis", (6 * sx, 21, 15), (12 * sx, 1, 12))
        for k in range(1, 6):
            add(f"BPhalange{k}.{side}", f"Femur.{side}", (8 * sx + k * 0.5 * sx, 9, 1), (8 * sx + k * 0.5 * sx, 5, 0.5))
    names = [b[0] for b in bones]
    ix = {n: i for i, n in enumerate(names)}
    parent = np.array([ix[b[1]] if b[1] else -1 for b in bones])
    rest = np.zeros((len(bones), 4, 4))
    for i, (_, _, h, t) in enumerate(bones):
        rest[i] = np.eye(4)
        rest[i, :3, :3] = frame_of(t - h)
        rest[i, :3, 3] = h
    tails = np.array([b[3] for b in bones])
    # skin: a tube along nose -> head -> neck -> spine -> pelvis -> tail, plus two ears
    chain = [((0, -58, 30), "Head", 1.5)]
    for t in np.linspace(1, 0, 6)[:-1]:
        chain.append(((0, -56 + 22 * (1 - t), 33 - 1 * (1 - t)), "Head", 3 + 6 * (1 - t)))
    for i in range(8, 0, -1):
        chain.append((tuple(nk[i - 1]), f"Neck{i}", 8.5))
    for i in range(17, 0, -1):
        r = 11 + 2 * math.sin(math.pi * (17 - i) / 16)
        chain.append((tuple(sp[i - 1]), f"Spine{i}", r))
    chain.append(((0, 25, 12), "Pelvis", 7))
    for i in range(23):
        chain.append((tuple(tl[i + 1]), f"Tail{i + 1}", 1.6 - 1.1 * i / 22))
    V, Wl, F = [], [], []
    ring = 12
    for k, (c, b, r) in enumerate(chain):
        c = np.array(c, float)
        for j in range(ring):
            a = 2 * math.pi * j / ring
            p = c + np.array([r * math.cos(a), 0, 0.8 * r * math.sin(a)])
            if b.startswith(("Spine", "Neck", "Pelvis", "Head")):
                p[2] = max(p[2], 2.5)
            V.append(p)
            Wl.append(b)
        if k:
            o, q = (k - 1) * ring, k * ring
            for j in range(ring):
                j2 = (j + 1) % ring
                F += [(o + j, q + j, q + j2), (o + j, q + j2, o + j2)]
    for sx in (1, -1):
        c0 = len(V)
        centre = np.array([8.5 * sx, -34, 44])
        V.append(centre)
        Wl.append("Head")
        for j in range(10):
            a = 2 * math.pi * j / 10
            V.append(centre + np.array([3.5 * math.cos(a), 3.5 * math.sin(a), -1.0]))
            Wl.append("Head")
        for j in range(10):
            F.append((c0, c0 + 1 + j, c0 + 1 + (j + 1) % 10))
    V = np.array(V)
    W = np.zeros((len(V), len(names)), np.float32)
    for v, b in enumerate(Wl):
        W[v, ix[b]] = 1.0
    F = np.array(F)
    rgb = np.full((len(F), 3), 45, np.uint8)
    tail_faces = np.array([Wl[f[0]].startswith("Tail") for f in F])
    rgb[tail_faces] = 215                       # the tail is pale in IR, as on real footage
    np.savez_compressed(path, names=np.array(names), parent=parent, rest=rest, tails=tails, V=V, F=F, W=W,
                        Vd=V, Fd=F, Wd=W, face_rgb=rgb, meta="{}")
    return path


def truth(n):
    from mufasa.bodymodel.fit import neutral
    out = []
    for k in range(n):
        t = k / max(1, n - 1)
        out.append(neutral(tx=300 + 340 * t, ty=260 + 60 * math.sin(2 * t), phi=0.3 + 0.9 * t, s=S_TRUE,
                           lum_yaw=3.5 * math.sin(3 * t), tho_yaw=3 * math.sin(3 * t + 0.5),
                           cer_yaw=-2.5 * math.sin(2 * t), head_yaw=18 * math.sin(4 * t),
                           tail_yaw_p=10 * math.sin(2 * t), tail_yaw_d=-8, head_pitch=5))
    return out


def make_video(rig, path: Path, poses, rng):
    import cv2

    from mufasa.bodymodel.render import OverlayStyle, draw_model
    bg = np.full((H_IMG, W_IMG, 3), 205, np.uint8)
    for _ in range(900):                    # bedding-like bright flakes
        x, y = rng.integers(0, W_IMG), rng.integers(0, H_IMG)
        g = int(rng.integers(170, 250))
        cv2.rectangle(bg, (int(x), int(y)), (int(x + rng.integers(4, 14)), int(y + rng.integers(4, 14))), (g, g, g), -1)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W_IMG, H_IMG))
    style = OverlayStyle(alpha=1.0, outline=False, labels=False, info=False)
    for p in poses:
        wr.write(draw_model(bg, rig, p, style))
    wr.release()


def make_labels(rig, markers, poses, path: Path, rng, bad_frames):
    import pandas as pd

    from mufasa.bodymodel.fit import to_image
    cols = {}
    for k, p in enumerate(poses):
        mp = markers.positions(rig, rig.skinning_matrices(p))
        for n, X in mp.items():
            uv = to_image(X[None], p)[0] + rng.normal(0, 2.0, 2)
            lk = 0.95 if rng.random() > 0.12 else 0.2
            if k in bad_frames and n.startswith("head"):
                uv = uv + np.array([45.0, -35.0])           # a confident but wrong head
                lk = 0.9
            cols.setdefault(n + "_x", []).append(uv[0])
            cols.setdefault(n + "_y", []).append(uv[1])
            cols.setdefault(n + "_p", []).append(lk)
    pd.DataFrame(cols).to_parquet(path, index=False)
    return path


def errors(res, poses, rig, markers):
    """Body points (the markers on the model, in px), head direction and heading errors."""
    from mufasa.bodymodel.fit import to_image

    def pts(p):
        mp = markers.positions(rig, rig.skinning_matrices(p))
        names = [n for n in mp if not n.startswith("tail_V1") and not n.startswith("tail_V3")]
        return to_image(np.array([mp[n] for n in names]), p)
    pos = np.array([np.linalg.norm(pts(r.params) - pts(t), axis=1).mean() for r, t in zip(res, poses)])
    head = np.array([abs(r.params["head_yaw"] + r.params["cer_yaw"] * 8 - t["head_yaw"] - t["cer_yaw"] * 8)
                     for r, t in zip(res, poses)])
    phi = np.array([abs((r.params["phi"] - t["phi"] + math.pi) % (2 * math.pi) - math.pi) for r, t in zip(res, poses)])
    return pos, head, np.degrees(phi)


def main() -> int:
    import cv2

    from mufasa.bodymodel import fit as BF
    from mufasa.bodymodel.render import render_overlay_video
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rig = BF.Rig(make_rig(td / "syn.rig.npz"))
        # ---- kinematics
        G0 = rig.skinning_matrices(BF.neutral())
        check("rest pose reproduces the rest mesh", np.allclose(BF.Rig.lbs(G0, rig.V, rig.W), rig.V, atol=1e-9))
        p = BF.neutral(rear=40)
        G = rig.skinning_matrices(p)
        paw = (G[rig.ix["BPhalange3.L"]] @ np.r_[rig.heads[rig.ix["BPhalange3.L"]], 1])[:3]
        nose = BF.Rig.lbs(G, rig.V, rig.W)[rig.V[:, 1].argmin()]
        check(f"rearing turns the body about the hind paws (paw z {paw[2]:.1f}, nose z {nose[2]:.0f})",
              abs(paw[2] - rig.heads[rig.ix["BPhalange3.L"]][2]) < 1.0 and nose[2] > 50)
        q = rig.local_quaternions(BF.neutral(lum_yaw=4, head_pitch=-20))
        Rr = rig.rest[rig.ix["Head"], :3, :3]
        w, x, y, z = q["Head"]
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        check("local quaternions (as Blender pose bones) match the joint rotation",
              np.allclose(Rr @ Rq @ Rr.T, BF.rot("x", -20 * BF.DEG), atol=1e-9) and len(q) >= 17 + 8 + 1)
        check("limits documented with sources",
              "Dewan" in BF.LIMITS["lumbar (Spine1-6)"] and "hind paws" in BF.LIMITS["body"])

        # ---- data
        n = 24
        poses = truth(n)
        # the video goes on: the animal elsewhere in the arena (for the median background)
        from mufasa.bodymodel.fit import neutral
        extra = [neutral(tx=float(rng.uniform(100, 860)), ty=float(rng.uniform(100, 440)),
                         phi=float(rng.uniform(-3, 3)), s=S_TRUE) for _ in range(40)]
        make_video(rig, td / "syn.avi", poses + extra, rng)
        markers_true = BF.Markers.build(rig, None)
        bad = {6, 13, 19}
        make_labels(rig, markers_true, poses + extra, td / "syn.parquet", rng, bad)
        labels = BF.load_labels(td / "syn.parquet")
        mk = BF.Markers.build(rig, labels)
        check(f"marker spacing from the labels gives the scale ({mk.px_per_unit:.2f} vs {S_TRUE})",
              abs(mk.px_per_unit - S_TRUE) / S_TRUE < 0.08)
        bg = BF.background(td / "syn.avi", list(range(n + len(extra))))
        cap = cv2.VideoCapture(str(td / "syn.avi"))
        ok, img = cap.read()
        cap.release()
        exp = BF.silhouette_area(rig, S_TRUE)
        m, area = BF.animal_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg, BF.MaskConfig(), exp)
        check(f"animal segmented against the median background (area {area} vs body {exp:.0f})",
              m is not None and 0.7 < area / exp < 1.6)
        m2, _ = BF.animal_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg, BF.MaskConfig(area_range=(5, 9)), exp)
        check("implausible mask (size) is not used", m2 is None)

        # ---- per-frame vs forwards + backwards
        cfg_alone = BF.FitConfig(scale=S_TRUE, temporal=0.0, refine=False, evals_first=1200, evals_warm=350)
        cfg_fb = BF.FitConfig(scale=S_TRUE, temporal=0.5, refine=True, evals_first=1200, evals_warm=350)
        r_alone = BF.SequenceFitter(rig, td / "syn.avi", labels, cfg_alone, bg=bg).fit(range(n), say=lambda s: None)
        r_fb = BF.SequenceFitter(rig, td / "syn.avi", labels, cfg_fb, bg=bg).fit(range(n), say=lambda s: None)
        pa, ha, fa = errors(r_alone, poses, rig, markers_true)
        pb, hb, fb = errors(r_fb, poses, rig, markers_true)
        check(f"fit follows the animal (forwards+backwards: body points {np.median(pb):.1f} px, heading "
              f"{np.median(fb):.1f}°, median IoU {np.median([1 - r.parts.get('sil', 1) for r in r_fb]):.2f})",
              np.median(pb) < 6 and np.median(fb) < 6 and np.median([1 - r.parts["sil"] for r in r_fb]) > 0.75
              and all("sil" in r.parts for r in r_fb))
        bl = sorted(bad)
        check(f"frame history helps where labels are wrong: head error on bad frames "
              f"{np.mean(ha[bl]):.1f}° alone -> {np.mean(hb[bl]):.1f}°",
              np.mean(hb[bl]) < np.mean(ha[bl]))
        jit_a = np.mean(np.abs(np.diff([r.params["tx"] for r in r_alone], 2)))
        jit_b = np.mean(np.abs(np.diff([r.params["tx"] for r in r_fb], 2)))
        check(f"smoother over time (x jitter {jit_a:.2f} -> {jit_b:.2f} px/frame²) and not worse overall "
              f"(body points {np.mean(pa):.1f} -> {np.mean(pb):.1f} px)", jit_b < jit_a and np.mean(pb) <= np.mean(pa) + 0.5)
        check("all postures within the joint limits",
              all(lo - 1e-6 <= r.params[nm] <= hi + 1e-6 for r in r_fb for nm, lo, hi in BF.POSTURE))

        # ---- whole video in chunks, scale estimated, overlay
        run = BF.RunConfig(start=0, stop=n, step=2, workers=2, chunk=6, overlap=2,
                           fit=BF.FitConfig(scale=None, evals_first=1000, evals_warm=300))
        out = BF.fit_video(td / "syn.rig.npz", td / "syn.avi", td / "syn.parquet", run, td / "out" / "syn_fit.parquet",
                           say=lambda s: None)
        df, meta = BF.read_fit(out)
        check(f"fit_video: every 2nd frame, two workers, scale estimated ({meta['scale_px_per_unit']:.2f})",
              list(df["frame"]) == list(range(0, n, 2)) and abs(meta["scale_px_per_unit"] - S_TRUE) / S_TRUE < 0.1
              and "limits" in meta)
        p5 = BF.params_at(df, 5)
        check("params_at interpolates between fitted frames",
              p5 is not None and min(df["tx"].iloc[2], df["tx"].iloc[3]) <= p5["tx"] <= max(df["tx"].iloc[2],
                                                                                           df["tx"].iloc[3]))
        ov = render_overlay_video(out, td / "out" / "syn_overlay.avi", say=lambda s: None)
        c = cv2.VideoCapture(str(ov))
        check("overlay video, every frame of the range", int(c.get(cv2.CAP_PROP_FRAME_COUNT)) == n - 1)  # 0..22
        c.release()

        # ---- the tab
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.utils.config_creator import ProjectConfigCreator
        cfgp = ProjectConfigCreator(project_path=str(td), project_name="p", target_list=[],
                                    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1,
                                    file_type="parquet", body_parts=["head_nose", "back_L2"]).config_path
        root = Path(cfgp).parent
        (root / "sources" / "videos").mkdir(parents=True, exist_ok=True)
        (root / "sources" / "pose").mkdir(parents=True, exist_ok=True)
        (root / "sources" / "bodymodel").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(td / "syn.avi", root / "sources" / "videos" / "syn.avi")
        shutil.copy(td / "syn.parquet", root / "sources" / "pose" / "syn.parquet")
        shutil.copy(td / "syn.rig.npz", root / "sources" / "bodymodel" / "syn.rig.npz")
        from mufasa.ui_qt.forms.bodymodel import BodyFitForm, RigImportForm, run_body_fit
        from mufasa.ui_qt.pages.bodymodel_page import build_bodymodel_page  # noqa: F401
        f = BodyFitForm(config_path=cfgp)
        f.stop.setValue(6)
        f.step.setValue(3)
        f.render.setChecked(True)
        args = f.collect_args()
        check("form: rig, video and labels found; range and options collected",
              args["rig"].endswith("syn.rig.npz") and args["pose"].endswith("syn.parquet") and args["stop"] == 6
              and args["step"] == 3 and args["render"] and args["refine"])
        rd = run_body_fit(root, **{**args, "workers": 1}, evals_first=600, evals_warm=200)
        check("run_body_fit: derived/bodymodel/<run>/ with the fit, the overlay and run.toml",
              rd.parent.name == "bodymodel" and (rd / "syn_fit.parquet").is_file()
              and (rd / "syn_overlay.mp4").is_file() and (rd / "run.toml").is_file())
        rf = RigImportForm(config_path=cfgp)
        try:
            rf.collect_args()
            ok = False
        except ValueError as e:
            ok = ".blend" in str(e)
        check("rig import form asks for a .blend", ok)
        from mufasa.section_provenance import SECTIONS
        check("sections registered on the Body model page",
              SECTIONS["bodymodel_fit"].page == "Body model" and "bodymodel_rig" in SECTIONS["bodymodel_fit"].depends_on)
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "body-fit", str(td / "syn.rig.npz"),
                            str(td / "syn.avi"), str(td / "syn.parquet"), "--out", str(td / "cli_fit.parquet"),
                            "--stop", "4", "--scale", str(S_TRUE)], capture_output=True, text=True, cwd=REPO,
                           timeout=900)
        check(f"CLI body-fit ({r.returncode})", r.returncode == 0 and (td / "cli_fit.parquet").is_file())
        if r.returncode:
            print(r.stderr[-1500:])

        # ---- .blend export (only where Blender's Python module is installed)
        has_bpy = subprocess.run([sys.executable, "-c", "import bpy"], capture_output=True).returncode == 0
        if has_bpy:
            script = td / "mk.py"
            script.write_text(
                "import bpy,sys\n"
                "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
                "a=bpy.data.armatures.new('Armature');o=bpy.data.objects.new('Armature',a)\n"
                "bpy.context.scene.collection.objects.link(o);bpy.context.view_layer.objects.active=o\n"
                "bpy.ops.object.mode_set(mode='EDIT')\n"
                "b=a.edit_bones.new('Head');b.head=(0,0,1);b.tail=(0,-2,1)\n"
                "c=a.edit_bones.new('Neck1');c.head=(0,2,1);c.tail=(0,0,1);b.parent=c\n"
                "bpy.ops.object.mode_set(mode='OBJECT')\n"
                "bpy.ops.mesh.primitive_cube_add(size=1,location=(0,-1,1));s=bpy.context.object;s.name='Skin'\n"
                "g=s.vertex_groups.new(name='Head');g.add(list(range(8)),1.0,'REPLACE')\n"
                "m=s.modifiers.new('a','ARMATURE');m.object=o\n"
                "bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])\n")
            subprocess.run([sys.executable, str(script), str(td / "t.blend")], capture_output=True, timeout=300)
            from mufasa.ui_qt.forms.bodymodel import export_rig
            rp = export_rig(td / "t.blend", td / "t.rig.npz", say=lambda s: None)
            rt = BF.Rig(rp)
            check("rig from a .blend (bpy in a separate process): bones parents-first, skin and weights",
                  rt.names == ["Neck1", "Head"] and len(rt.V) == 8 and np.allclose(rt.W[:, 1], 1))
        else:
            check("rig from a .blend: skipped (bpy not installed)", True)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kq_body_model_fit: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
