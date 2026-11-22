"""Smoke test for patch 122lg — a spine that goes from hunched to stretched.

User: "the spine doesn't have enough flexture, It should be able to go from hunched to stretched
more easily".

On K7, the back's path from head to tail base spans 162–291 mm (5–95 %), but the fitted model's
stayed at 173–184 mm: the virtual rodent's thorax is one rigid body and its bones keep their
lengths, so bending barely changes its length. The marker error grew with the stretch (34 mm on the
longest tenth). And the lumbar limits were the wrong way round: the model's extension joints
flex +45° and extend −30°, the rig allowed +10° / −15° per joint, so hunched rats sat at the
flexion limit in 27 % of frames.

* lumbar ±15° per joint: 90° of flexion as the model allows (and extension past the model's, for
  its rigid thorax);
* ``stretch`` −0.3…+0.4: each lumbar and cervical joint and the atlas slid along its bone from
  its parent's, so the spine and neck are 30 % shorter … 40 % longer; the pelvis, hind legs and
  tail stay put; the per-frame and batched fits move the joints alike;
* the fit recovers a hunched and a stretched rat, which it could not with the length fixed.
"""
from __future__ import annotations

import os
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


CHAIN = ["head_nose", "head_back", "back_T4", "back_T8", "back_L2", "back_L6", "back_V2"]


def path_len(pos):
    return sum(float(np.linalg.norm(np.asarray(pos[b]) - np.asarray(pos[a]))) for a, b in zip(CHAIN[:-1], CHAIN[1:]))


def main() -> int:
    import pandas as pd

    import mufasa.bodymodel.fit as BF
    import mufasa.bodymodel.rodent_rig as RR
    from mufasa.pose3d import rig_batch
    from mufasa.pose3d import scene_models as SM
    from mufasa.pose3d.rodent_model import RodentModel
    from mufasa.ui_qt.pose3d_viewer import load_pose3d

    # the limits, against the virtual rodent's own joints
    m = RodentModel()
    ext = [k for k, j in enumerate(m.joints) if j.name in ("vertebra_1_extend", "vertebra_4_extend")]
    flex_total = float(np.degrees(m.hi[ext]).sum())
    rig = BF.Rig(SM.builtin_rig_path())
    post = dict((n, (lo, hi)) for n, lo, hi in rig.posture)
    n_lum = len(rig.groups["lum"])
    check(f"lumbar flexion {post['lum_pitch'][1]}° × {n_lum} joints = the model's {flex_total:.0f}° (it was 10°, the "
          f"model's extension); extension {post['lum_pitch'][0]}°; stretch {post['stretch']}; rig v{RR.RIG_VERSION}",
          post["lum_pitch"][1] * n_lum == flex_total and post["lum_pitch"][0] == -15 and post["stretch"] == RR.STRETCH
          and RR.RIG_VERSION >= 3 and SM.builtin_rig_path().name == f"virtual_rodent_v{RR.RIG_VERSION}.rig.npz")

    mk = BF.Markers.build(rig, None)

    def markers(**kw):
        return mk.positions(rig, rig.skinning_matrices(BF.neutral(**kw)))
    rest, hunch = markers(), markers(lum_pitch=15)
    check(f"+ flexes: lumbar +15° lowers the nose {rest['head_nose'][2] - hunch['head_nose'][2]:.0f} mm "
          "(the hips stay)", hunch["head_nose"][2] < rest["head_nose"][2] - 20
          and np.allclose(hunch["hip_left"], rest["hip_left"]))

    lens = {k: path_len(markers(stretch=k)) for k in (RR.STRETCH[0], 0.0, RR.STRETCH[1])}
    short, long_ = markers(stretch=RR.STRETCH[0]), markers(stretch=RR.STRETCH[1])
    fixed = all(np.allclose(short[n], rest[n]) and np.allclose(long_[n], rest[n])
                for n in ("hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32"))
    check(f"stretch: nose to tail base {lens[RR.STRETCH[0]]:.0f} … {lens[0.0]:.0f} … {lens[RR.STRETCH[1]]:.0f} mm "
          f"({lens[RR.STRETCH[0]] / lens[0.0]:.2f}× … {lens[RR.STRETCH[1]] / lens[0.0]:.2f}×); pelvis, hips and tail "
          "unmoved", 0.75 < lens[RR.STRETCH[0]] / lens[0.0] < 0.85 and 1.2 < lens[RR.STRETCH[1]] / lens[0.0] < 1.35
          and fixed)

    old = BF.Rig.__new__(BF.Rig)
    old.__dict__.update(rig.__dict__)
    old.stretch_offset = np.zeros_like(rig.stretch_offset)
    check("a rig without stretch bones (the HBP mouse) ignores stretch",
          np.allclose(old.skinning_matrices(BF.neutral(stretch=0.3)), old.skinning_matrices(BF.neutral())))

    # synthetic rats: hunched (short, flexed) and stretched (long, extended), then fitted
    # (hunched: the back arched up, the front raised on the hind paws, the neck lifted; stretched:
    # long and low, the back extended and the neck bent down level)
    poses = [dict(lum_pitch=13.0, cer_pitch=-8.0, stretch=-0.22, head_pitch=-20.0, rear=20.0, tail_pitch=4.0),
             dict(lum_pitch=-8.0, cer_pitch=6.0, stretch=0.3, head_pitch=-5.0, rear=-5.0, tail_pitch=0.0)]
    from mufasa.pose3d.rig_floor import trunk_vertices
    tv = trunk_vertices(rig)

    def trunk_low(p):
        G = rig.skinning_matrices(BF.neutral(**p))
        return float(BF.Rig.lbs(G, rig.verts(BF.neutral(**p), True), rig.Wd)[tv, 2].min())
    lows = [min(trunk_low(p), min(v[2] for v in markers(**p).values())) for p in poses]
    lens2 = [path_len(markers(**p)) for p in poses]
    check(f"the test's rats: hunched {lens2[0]:.0f} mm, stretched {lens2[1]:.0f} mm nose to tail base, both above "
          f"the floor, trunk and markers ({lows[0]:.0f}, {lows[1]:.0f} mm)", min(lows) > 0 and lens2[1] > 1.4 * lens2[0])
    T = 40

    def truth_pose(i):
        """Frames 0–19: hunched to stretched; 20–39: one, then the other, every frame."""
        if i < 20:
            a = i / 19.0
            return {k: (1 - a) * poses[0][k] + a * poses[1][k] for k in poses[0]}
        return dict(poses[i % 2 == 0 and 1 or 0])
    rows = []
    for i in range(T):
        pos = mk.positions(rig, rig.skinning_matrices(BF.neutral(**truth_pose(i))))
        P = SM._yaw(1.2 * np.array(list(pos.values())), 0.4) + np.array([3.0 * i, 0.0, 0.0])
        row = {"frame": i}
        for nm, X in zip(pos, P):
            row.update({f"{nm}_x": X[0], f"{nm}_y": X[1], f"{nm}_z": X[2]})
        rows.append(row)
    truth = [truth_pose(i)["stretch"] for i in range(T)]
    with tempfile.TemporaryDirectory() as td:
        pd.DataFrame(rows).to_parquet(Path(td) / "s_3d.parquet", index=False)
        pose = load_pose3d(Path(td) / "s_3d.parquet")
        posed = SM.load_model("virtual_rodent", pose, size=1.2)
        si = [n for n, _, _ in posed.posture].index("stretch")
        frames = [0, 19, 20, 21, 30, 31]
        fits = {f: posed.fit_frame(f) for f in frames}
        err = max(h[1] for h in fits.values())
        s_err = max(abs(fits[f][0][4 + si] - truth[f]) for f in frames)
        check(f"per-frame fit: hunched and stretched rats posed within {err:.1f} mm, stretch within {s_err:.3f}",
              err < 3.0 and s_err < 0.05)
        stiff = SM.load_model("virtual_rodent", pose, size=1.2)
        stiff.posture = [(n, -1e-9, 1e-9) if n == "stretch" else (n, lo, hi) for n, lo, hi in stiff.posture]
        err0 = max(stiff.fit_frame(f)[1] for f in frames)
        check(f"with the length fixed (as before), the same rats are off by {err0:.1f} mm", err0 > 3 * max(err, 1.0))
        if rig_batch.available():
            import torch
            bat = SM.load_model("virtual_rodent", pose, size=1.2)
            bat.fit_frames(range(T), device="cpu")
            errb = max(bat.fit_frame(f)[1] for f in range(T))
            sb = max(abs(bat.fit_frame(f)[0][4 + si] - truth[f]) for f in range(T))
            prob = rig_batch.RigProblem(bat, "cpu")
            x = torch.tensor(bat.fit_frame(25)[0])
            P, _, _ = bat._markers_world(x.numpy())
            par = float(np.abs(prob.markers(x).numpy() - P).max())
            check(f"batched fit: every frame within {errb:.1f} mm, stretch within {sb:.3f}; batched = per-frame "
                  f"kinematics ({par:.0e} mm)", errb < 3.0 and sb < 0.05 and par < 1e-9)
        else:
            check("batched fit: PyTorch not installed (skipped)", True)

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: stretch and the lumbar range", "stretch" in doc and "hunched" in doc and "±15°" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lg_spine_stretch: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
