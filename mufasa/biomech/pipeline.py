"""
mufasa.biomech.pipeline
=======================

Patch 122li: a 3-D pose file and its gait → joint torques and ground forces, frame by frame.

1. The model scaled to the animal (:func:`mufasa.pose3d.gait.animal_size`, as the gait).
2. Its labels calibrated on frames spread over the whole recording (:func:`.fit.calibrate`).
3. Every joint fitted, frame by frame, to the labels and the gait's paws (:func:`.fit.fit_trajectory`).
4. Velocities, accelerations, inverse dynamics, ground forces split as the gait splits the load
   (:mod:`.dynamics`).

Writes ``<video>_dynamics.parquet``: per frame the fit's error, each paw's force (N), the
residual at the root, and per joint its angle (deg) and net torque (mN·m); the summary and the
calibrated label sites in the file's metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mufasa.pose3d.gait import PAWS


def run_dynamics(pose3d, gait=None, out=None, *, frames=None, fps: float | None = None,
                 mass_g: float | None = None, calibrate_frames: int = 400, muscles=None,
                 optimise: bool = False, iters: int = 40, limits: str = "model", animal=None,
                 progress=None) -> tuple[Path, dict]:
    """Joint torques for a 3-D pose file (and its gait file: the paws' targets and load split;
    without one, paws planted). ``frames``: (first, last) frame numbers, inclusive (default all).
    ``muscles``: an OpenSim hindlimb model (``.osim``), a saved muscle set (``.json``) or a
    :class:`.osim.MuscleSet`: the hindlimb muscles' forces too (122lj). ``optimise``: then the
    motion, the paws' forces and the activations re-solved together over windows of frames, for
    tracking, balance and effort (:mod:`.optimise`, 122lk; about a second a frame).
    ``limits``: "model" (the virtual rodent's ranges, default) or "rat" (the rat's knees and
    ankles, narrower). ``animal``: an animal file (:mod:`.animal`): its size, mass, limits and label
    places, and no calibration here (122ln).
    Returns (the file written, its summary)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mufasa.biomech import dynamics as D
    from mufasa.biomech import fit as F
    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import BodyPhysics, animal_size, read_gait
    from mufasa.pose3d.io import marker_names, read_pose3d
    from mufasa.pose3d.leg_ik import GaitTargets

    pose3d = Path(pose3d)
    df, meta = read_pose3d(pose3d)
    names = marker_names(df)
    xyz = np.stack([df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names], 1)
    allf = df["frame"].to_numpy(int) if "frame" in df else np.arange(len(df))
    ms = None
    if muscles is not None:
        from mufasa.biomech.osim import MuscleSet, load_muscles
        ms = muscles if isinstance(muscles, MuscleSet) else load_muscles(muscles)
    say = progress or (lambda *_: None)
    if animal is not None:
        from mufasa.biomech.animal import build_for, load_animal
        an = animal if isinstance(animal, dict) else load_animal(animal)
        M = build_for(an, muscles=ms, **({"mass_kg": mass_g / 1000.0} if mass_g else {}))
        size = M.size
        say(f"animal {an.get('name')}: size {size:.3f}, labels as calibrated on {an.get('frames')} frames")
        cal = {"rms_mm": an.get("calibration_rms_mm", []), "shift_mm": an.get("label_shift_mm", {})}
    else:
        size = animal_size(BodyPhysics.virtual_rodent(), names, xyz)
        M = build(size=size, mass_kg=None if not mass_g else mass_g / 1000.0, muscles=ms, limits=limits)
        good = np.flatnonzero(np.isfinite(xyz).all((1, 2)))
        if len(good) < 20:
            raise ValueError(f"{pose3d.name}: under 20 frames with every label; nothing to calibrate on")
        rng = np.random.default_rng(0)
        pick = rng.choice(good, min(calibrate_frames, len(good)), replace=False)
        say(f"calibrating the labels on {len(pick)} frames")
        cal = F.calibrate(M, names, xyz[pick])

    sel = np.arange(len(df))
    if frames is not None:
        a, b = frames
        sel = np.flatnonzero((allf >= a) & (allf <= b))
    targets, shares, anchors = None, None, None
    if gait is not None:
        gf, gmeta = read_gait(gait)
        tg = GaitTargets(gf, size=size)
        targets = [tg.at(int(f)) for f in allf[sel]]
        g = gf.set_index("frame")
        shares = g.reindex(allf[sel])[[f"{p}_load" for p in PAWS]].to_numpy(float)
        if all(f"{p}_observed" in g for p in PAWS) and g[[f"{p}_observed" for p in PAWS]].to_numpy().any():
            gs = g.reindex(allf[sel])                       # 122lo: seen paws anchor stage 4
            anchors = np.full((len(sel), 4, 2), np.nan)
            for k, p in enumerate(PAWS):
                on = gs[f"{p}_observed"].fillna(False).to_numpy(bool) & (gs[f"{p}_contact"].to_numpy() == 1)
                anchors[on, k] = gs.loc[on, [f"{p}_x", f"{p}_y"]].to_numpy(float) / 1000.0
        fps = fps or (gmeta.get("summary") or {}).get("fps")
    fps = float(fps or meta.get("fps") or 30.0)
    say(f"fitting {len(sel)} frames")
    fit = F.fit_trajectory(M, names, xyz[sel], targets,
                           progress=(lambda t, n: say(f"fitting: {t}/{n}")) if progress else None)
    Q, V, A = D.derivatives(M, fit["qpos"], fps)
    inv = D.inverse(M, Q, V, A, fit["modes"], shares)
    so = None
    if ms is not None:
        from mufasa.biomech import muscle as MU
        say(f"muscle forces: {len(ms.muscles)} muscles")
        so = MU.static_optimisation(M, Q, V, inv["tau"])
    rms = fit["rms_mm"]
    before = None
    if optimise:
        from mufasa.biomech import optimise as OP
        before = D.summarise(M, D.dynamics_table(M, allf[sel], Q, rms, inv))
        if so is not None:
            before.update(MU.summarise_muscles(so))
        say(f"optimising {len(sel)} frames in windows of {OP.WINDOW}")
        op = OP.optimise_trajectory(M, names, xyz[sel], Q, fit["modes"], fps, F0=inv["grf"],
                                    A0=None if so is None else so["act"], muscles=so is not None, iters=iters,
                                    anchors=anchors,
                                    progress=(lambda i, n: say(f"optimising: window {i}/{n}")) if progress else None)
        Q, V, A, rms = op["qpos"], op["V"], op["A"], op["rms_mm"]
        inv = {"tau": op["tau"], "grf": op["grf"], "res_f": op["res_f"], "res_t": op["res_t"], "down": op["down"]}
        for k in range(4):
            inv["grf"][~op["down"][:, k], k] = 0.0
        if so is not None:
            so = OP.muscle_outputs(M, Q, V, op["act"], op["tau"])
    table = D.dynamics_table(M, allf[sel], Q, rms, inv)
    table = _with_paws(M, table, Q)
    summ = D.summarise(M, table)
    if so is not None:
        import pandas as pd
        table = pd.concat([table, pd.DataFrame(MU.muscle_columns(so))], axis=1)
        summ.update(MU.summarise_muscles(so))
        a = so["act"]
        summ["activation_change_per_frame"] = round(float(np.nanmean(np.abs(np.diff(a, axis=0)))), 4)
    if before is not None:
        summ["before_optimising"] = before
    summ.update({"size": round(float(size), 4), "fps": fps, "limits": M.limits,
                 "animal": (an.get("name") if animal is not None else None), "calibration_rms_mm": [round(x, 1) for x in cal["rms_mm"]],
                 "label_shift_mm": cal["shift_mm"], "gait": str(gait) if gait else None})
    out = Path(out) if out else pose3d.with_name(pose3d.stem.replace("_3d", "") + "_dynamics.parquet")
    sites = {n: M.model.site_pos[s].tolist() for n, s in M.labels.items()}
    tbl = pa.Table.from_pandas(table, preserve_index=False)
    md = dict(tbl.schema.metadata or {})
    extra = {}
    if ms is not None:
        extra = {"muscles": [m.name for m in ms.muscles], "muscle_source": ms.source,
                 "f_max_N": dict(zip([m.name for m in ms.muscles], so["f_max"].round(3).tolist(), strict=True))}
    md[b"mufasa_dynamics"] = json.dumps({"summary": summ, "label_sites_m": sites, "joints": M.hinge, **extra,
                                         "source": str(pose3d),
                                         "method": "virtual rodent (MuJoCo), labels calibrated (STAC), "
                                                   "inverse dynamics, ground forces at planted paws"
                                                   + ("; optimised over windows for tracking, balance and "
                                                      "effort" if optimise else "")}).encode()
    pq.write_table(tbl.replace_schema_metadata(md), out)
    return out, summ


def _with_paws(M, table, Q):
    """Each paw's sole (the middle of its two sole points), mm, per frame."""
    from mufasa.biomech.mjmodel import mujoco
    mj, d = mujoco(), M.data()
    P = np.full((len(Q), 4, 3), np.nan)
    for t in range(len(Q)):
        if not np.isfinite(Q[t]).all():
            continue
        d.qpos[:] = Q[t]
        mj.mj_kinematics(M.model, d)
        for k, p in enumerate(PAWS):
            P[t, k] = d.site_xpos[list(M.soles[p])].mean(0) * 1000.0
    for k, p in enumerate(PAWS):
        for c, ax in enumerate("xyz"):
            table[f"{p}_paw_{ax}_mm"] = P[:, k, c]
    return table


def read_dynamics(path):
    """(table, metadata) of a ``_dynamics.parquet``."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return t.to_pandas(), json.loads((t.schema.metadata or {}).get(b"mufasa_dynamics", b"{}"))


__all__ = ["read_dynamics", "run_dynamics"]
