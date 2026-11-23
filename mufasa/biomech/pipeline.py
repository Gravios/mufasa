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
                 progress=None) -> tuple[Path, dict]:
    """Joint torques for a 3-D pose file (and its gait file: the paws' targets and load split;
    without one, paws planted). ``frames``: (first, last) frame numbers, inclusive (default all).
    ``muscles``: an OpenSim hindlimb model (``.osim``), a saved muscle set (``.json``) or a
    :class:`.osim.MuscleSet`: the hindlimb muscles' forces too (122lj).
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
    size = animal_size(BodyPhysics.virtual_rodent(), names, xyz)
    ms = None
    if muscles is not None:
        from mufasa.biomech.osim import MuscleSet, load_muscles
        ms = muscles if isinstance(muscles, MuscleSet) else load_muscles(muscles)
    M = build(size=size, mass_kg=None if not mass_g else mass_g / 1000.0, muscles=ms)
    say = progress or (lambda *_: None)

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
    targets, shares = None, None
    if gait is not None:
        gf, gmeta = read_gait(gait)
        tg = GaitTargets(gf, size=size)
        targets = [tg.at(int(f)) for f in allf[sel]]
        g = gf.set_index("frame")
        shares = g.reindex(allf[sel])[[f"{p}_load" for p in PAWS]].to_numpy(float)
        fps = fps or (gmeta.get("summary") or {}).get("fps")
    fps = float(fps or meta.get("fps") or 30.0)
    say(f"fitting {len(sel)} frames")
    fit = F.fit_trajectory(M, names, xyz[sel], targets,
                           progress=(lambda t, n: say(f"fitting: {t}/{n}")) if progress else None)
    Q, V, A = D.derivatives(M, fit["qpos"], fps)
    inv = D.inverse(M, Q, V, A, fit["modes"], shares)
    table = D.dynamics_table(M, allf[sel], Q, fit["rms_mm"], inv)
    summ = D.summarise(M, table)
    if ms is not None:
        import pandas as pd

        from mufasa.biomech import muscle as MU
        say(f"muscle forces: {len(ms.muscles)} muscles")
        so = MU.static_optimisation(M, Q, V, inv["tau"])
        table = pd.concat([table, pd.DataFrame(MU.muscle_columns(so))], axis=1)
        summ.update(MU.summarise_muscles(so))
    summ.update({"size": round(float(size), 4), "fps": fps, "calibration_rms_mm": [round(x, 1) for x in cal["rms_mm"]],
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
                                                   "inverse dynamics, ground forces at planted paws"}).encode()
    pq.write_table(tbl.replace_schema_metadata(md), out)
    return out, summ


def read_dynamics(path):
    """(table, metadata) of a ``_dynamics.parquet``."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return t.to_pandas(), json.loads((t.schema.metadata or {}).get(b"mufasa_dynamics", b"{}"))


__all__ = ["read_dynamics", "run_dynamics"]
