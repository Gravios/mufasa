"""
mufasa.biomech.smoother
=======================

Patch 122lm: smoothing 3-D labels with the body (``mufasa-pose3d smooth --model``).

The 3-D Kalman smoother (:mod:`mufasa.pose3d.kalman3d`) smooths each label on its own: a
constant-velocity model per marker, with no skeleton. Bones can stretch, joints can bend the
wrong way, and two labels swapped between frames go unseen.

Here the labels are explained by the virtual rodent instead (:mod:`.mjmodel`, its label sites
calibrated as in :func:`.fit.calibrate`):

1. every joint fitted frame by frame (:func:`.fit.fit_trajectory`);
2. then whole stretches of frames re-solved together (:func:`.optimise.optimise_window`
   without physics):
   * the labels (σ :data:`LABEL_SIGMA_MM`), robustly: a label further off than
     :data:`ROBUST` σ is down-weighted (Huber), so one bad label does not drag the body;
   * every joint's and the root's second difference (smooth motion; σ :data:`SMOOTH_RAD`, firmer
     than stage 4's, which has physics to hold the motion);
   * a weak pull to the stance, and every joint within its range;
   * the paws above the floor.
   Windows of :data:`WINDOW` frames, each holding the one before's last two (as in stage 4).

Out come the model's labels, in every frame (gaps filled), on a body whose bones keep their
lengths and whose joints stay in range. Each label also gets its distance from the model
(``<label>_model_mm``) and a flag where it disagreed with the body (``<label>_flag``: an
outlier, or a swap), and the joint angles can be saved too.
"""
from __future__ import annotations

import numpy as np

from mufasa.biomech.mjmodel import RodentMJ, mujoco

LABEL_SIGMA_MM = 8.0
ROBUST = 2.5
WINDOW, STEP = 150, 130
SMOOTH_RAD = 0.03          # σ of a joint's second difference, rad a frame² (stage 4 uses 0.1)
FLAG_SIGMA = 4.0            # a label this many σ off the body (and 3× its own median) is flagged


def smooth_kinematics(M: RodentMJ, names, xyz_mm, fps: float, *, window: int = WINDOW, step: int = STEP,
                      iters: int = 20, label_sigma: float = LABEL_SIGMA_MM, robust: float | None = ROBUST,
                      smooth_rad: float = SMOOTH_RAD, progress=None) -> dict:
    """``xyz_mm`` (T, L, 3) labels (NaN unseen) → {"qpos" (T, nq), "labels_mm" (T, L, 3) the
    model's, "err_mm" (T, L) each seen label's distance from it, "flag" (T, L)}. Frames before
    the first with four labels, and after the last, stay unfitted (NaN)."""
    from mufasa.biomech import fit as F
    from mufasa.biomech import optimise as OP
    xyz_mm = np.asarray(xyz_mm, float)
    T = len(xyz_mm)
    say = progress or (lambda *_: None)
    say(f"fitting {T} frames")
    fit = F.fit_trajectory(M, names, xyz_mm, None)
    Q0 = fit["qpos"]
    ok = np.isfinite(Q0).all(1)
    # unfitted frames inside: carried from the last fitted one (the smoother bridges them)
    first = int(np.argmax(ok)) if ok.any() else T
    last = int(T - 1 - np.argmax(ok[::-1])) if ok.any() else -1
    for t in range(first + 1, last + 1):
        if not ok[t]:
            Q0[t] = Q0[t - 1]
    out = {"qpos": np.full_like(Q0, np.nan), "labels_mm": np.full((T, len(names), 3), np.nan),
           "err_mm": np.full((T, len(names)), np.nan), "flag": np.zeros((T, len(names)), bool)}
    if last - first + 1 < 3:
        return out
    sl = slice(first, last + 1)
    n = last - first + 1
    say(f"smoothing {n} frames in windows of {window}")
    op = OP.optimise_trajectory(M, names, xyz_mm[sl], Q0[sl], [{}] * n, fps, muscles=False,
                                window=window, step=step, iters=iters, physics=False, label_sigma=label_sigma,
                                robust=robust, smooth_rad=smooth_rad, progress=progress and (lambda i, k: say(f"smoothing: window {i}/{k}")))
    Q = op["qpos"]
    out["qpos"][sl] = Q
    fitter = F.PoseFitter(M, names)
    col = [list(names).index(k) for k in fitter.names]
    mj, d = mujoco(), M.data()
    L = np.full((n, len(names), 3), np.nan)
    for t in range(n):
        if not np.isfinite(Q[t]).all():
            continue
        d.qpos[:] = Q[t]
        mj.mj_kinematics(M.model, d)
        L[t][col] = d.site_xpos[fitter.sites] * 1000.0
    out["labels_mm"][sl] = L
    err = np.linalg.norm(L - xyz_mm[sl], axis=2)
    out["err_mm"][sl] = err
    limit = np.maximum(FLAG_SIGMA * label_sigma, 3.0 * np.nanmedian(err, axis=0))[None]
    out["flag"][sl] = np.nan_to_num(err > limit, nan=0).astype(bool)
    return out


def smooth_pose3d(path, out, *, fps: float | None = None, calibrate_frames: int = 400, joints_out=None,
                  progress=None, **kw) -> dict:
    """A 3-D pose file smoothed with the body: the model's labels in place of the observed
    ones, with ``<label>_model_mm`` and ``<label>_flag``; ``joints_out``: the joint angles too
    (``q_<joint>_deg``, the root's position mm and orientation). Returns a summary."""
    from pathlib import Path

    import pandas as pd

    from mufasa.biomech import fit as F
    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import BodyPhysics, animal_size
    from mufasa.pose3d.io import marker_names, read_pose3d, write_pose3d
    df, meta = read_pose3d(path)
    names = marker_names(df)
    xyz = np.stack([df[[f"{k}_x", f"{k}_y", f"{k}_z"]].to_numpy(float) for k in names], 1)
    fps = float(fps or meta.get("fps") or 30.0)
    size = animal_size(BodyPhysics.virtual_rodent(), names, xyz)
    M = build(size)
    use = [k for k in names if k in M.labels]
    if len(use) < 4:
        raise ValueError(f"{Path(path).name}: under 4 labels the virtual rodent has ({use})")
    iu = [names.index(k) for k in use]
    good = np.flatnonzero(np.isfinite(xyz[:, iu]).all((1, 2)))
    say = progress or (lambda *_: None)
    if len(good) >= 20 and calibrate_frames:
        pick = np.random.default_rng(0).choice(good, min(calibrate_frames, len(good)), replace=False)
        say(f"calibrating the labels on {len(pick)} frames")
        F.calibrate(M, use, xyz[pick][:, iu])
    res = smooth_kinematics(M, use, xyz[:, iu], fps, progress=progress, **kw)
    new = df.copy()
    for j, k in enumerate(use):
        for c, ax in enumerate("xyz"):
            new[f"{k}_{ax}"] = res["labels_mm"][:, j, c]
        new[f"{k}_model_mm"] = res["err_mm"][:, j]
        new[f"{k}_flag"] = res["flag"][:, j]
    meta = dict(meta)
    meta["method"] = (meta.get("method", "") + "; smoothed with the body (virtual rodent, joints and root "
                      "smooth, robust to bad labels)").lstrip("; ")
    write_pose3d(new, out, meta)
    if joints_out:
        Q = res["qpos"]
        cols = {"frame": df["frame"].to_numpy() if "frame" in df else np.arange(len(df))}
        for c, ax in enumerate("xyz"):
            cols[f"root_{ax}_mm"] = Q[:, c] * 1000.0
        for c, ax in enumerate("wxyz"):
            cols[f"root_q{ax}"] = Q[:, 3 + c]
        for n in M.hinge:
            cols[f"q_{n}_deg"] = np.degrees(Q[:, M.qadr[n]])
        pd.DataFrame(cols).to_parquet(joints_out, index=False)
    E = res["err_mm"]
    seen = np.isfinite(xyz[:, iu]).all(2)
    moved = np.linalg.norm(res["labels_mm"] - xyz[:, iu], axis=2)
    return {"labels": use, "size": round(float(size), 3), "frames": int(np.isfinite(res["qpos"]).all(1).sum()),
            "model_mm_median": round(float(np.nanmedian(E)), 1),
            "flagged": round(float(res["flag"][seen].mean()), 4) if seen.any() else 0.0,
            "filled": int((~seen & np.isfinite(res["labels_mm"]).all(2)).sum()),
            "moved_mm_median": round(float(np.nanmedian(moved)), 1)}


__all__ = ["LABEL_SIGMA_MM", "ROBUST", "smooth_kinematics", "smooth_pose3d"]
