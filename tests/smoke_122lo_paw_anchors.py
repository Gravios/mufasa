"""Smoke test for patch 122lo — labelled paws anchor the gait and the biomechanical model.

User: "Would it help if I labeled the paws when visible as anchor points for the gait
extrapolation?"; "Paw-label anchors".

A synthetic rat gliding forward (the virtual rodent's labels) with four labelled paws stepping
in a walk (each planted, then lifted and carried a stride forward), one paw hidden half the time:

* the paw names are recognised (several spellings);
* where a paw is seen, the gait's stances are the seen ones, planted where they were seen;
  where it is not, the gait infers it as before;
* stage 4 (``dynamics --optimise``) keeps the seen planted paws where they were seen.
"""
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def walk(T=150, fps=30.0, speed=100.0):
    """Body labels from the virtual rodent gliding at ``speed`` mm/s, and four paws stepping."""
    import mujoco

    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import PAWS
    M = build(1.0)
    m = M.model
    q = np.zeros(m.nq)
    q[3] = 1.0
    for n, v in zip(M.hinge, M.q_stance, strict=True):
        q[M.qadr[n]] = v
    d = M.data()
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    q[2] -= min(d.site_xpos[list(s)][:, 2].min() for s in M.soles.values())
    names = list(M.labels)
    X = np.zeros((T, len(names), 3))
    home = {}
    for t in range(T):
        qq = q.copy()
        qq[0] += speed * t / fps / 1000.0
        d.qpos[:] = qq
        mujoco.mj_kinematics(m, d)
        X[t] = d.site_xpos[[M.labels[n] for n in names]] * 1000
        if t == 0:
            home = {p: d.site_xpos[list(M.soles[p])].mean(0) * 1000 for p in PAWS}
    period, duty = 18, 0.7                                    # frames a stride; stance share
    offs = {"LH": 0.0, "RF": 0.1, "RH": 0.5, "LF": 0.6}
    stride = speed * period / fps
    paws = np.zeros((T, 4, 3))
    stance = np.zeros((T, 4), bool)
    for k, p in enumerate(PAWS):
        for t in range(T):
            ph = (t / period + offs[p]) % 1.0
            n = np.floor(t / period + offs[p])
            base = home[p] + np.array([stride * n - stride * offs[p], 0, 0])
            if ph < duty:
                paws[t, k] = base
                stance[t, k] = True
            else:
                s = (ph - duty) / (1 - duty)
                paws[t, k] = base + np.array([stride * s, 0, 8 * np.sin(np.pi * s)])
        paws[:, k, 2] = np.maximum(paws[:, k, 2], home[p][2])
    return M, names, X, paws, stance


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        check("MuJoCo not installed: skipped (pip install \"mufasa[biomech]\")", True)
        return report()
    import pandas as pd

    from mufasa.pose3d import gait as G
    from mufasa.pose3d.io import write_pose3d

    check("paw names: paw_LF, forepaw_right, LEFT_HINDPAW, hindpaw_R recognised; others not",
          G.paw_labels(["head_nose", "paw_LF", "forepaw_right", "LEFT_HINDPAW", "hindpaw_R", "paw"])
          == {"LF": 1, "RF": 2, "LH": 3, "RH": 4})

    M, names, X, paws, stance = walk()
    T, fps = len(X), 30.0
    pnames = ["paw_LF", "paw_RF", "hindpaw_left", "hindpaw_right"]
    hide = np.zeros(T, bool)
    hide[T // 2:] = True                                       # LF unseen in the second half
    P = paws.copy()
    P[hide, 0] = np.nan
    allx = np.concatenate([X, P], 1)
    alln = names + pnames
    res = G.estimate_gait(alln, allx, np.arange(T), fps)
    ref = G.estimate_gait(names, X, np.arange(T), fps)
    f = res.frames
    seen = f["LF_observed"].to_numpy()
    # the frame either side of a lift-off or touch-down is either (the paw has not moved yet)
    clear = np.ones((T, 4), bool)
    edge = np.flatnonzero(np.diff(stance.astype(int), axis=0).any(1))
    for e in edge:
        for k in range(4):
            if stance[e, k] != stance[e + 1, k]:
                clear[max(e - 1, 0):e + 3, k] = False
    clear[:5] = clear[-5:] = False
    agree = {p: float(((f[f"{p}_contact"].to_numpy() == 1) == stance[:, k])[clear[:, k]].mean())
             for k, p in enumerate(G.PAWS)}
    agree_ref = {p: float(((ref.frames[f"{p}_contact"].to_numpy() == 1) == stance[:, k])[clear[:, k]].mean())
                 for k, p in enumerate(G.PAWS)}
    check(f"stance or swing as seen, frame by frame (away from the frame either side of a lift-off or touch-down): {', '.join(f'{p} {v:.0%}' for p, v in agree.items() if p != 'LF')} "
          f"(inferred without paw labels: {', '.join(f'{p} {v:.0%}' for p, v in agree_ref.items() if p != 'LF')})",
          all(agree[p] > 0.97 for p in ("RF", "LH", "RH")))
    on = (f["RH_contact"].to_numpy() == 1) & stance[:, 3]
    err = np.linalg.norm(f.loc[on, ["RH_x", "RH_y"]].to_numpy() - paws[on, 3, :2], axis=1)
    err_ref = np.linalg.norm(ref.frames.loc[on, ["RH_x", "RH_y"]].to_numpy() - paws[on, 3, :2], axis=1)
    check(f"planted where seen: RH {np.median(err):.1f} mm from the truth (inferred: {np.nanmedian(err_ref):.1f} mm)",
          np.median(err) < 2.0 and np.median(err) < np.nanmedian(err_ref))
    c1 = clear[:, 0].copy()
    c1[T // 2:] = False
    a1 = float(((f["LF_contact"].to_numpy() == 1) == stance[:, 0])[c1].mean())
    check(f"LF seen in the first half only: {seen.mean():.0%} of frames (summary {res.summary['LF_seen']}); "
          f"as seen there ({a1:.0%}); inferred after (contacts {np.isin(f['LF_contact'].to_numpy()[T // 2:], (0, 1)).mean():.0%} placed)",
          abs(seen.mean() - 0.5) < 0.02 and a1 > 0.9 and abs(res.summary["LF_seen"] - 0.5) < 0.02
          and not f["LF_observed"].to_numpy()[T // 2:].any())

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = pd.DataFrame({"frame": np.arange(T)})
        for j, n in enumerate(alln):
            for c, ax in enumerate("xyz"):
                df[f"{n}_{ax}"] = allx[:, j, c]
        write_pose3d(df, td / "w_3d.parquet", {"fps": fps})
        gp, _ = G.write_gait(res, td / "w_gait.parquet")
        from mufasa.tools import pose3d_cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["dynamics", str(td / "w_3d.parquet"), "--gait", str(gp), "--calibrate", "40",
                                  "--frames", "20:59", "--optimise", "--iters", "25"])
        from mufasa.biomech.pipeline import read_dynamics
        tab, meta = read_dynamics(td / "w_dynamics.parquet")
        fr = tab["frame"].to_numpy()
        down = tab["LH_down"].to_numpy(bool) & stance[fr, 2]
        e = np.linalg.norm(tab.loc[down, ["LH_paw_x_mm", "LH_paw_y_mm"]].to_numpy() - paws[fr[down], 2, :2], axis=1)
        check(f"stage 4 keeps the seen planted paws where they were seen: LH within {np.median(e):.1f} mm (median), "
              f"{np.max(e):.1f} worst; exit {rc}", rc == 0 and np.median(e) < 4.0)

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs/pose3d.md: labelled paws", "paw_LF" in doc and "Labelled paws" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lo_paw_anchors: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
