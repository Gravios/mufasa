"""Smoke test for patch 122lm — smoothing 3-D labels with the body
(``mufasa-pose3d smooth --model``).

User: "does the 3d kalman filter also generate a model based on kinematics?" (it does not: one
constant-velocity filter per marker); "Skeletal smoother".

A synthetic rat (the virtual rodent walking and turning its head), its labels with noise, a gap,
two labels swapped for a stretch, one wild label:

* the body's labels are nearer the truth than the noisy ones, and than the per-marker Kalman
  smoother's;
* the head's labels keep their distances (one bone);
* the gap is filled, the swap and the wild label are flagged and do not drag the body;
* the joint angles are saved;
* ``mufasa-pose3d smooth --model`` (and ``mufasa run smooth --model``).
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


def main() -> int:
    try:
        import mujoco
    except ImportError:
        check("MuJoCo not installed: skipped (pip install \"mufasa[biomech]\")", True)
        return report()
    import pandas as pd

    from mufasa.biomech.mjmodel import build
    from mufasa.biomech.smoother import smooth_kinematics
    from mufasa.pose3d.io import read_pose3d, write_pose3d
    from mufasa.pose3d.kalman3d import Kalman3DConfig, smooth_markers

    M = build(1.0)
    m = M.model
    names = list(M.labels)
    q = np.zeros(m.nq)
    q[3] = 1.0
    for n, v in zip(M.hinge, M.q_stance, strict=True):
        q[M.qadr[n]] = v
    T, fps = 120, 30.0
    d = M.data()
    truth = np.zeros((T, len(names), 3))
    Qt = np.tile(q, (T, 1))
    for t in range(T):
        Qt[t, 0] += 0.1 * t / fps
        Qt[t, M.qadr["atlas"]] += 0.3 * np.sin(t / 8.0)
        Qt[t, M.qadr["vertebra_cervical_1_bend"]] += 0.2 * np.sin(t / 11.0)
        Qt[t, M.qadr["vertebra_1_extend"]] += 0.1 * np.sin(t / 15.0)
        d.qpos[:] = Qt[t]
        mujoco.mj_kinematics(m, d)
        truth[t] = d.site_xpos[[M.labels[n] for n in names]] * 1000
    rng = np.random.default_rng(0)
    obs = truth + rng.normal(0, 3.0, truth.shape)
    gap = (slice(40, 52), names.index("head_left"))
    obs[gap] = np.nan
    a, b = names.index("hip_left"), names.index("hip_right")
    obs[70:80, [a, b]] = obs[70:80, [b, a]]                       # swapped for 10 frames
    wild = (95, names.index("back_T8"))
    obs[wild] += np.array([60.0, -40.0, 30.0])                    # one wild label

    out = smooth_kinematics(M, names, obs, fps, window=60, step=50, iters=20, label_sigma=4.0)
    L = out["labels_mm"]
    good = np.ones((T, len(names)), bool)
    good[70:80, [a, b]] = False
    good[wild] = False
    e_body = np.nanmedian(np.linalg.norm(L - truth, axis=2)[5:-5])
    e_obs = np.nanmedian(np.linalg.norm(obs - truth, axis=2)[good])
    kal, _ = smooth_markers(obs, Kalman3DConfig(fps=fps))
    e_kal = np.nanmedian(np.linalg.norm(kal - truth, axis=2)[5:-5])
    check(f"the body's labels: {e_body:.2f} mm from the truth (median), the noisy labels {e_obs:.2f}, "
          f"the per-marker Kalman {e_kal:.2f}", e_body < 0.7 * e_obs and e_body < e_kal)
    i, j = names.index("head_nose"), names.index("head_back")
    sd_body = np.nanstd(np.linalg.norm(L[:, i] - L[:, j], axis=1))
    sd_kal = np.nanstd(np.linalg.norm(kal[:, i] - kal[:, j], axis=1))
    check(f"one bone: the nose–back-of-head distance varies by {sd_body:.3f} mm (sd), the Kalman's {sd_kal:.2f} mm",
          sd_body < 0.01 and sd_kal > 0.5)
    eg = np.linalg.norm(L[gap] - truth[gap], axis=1)
    check(f"the 12-frame gap filled, {eg.max():.1f} mm from the truth at worst", np.isfinite(eg).all() and eg.max() < 5)
    fl = out["flag"]
    es = np.linalg.norm(L[70:80][:, [a, b]] - truth[70:80][:, [a, b]], axis=2)
    check(f"the swapped hips flagged in {fl[70:80][:, [a, b]].mean():.0%} of their frames, the wild label "
          f"{'flagged' if fl[wild] else 'not flagged'}; elsewhere {fl[good].mean():.2%} flagged; the body's hips "
          f"through the swap {es.max():.1f} mm from the truth",
          fl[70:80][:, [a, b]].mean() >= 0.8 and fl[wild] and fl[good].mean() < 0.01 and es.max() < 8)
    st = np.abs(np.diff(out["qpos"][2:-2, 7:], axis=0))
    tr = np.abs(np.diff(Qt[:, 7:], axis=0))
    check(f"joints move smoothly: at most {np.degrees(st.max()):.1f}° between frames (the truth: "
          f"{np.degrees(tr.max()):.1f}°)", np.degrees(st.max()) < 4)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = pd.DataFrame({"frame": np.arange(T)})
        for k, n in enumerate(names):
            for c, ax in enumerate("xyz"):
                df[f"{n}_{ax}"] = obs[:, k, c]
        write_pose3d(df, td / "s_3d.parquet", {"fps": fps})
        from mufasa.cli.workbench_launcher import main as mufasa
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = mufasa(["run", "smooth", str(td / "s_3d.parquet"), "--out", str(td / "s_body.parquet"), "--model",
                         "--calibrate", "60", "--label-sigma", "4", "--joints-out", str(td / "s_joints.parquet")])
        sm, meta = read_pose3d(td / "s_body.parquet")
        jt = pd.read_parquet(td / "s_joints.parquet")
        check(f"mufasa run smooth --model: exit {rc}; the labels, <label>_model_mm and _flag, the joints "
              f"({len([c for c in jt if c.startswith('q_')])}) saved; '{buf.getvalue().strip().splitlines()[-1][:70]}…'",
              rc == 0 and "head_left_model_mm" in sm and "hip_left_flag" in sm and sm["head_left_x"].notna().all()
              and "with the body" in meta.get("method", "") and len([c for c in jt if c.startswith("q_")]) == 67)

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs/pose3d.md: smoothing with the body", "--model" in doc and "with the body" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lm_skeletal_smoother: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
