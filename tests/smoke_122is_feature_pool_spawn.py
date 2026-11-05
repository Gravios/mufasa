"""Smoke test for patch 122is — parallel feature extraction survives OpenMP.

User report: Compute Features with 16 workers → every video failed with
"BrokenProcessPool: A process in the process pool was terminated abruptly".
The workers had been forked (Linux default) from a process that had already
run numba-parallel kernels, i.e. had live GNU OpenMP threads; such a child
aborts ("fork() called from a process already using GNU OpenMP, this is
unsafe"). The pool now starts workers with "spawn".

Runs for real: OpenMP threads started in this process first (as Mufasa's
kernels do), then FeatureSubsetsCalculator with 2 workers; its output is
compared with a sequential run.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _project(n_videos: int):
    import cv2
    import numpy as np
    import pandas as pd

    from mufasa.mixins.config_reader import ConfigReader
    from mufasa.project_layout import project_paths_from_config
    from mufasa.utils.config_creator import ProjectConfigCreator

    bps = ["nose", "ear_l", "ear_r", "center", "tail"]
    tmp = Path(tempfile.mkdtemp())
    cfg = ProjectConfigCreator(
        project_path=str(tmp), project_name="pp", target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=bps,
    ).config_path
    P = project_paths_from_config(cfg)
    vd = Path(P["video_dir"])
    vd.mkdir(parents=True, exist_ok=True)
    names = [f"v{i}" for i in range(n_videos)]
    for v in names:
        w = cv2.VideoWriter(str(vd / f"{v}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
        for _ in range(3):
            w.write(np.zeros((360, 640, 3), np.uint8))
        w.release()
    pd.DataFrame({"Video": names, "fps": 30, "Resolution_width": 640,
                  "Resolution_height": 360, "Distance_in_mm": 100,
                  "pixels/mm": 2.0}).to_csv(P["video_info_path"], index=False)
    cr = ConfigReader(config_path=cfg, read_video_info=False)
    os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for v in names:
        pd.DataFrame({f"{b}_{c}": (rng.uniform(0, 600, 400) if c != "p" else 0.99)
                      for b in bps for c in "xyp"}).to_parquet(
            os.path.join(cr.outlier_corrected_dir, f"{v}.parquet"))
    return cfg, P


def _run(cfg, P, n_workers: int):
    import mufasa.feature_extractors.feature_subsets as FS
    fams = [FS.TWO_POINT_BP_DISTANCES, FS.ANIMAL_CONVEX_HULL_AREA,
            FS.FRAME_BP_MOVEMENT]
    FS.FeatureSubsetsCalculator(
        config_path=cfg, feature_families=fams, file_checks=False,
        save_dir=None, data_dir=None,
        derived_features_dir=P["derived_features_dir"], n_workers=n_workers,
        overwrite_existing=True).run()


def main() -> int:
    import numba
    import numpy as np
    import pandas as pd

    checks: list[tuple[str, bool]] = []

    def check(name, ok):
        checks.append((name, bool(ok)))

    # Start OpenMP worker threads in this process, as Mufasa's parallel
    # numba kernels do before the pool is created.
    @numba.njit(parallel=True)
    def _warm(a):
        s = 0.0
        for i in numba.prange(a.shape[0]):
            s += a[i]
        return s

    _warm(np.ones(100_000))
    layer = numba.threading_layer()

    cfg, P = _project(4)
    err = ""
    try:
        _run(cfg, P, n_workers=2)
    except Exception as exc:
        err = repr(exc)[:200]
    check(f"parallel run succeeds after OpenMP/numba threads started "
          f"(threading layer: {layer})", err == "")
    out = sorted(Path(P["derived_features_dir"]).rglob("*.parquet"))
    check("every video × family written", len(out) == 4 * 3)
    par = {p.relative_to(P["derived_features_dir"]): pd.read_parquet(p) for p in out}

    _run(cfg, P, n_workers=1)
    seq = {p.relative_to(P["derived_features_dir"]): pd.read_parquet(p)
           for p in sorted(Path(P["derived_features_dir"]).rglob("*.parquet"))}
    check("parallel output equals the sequential output",
          par.keys() == seq.keys()
          and all(par[k].equals(seq[k]) for k in par))

    src = (REPO / "mufasa/feature_extractors/feature_subsets.py").read_text()
    check("the pool uses the spawn start method",
          'mp_context=_mp.get_context("spawn")' in src)

    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"smoke_122is_feature_pool_spawn: {n_pass}/{len(checks)} checks passed")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
