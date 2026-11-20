"""Smoke test for patch 122kz — the pose file contract, in one place (audit findings 1, 6 and 10).

User: "Start on it" (the audit's first patch: the pose file contract).

* one rule finds markers (``pose_markers``): ``<m>_var`` columns are skipped when ``<m>`` is a
  marker (the Kalman v2 smoother's variances), a marker whose own name ends in ``_var`` is kept;
* every reader uses it: the single-camera 2-D loader, the 3-D viewer (and its extent), the 3-D
  features, gait, and the body model's 2-D labels — a 3-D file with phantom markers far away
  reads as its real markers, without being rebuilt;
* ``check_pose3d`` names what breaks the contract: no frame rate, frames out of order or
  missing, markers below the floor, units, derived columns; gait reports it and says when it
  assumed 30 fps;
* C3D import: ``floor_z`` moves the floor to z = 0; markers below the floor are reported.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def table_3d(T=200, phantoms=True, seed=0):
    """A 3-D pose table of a rat-shaped marker set walking, with Kalman-style phantom columns."""
    rng = np.random.default_rng(seed)
    names = ["head_nose", "head_mid", "head_back", "head_left", "head_right", "back_T4", "back_T8",
             "back_L2", "back_L6", "back_V2", "hip_left", "hip_right", "tail_V6"]
    base = {"head_nose": (90, 0, 45), "head_mid": (70, 0, 55), "head_back": (55, 0, 60),
            "head_left": (62, 12, 58), "head_right": (62, -12, 58), "back_T4": (30, 0, 70),
            "back_T8": (5, 0, 72), "back_L2": (-20, 0, 70), "back_L6": (-45, 0, 62),
            "back_V2": (-75, 0, 45), "hip_left": (-55, 18, 40), "hip_right": (-55, -18, 40),
            "tail_V6": (-100, 0, 25)}
    cols = {"frame": np.arange(T)}
    for n in names:
        p = np.array(base[n], float)
        for k, a in enumerate("xyz"):
            v = p[k] + (150.0 * np.arange(T) / 30 if a == "x" else 0) + rng.normal(0, 1, T)
            cols[f"{n}_{a}"] = v
        cols[f"{n}_likelihood"] = np.ones(T)
        if phantoms:                                     # as the old reconstruction wrote them
            for a in "xyz":
                cols[f"{n}_var_{a}"] = rng.normal(0, 1, T) * (2e6 if a != "z" else 0)
            cols[f"{n}_var_likelihood"] = np.ones(T)
    cols["lumbar_angle_deg"] = np.zeros(T)
    return pd.DataFrame(cols), names


def main() -> int:
    from mufasa.pose3d.io import check_pose3d, marker_names, pose_markers, write_pose3d

    # the rule
    cols2 = ["frame", "nose_x", "nose_y", "nose_p", "nose_var_x", "nose_var_y", "ear_var_x", "ear_var_y", "tail_x"]
    cols3 = ["a_x", "a_y", "a_z", "a_var_x", "a_var_y", "a_var_z", "b_x", "b_y"]
    check(f"rule: 2-D {pose_markers(cols2, 2)}, 3-D {pose_markers(cols3, 3)} (variances of a marker skipped; "
          "'ear_var' kept, no 'ear' marker; 'b' not 3-D)",
          pose_markers(cols2, 2) == ["nose", "ear_var"] and pose_markers(cols3, 3) == ["a"])

    # the single-camera 2-D loader, on a Kalman v2 smoothed 2-D table
    from mufasa.pose3d.mono import load_pose2d
    T = 60
    d2 = {}
    for m in ("head_nose", "head_back"):
        d2[m + "_x"], d2[m + "_y"], d2[m + "_p"] = np.linspace(10, 50, T), np.full(T, 20.0), np.ones(T)
        d2[m + "_var_x"], d2[m + "_var_y"] = np.full(T, 0.5), np.full(T, 0.5)
    p2 = load_pose2d(pd.DataFrame(d2))
    check(f"single-camera 2-D loader: markers {p2.names} (the smoother's variances are not markers)",
          p2.names == ["head_nose", "head_back"])

    # the body model's 2-D labels
    from mufasa.bodymodel import fit as BF
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pd.DataFrame(d2).to_parquet(td / "v_smoothed.parquet", index=False)   # as the smoother writes it
        try:
            lab = BF.load_labels(td / "v_smoothed.parquet")
            check(f"body model 2-D labels: {sorted(lab)}", sorted(lab) == ["head_back", "head_nose"])
        except Exception as e:  # noqa: BLE001
            check(f"body model 2-D labels: {type(e).__name__}: {e}", False)

        # every 3-D reader, on a file written with phantom markers
        df, names = table_3d()
        path = write_pose3d(df, td / "v_3d.parquet", {"units": "mm", "method": "test"})
        from mufasa.ui_qt.pose3d_viewer import load_pose3d
        pose = load_pose3d(path)
        ext = pose.extent()
        from mufasa.feature_extractors.features_3d import load_pose3d_aligned
        fn, fxyz, _ = load_pose3d_aligned(path, len(df))
        check(f"3-D readers see {len(marker_names(df))} markers (file: {len(names)} real + {len(names)} phantoms): "
              f"io, viewer {len(pose.names)}, features {len(fn)}",
              marker_names(df) == names and pose.names == names and fn == names)
        check(f"viewer extent within metres ({[tuple(round(v) for v in e) for e in ext]}), not set by phantoms",
              all(abs(v) < 2000 for e in ext for v in e))

        # the contract check
        w = check_pose3d(df, {"units": "mm"})
        check("check: no frame rate; the derived column sets named; nothing else here",
              any(s.startswith("no frame rate") for s in w) and any("derived column sets" in s for s in w)
              and len(w) == 2)
        bad = df.copy()
        bad.loc[50:, "frame"] += 5
        low = df.copy()
        for n in names:
            low[n + "_z"] -= 60
        w2 = check_pose3d(bad, {"fps": 30}, names)
        w3 = check_pose3d(low, {"fps": 30, "units": "m"}, names)
        rev = df.iloc[::-1].reset_index(drop=True)
        w4 = check_pose3d(rev, {"fps": 30}, names)
        check(f"check: gaps ({w2[0] if w2 else '—'}); below the floor and units ({len(w3) - 1}); order ({w4[0] if w4 else '—'})",
              "1 gaps in the frames" in w2 and any("below the floor" in s for s in w3)
              and any("units" in s for s in w3) and "frames out of order or repeated" in w4)

        # gait reports it
        from mufasa.pose3d.gait import gait_from_pose3d
        r = gait_from_pose3d(path)
        r2 = gait_from_pose3d(path, fps=25.0)
        c1, c2 = r.summary["contract"], r2.summary["contract"]
        check(f"gait: says it assumed 30 fps ({[s for s in c1 if 'fps' in s][:1]}); given the rate, it does not",
              any("assumed 30 fps" in s for s in c1) and not any("fps" in s or "frame rate" in s for s in c2)
              and r.summary["fps"] == 30.0 and r2.summary["fps"] == 25.0)

        # C3D: the floor
        from smoke_122kl_c3d_import import write_c3d

        from mufasa.pose3d.c3d import import_c3d
        from mufasa.pose3d.io import read_pose3d
        TT = 150
        t = np.arange(TT) / 100
        X = np.stack([np.c_[100 * np.sin(t + k), 80 * np.cos(t + k), 30 + 15 * k + 0 * t] for k in range(4)], 1)
        X[..., 2] += 400.0                                      # this lab's floor is at 400 mm
        lab4 = ["head_nose", "head_back", "back_T4", "back_L6"]
        write_c3d(td / "trial.c3d", X, np.zeros((TT, 4), bool), lab4)
        X2 = X.copy()
        X2[..., 2] -= 500.0                                     # a floor above the markers
        write_c3d(td / "below.c3d", X2, np.zeros((TT, 4), bool), lab4)
        said = []
        out = import_c3d(td / "trial.c3d", td / "mocap", floor_z=400.0, say=said.append)
        o2 = import_c3d(td / "below.c3d", td / "mocap", say=said.append)
        d1, m1 = read_pose3d(out)
        d2_, m2 = read_pose3d(o2)
        z1 = d1[[f"{n}_z" for n in lab4]].to_numpy()
        check(f"C3D: floor moved to z = 0 (heights {z1.min():.0f}-{z1.max():.0f} mm); a floor above the markers "
              f"is reported ({said[:1]})",
              abs(z1.min() - 30) < 1 and m1.get("floor_z_moved_mm") == 400.0 and m1["report"]["contract"] == []
              and any("below the floor" in s for s in m2["report"]["contract"]) and len(said) == 1)

    # C3D form and CLI
    from mufasa.tools.pose3d_cli import build_parser
    a = build_parser().parse_args(["import-c3d", "x.c3d", "--out", "o", "--floor-z", "12.5"])
    src = (REPO / "mufasa" / "ui_qt" / "forms" / "pose3d.py").read_text()
    check("C3D form and CLI: the floor height (--floor-z)", a.floor_z == 12.5 and "Floor height in the file" in src)
    # no copies of the rule left in the 3-D readers
    copies = []
    for f in ["mufasa/ui_qt/pose3d_viewer.py", "mufasa/feature_extractors/features_3d.py", "mufasa/pose3d/pipeline.py",
              "mufasa/pose3d/mono.py", "mufasa/pose3d/gait.py", "mufasa/bodymodel/fit.py"]:
        if 'endswith("_x")' in (REPO / f).read_text():
            copies.append(f)
    check(f"one rule: no reader keeps its own copy ({copies or 'none left'})", not copies)
    io_doc = (REPO / "mufasa" / "pose3d" / "io.py").read_text()
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("contract written down: millimetres, z up, floor at z = 0, fps (io.py and docs)",
          "floor at z = 0" in io_doc and "fps" in io_doc and "## The 3-D pose file" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kz_pose_file_contract: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
