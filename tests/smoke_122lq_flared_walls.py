"""Smoke test for patch 122lq — cage walls that flare outward; the Rat7M height-model builder per
rat and in parallel.

User: "I think the walls of the cage are angled outward by 10-15 degrees"; "There are about 100
of those files over six rats".

* a straight-down camera over a box whose walls flare 12°: its corner edges, marked, give the
  flare back (``--wall-tilt auto``) and the point below the camera, where taking them for
  vertical walls misplaces it;
* a given flare is kept in the box and saved with the camera;
* the box's limits widen with height: a head against a flaring wall is no longer pushed in
  (single-camera limits, and the GPU solver's);
* the 3-D file carries the flare, the viewer reads it and draws the box flaring;
* ``mufasa-pose3d calibrate-box --wall-tilt``; the calibration form's *Walls flare outward*;
* ``scripts/build_rat7m_prior.py --per-rat --pattern --workers``.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def main() -> int:
    from mufasa.pose3d.camera import Camera, box_xy_at, calibrate_from_box_marks
    H, L, W, WALL, TILT = 500.0, 400.0, 300.0, 250.0, 12.0
    true_p = np.array([331.0, 186.0])
    cam = Camera.straight_down(true_p, 0.6 * H, H, (640, 360), center_xy=(8.0, -5.0))
    t = np.tan(np.radians(TILT))
    floor = np.array([[-L / 2, W / 2, 0], [L / 2, W / 2, 0], [L / 2, -W / 2, 0], [-L / 2, -W / 2, 0]])
    rng = np.random.default_rng(0)
    edges = []
    for c in floor:
        s = np.sign(c[:2])
        pts = np.array([[c[0] + s[0] * t * z, c[1] + s[1] * t * z, z] for z in np.linspace(5, WALL, 25)])
        edges.append((cam.project(pts) + rng.normal(0, 0.4, (25, 2))).tolist())
    q = cam.project(floor)
    kw = dict(floor_length_mm=L, floor_width_mm=W, wall_height_mm=WALL, vertical_edges_px=edges,
              camera_height_mm=H, image_size=(640, 360))
    c0, r0 = calibrate_from_box_marks(q, **kw)
    ca, ra = calibrate_from_box_marks(q, wall_tilt_deg="auto", **kw)
    e0 = np.linalg.norm(np.array(r0.principal_px) - true_p)
    ea = np.linalg.norm(np.array(ra.principal_px) - true_p)
    check(f"walls flaring {TILT:.0f}°, edges marked: fitted {ra.wall_tilt_deg:.1f}°; the point below the camera "
          f"{ea:.1f} px off (taking the walls for vertical: {e0:.1f} px); report: '{ra.text().splitlines()[-1][:60]}…'",
          abs(ra.wall_tilt_deg - TILT) < 1.0 and ea < 2.0 and e0 > 4 * ea and ca.box["tilt_deg"] == round(ra.wall_tilt_deg, 2)
          and r0.wall_tilt_deg is None and "tilt_deg" not in c0.box)
    cg, rg = calibrate_from_box_marks(q, wall_tilt_deg=TILT, **kw)
    with tempfile.TemporaryDirectory() as td:
        f = cg.save(Path(td) / "cam.toml")
        back = Camera.load(f)
    check(f"a given flare kept and saved with the camera ({back.box.get('tilt_deg')}°), the edges fit to it",
          back.box.get("tilt_deg") == TILT and "as given" in rg.tilt_note
          and np.linalg.norm(np.array(rg.principal_px) - true_p) < 2.0)
    x0, x1, y0, y1 = box_xy_at(cg.box, np.array([0.0, 200.0]))
    check(f"the box widens with height: x {cg.box['x'][1]:.0f} → {x1[1]:.0f} mm at 200 mm (tan 12° × 200 = "
          f"{200 * t:.0f})", abs(x1[1] - x1[0] - 200 * t) < 1e-9 and abs(y0[0] - y0[1] - 200 * t) < 1e-9)

    from mufasa.pose3d.mono import _box_hinge
    head = np.array([[cg.box["x"][1] + 30.0, 0.0, 200.0]])          # 30 mm past the floor's wall line, 200 mm up
    pen_v = sum(float(h.sum()) for h in _box_hinge(head, {k: v for k, v in cg.box.items() if k != "tilt_deg"}, -20.0))
    pen_t = sum(float(h.sum()) for h in _box_hinge(head, cg.box, -20.0))
    check(f"a head 200 mm up, 30 mm past the floor's wall line: pushed back {pen_v:.0f} mm with vertical walls, "
          f"{pen_t:.0f} with the flare (the wall is {200 * t:.0f} mm further out there)", pen_v > 5 and pen_t == 0)
    try:
        import torch  # noqa: F401

        from mufasa.pose3d import gpu
        src = Path(gpu.__file__).read_text()
        check("the GPU solver's wall limits flare too", "fl * torch.relu(Xw[:, 2])" in src)
    except ImportError:
        check("PyTorch not installed: the GPU solver not checked", True)

    # the 3-D file and the viewer
    import pandas as pd

    from mufasa.pose3d.io import write_pose3d
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = pd.DataFrame({"frame": np.arange(5), "nose_x": 0.0, "nose_y": 0.0, "nose_z": 40.0})
        write_pose3d(df, td / "v_3d.parquet", {"units": "mm", "box_x": cg.box["x"], "box_y": cg.box["y"],
                                               "box_z": cg.box["z"], "box_tilt_deg": TILT, "edges": []})
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget, load_pose3d
        P = load_pose3d(td / "v_3d.parquet")
        w = Pose3DViewerWidget(None, td / "v_3d.parquet")
        w.resize(500, 400)
        w.show()
        app.processEvents()
        img = w.volume.grab()
        w.cleanup()
        check(f"the 3-D file carries the flare; the viewer reads it ({P.box_tilt_deg}°) and draws the box",
              P.box_tilt_deg == TILT and not img.isNull())
    src = (REPO / "mufasa/pose3d/pipeline.py").read_text()
    check("3-D inference writes the flare into the file", 'meta["box_tilt_deg"]' in src)

    # the command and the form
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.json").write_text(json.dumps({"floor_corners": q.tolist(), "vertical_edges": edges,
                                               "image_size": [640, 360]}))
        from mufasa.tools import pose3d_cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["calibrate-box", str(td / "m.json"), "--length", str(L), "--width", str(W),
                                  "--wall-height", str(WALL), "--camera-height", str(H), "--wall-tilt", "auto",
                                  "--out", str(td / "c.toml")])
        c = Camera.load(td / "c.toml")
        check(f"mufasa-pose3d calibrate-box --wall-tilt auto: exit {rc}, flare {c.box.get('tilt_deg')}° saved",
              rc == 0 and abs(c.box.get("tilt_deg", 0) - TILT) < 1.0 and "walls flare outward" in buf.getvalue())
    form = (REPO / "mufasa/ui_qt/forms/pose3d.py").read_text()
    check("the calibration form: Walls flare outward (degrees, or fitted from the corner edges)",
          'f.addRow("Walls flare outward", trow)' in form and "wall_tilt_deg=tilt" in form)

    # the height-model builder
    import subprocess

    import scipy.io as sio
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names = ['HeadF', 'HeadB', 'HeadL', 'SpineF', 'SpineM', 'SpineL', 'Offset1', 'Offset2', 'HipL', 'HipR',
                 'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 'KneeL', 'ShinL', 'ShinR']
        sizes = {"ratA_day1": 3000, "ratA_day2": 3000, "ratA_day3": 3000, "ratB_day1": 1000}
        for k, (stem, n) in enumerate(sizes.items()):
            r = np.random.default_rng(k)
            base = np.c_[r.normal(0, 50, n), r.normal(0, 50, n), np.abs(r.normal(60, 20, n))]
            mocap = {nm: base + r.normal(0, 15, (n, 3)) for nm in names}
            sub = (td / stem.split("_")[0])
            sub.mkdir(exist_ok=True)
            sio.savemat(sub / f"{stem}.mat", {"mocap": mocap})
        out = td / "prior.npz"
        res = subprocess.run([sys.executable, str(REPO / "scripts" / "build_rat7m_prior.py"), str(td), "--rows", "1000",
                              "--per-rat", "--workers", "2", "--pattern", "rat*_day*.mat", "--subject-regex",
                              r"rat([A-Z])", "--out", str(out)], capture_output=True, text=True)
        bad = res.returncode != 0 and "invalid literal" in res.stderr
        res = subprocess.run([sys.executable, str(REPO / "scripts" / "build_rat7m_prior.py"), str(td), "--rows", "1000",
                              "--per-rat", "--workers", "2", "--pattern", "rat*_day*.mat", "--subject-regex",
                              r"_day(\d)", "--out", str(out)], capture_output=True, text=True)
        z = np.load(out) if out.exists() else None
        counts = dict(zip(*np.unique(z["subject"], return_counts=True), strict=True)) if z is not None else {}
        check(f"build_rat7m_prior.py: other file names (--pattern, --subject-regex: a non-number refused), files in "
              f"subfolders, in parallel; --per-rat: rows per rat (here per day) {counts}",
              res.returncode == 0 and bad and len(counts) == 3 and max(counts.values()) <= 1000 // 3
              and min(counts.values()) == 1000 // 3)

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs/pose3d.md: walls that flare; the builder's options",
          "--wall-tilt" in doc and "--per-rat" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lq_flared_walls: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
