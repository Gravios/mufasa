"""Smoke test for patch 122la — 3-D inference says why a video has no camera.

User (screenshot of the 3-D inference section, every video "(no camera)"): "hmm, it seems it
can't find the videos???"

* camera files match in any letter case (``camera_k7.toml`` serves ``K7_cam_…``);
* the form lists the names it looked for, what the calibration folder holds, and what to do
  (Camera calibration with the video's prefix); each unmatched video's tooltip names its files;
* a missing folder is said to be missing; matched videos say nothing;
* *Mark the box…* opens on a frame of the video (``clicked`` passed False as the image:
  "TypeError: 'bool' object is not subscriptable"); no button in the UI hands its ``checked``
  flag to an optional parameter (the skeleton re-import's *Apply* skipped its confirmation).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def main() -> int:
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mufasa.pose3d.camera import camera_for_video, camera_names
    from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
    from mufasa.ui_qt.forms.pose3d import Pose3DInferenceForm

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        root.mkdir()
        ProjectPaths(root).ensure_skeleton()
        write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "c"})
        pose, cal = root / "sources" / "pose", root / "sources" / "calibration"
        pose.mkdir(parents=True, exist_ok=True)
        cal.mkdir(parents=True, exist_ok=True)
        stems = ["K7_cam_20251124_1510_A10-0-051", "K7_cam_20251124_1625_A12-0-052"]
        for s in stems:
            (pose / f"{s}.csv").write_text("head_nose_x,head_nose_y,head_nose_p\n1,2,1\n")
        f = Pose3DInferenceForm(config_path=str(root / PROJECT_CONFIG_FILENAME))

        def state():
            f.refresh()
            items = [f.videos.item(i) for i in range(f.videos.count())]
            return [it.text() for it in items], [it.toolTip() for it in items], f.cam_note.text()
        texts, tips, note = state()
        check(f"no camera files: every video unmatched; the note says what it looked for and what to do "
              f"({note[:90]}…)",
              all("(no camera)" in t for t in texts) and "camera_K7.toml" in note and "no camera files" in note
              and "Camera calibration with the name K7" in note and "camera_K7.toml" in tips[0])
        (cal / "cam_k7_old.toml").write_text("")
        texts, tips, note = state()
        check(f"other camera files: the note lists them ({note[note.find('The folder'):][:60]})",
              "cam_k7_old.toml" in note and "Rename one" in note)
        (cal / "camera_k7.toml").write_text("")
        texts, tips, note = state()
        check(f"camera_k7.toml (any letter case) serves K7_… ({camera_for_video(cal, stems[0]).name}); no note",
              not any("(no camera)" in t for t in texts) and note == ""
              and camera_for_video(cal, stems[0]).name == "camera_k7.toml"
              and camera_names(stems[0])[1] == "camera_K7.toml")
        f.cal_dir.setText(str(root / "nowhere"))
        texts, tips, note = state()
        check(f"a missing folder: said ({note[:70]})", "does not exist" in note)
        check("no folder at all: no camera, no error", camera_for_video(root / "nowhere", stems[0]) is None)

    # Mark the box: the button opens the dialog on the video's background frame
    import cv2
    import numpy as np

    import mufasa.ui_qt.forms.pose3d as FP
    with tempfile.TemporaryDirectory() as td:
        vid = Path(td) / "K7_cam_test.mp4"
        w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for k in range(20):
            w.write(np.full((48, 64, 3), 40 + k, np.uint8))
        w.release()
        seen = []

        class FakeDialog:
            def __init__(self, img, marks, parent):
                seen.append(img)

            def exec(self):
                return 0
        real = FP.BoxMarkDialog
        FP.BoxMarkDialog = FakeDialog
        try:
            cf = FP.CameraCalibrationForm(config_path=None)
            cf.video.addItem(vid.name, str(vid))
            cf.video.setCurrentIndex(cf.video.count() - 1)
            err = ""
            try:
                cf.mark_btn.click()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        finally:
            FP.BoxMarkDialog = real
        check(f"Mark the box: the dialog gets a frame of the video "
              f"({type(seen[0]).__name__ if seen else 'nothing'}{', ' + str(seen[0].shape) if seen and hasattr(seen[0], 'shape') else ''}{err})",
              len(seen) == 1 and isinstance(seen[0], np.ndarray) and seen[0].shape == (48, 64, 3))
    # no clicked/triggered/toggled wired to a slot whose optional first parameter takes the flag
    import ast
    import re
    bad = []
    for fpath in (REPO / "mufasa" / "ui_qt").rglob("*.py"):
        src = fpath.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defs = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.args.args and n.args.args[0].arg == "self":
                a = n.args
                if a.defaults and len(a.args) - len(a.defaults) <= 1:
                    defs.add(n.name)
        for m in re.finditer(r"\.(clicked|triggered|toggled)\.connect\(self\.(\w+)\)", src):
            if m.group(2) in defs:
                bad.append(f"{fpath.name}:{m.group(2)}")
    check(f"no button hands its checked flag to an optional parameter ({bad or 'none'})", not bad)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122la_camera_pairing: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
