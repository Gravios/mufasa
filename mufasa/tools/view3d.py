"""
``mufasa-view3d`` — 3-D marker viewer with the video below (patch 122ke).

    mufasa-view3d VIDEO [POSE3D]
    mufasa-view3d --pose POSE3D          # no video: the volume only

POSE3D: parquet/CSV with ``<marker>_x/_y/_z`` (and optionally ``frame``,
``<marker>_likelihood`` and ``mufasa_pose3d`` metadata: box, camera, edges).
Without POSE3D, a ``<video stem>*3d*.parquet`` beside the video is used.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mufasa-view3d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="the video (shown below the volume)")
    ap.add_argument("pose3d", nargs="?", help="3-D pose file")
    ap.add_argument("--pose", dest="pose_opt", help="3-D pose file (alternative to POSE3D)")
    return ap


def resolve(args) -> tuple[str | None, str | None, str | None]:
    """(video, pose, error)."""
    from mufasa.ui_qt.pose3d_viewer import find_pose3d
    video, pose = args.video, args.pose3d or args.pose_opt
    if video and video.lower().endswith((".parquet", ".csv")) and not pose:
        video, pose = None, video
    if pose is None and video:
        p = find_pose3d(video)
        pose = str(p) if p else None
    if pose is None:
        return video, None, "no 3-D pose file given or found beside the video"
    return video, pose, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video, pose, err = resolve(args)
    if err:
        print(f"mufasa-view3d: {err}", file=sys.stderr)
        return 2
    from PySide6.QtWidgets import QApplication

    from mufasa.ui_qt.pose3d_viewer import open_pose3d_viewer
    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = open_pose3d_viewer(None, video, pose)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
