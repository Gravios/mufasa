"""
smoke_pose_video_overlay
=========================

Headless smoke test for the pose-on-video overlay viewer.

Generates a synthetic 2-second video and matching synthetic
smoothed + raw pose files, opens the viewer in offscreen Qt
mode, advances through several frames, and verifies:

  Case 1: VideoSource opens the synthetic video and reports
          correct fps / frame count / dimensions.
  Case 2: Frame N can be read independently after seeking.
  Case 3: OverlayViewer constructs without error from typical
          inputs (video + smoothed only, video + smoothed +
          raw).
  Case 4: Advancing the scrubber updates the displayed frame
          index and pose row.
  Case 5: Layer toggles flip the visibility of the
          corresponding scene items.
  Case 6: Pose offset works — pose frame 0 corresponds to
          video frame `pose_offset`.
  Case 7: Past-end-of-pose: when video keeps going past the
          last pose frame, marker overlays hide gracefully.

Skips with an informative message if PySide6 or OpenCV are
not importable.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Force offscreen Qt before any Qt imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Repo on PYTHONPATH so subprocess and direct imports work
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

try:
    from PySide6.QtWidgets import QApplication  # type: ignore
    _has_qt = True
except ImportError:
    _has_qt = False


def _make_synthetic_video(path, fps=30, n_frames=60, w=320, h=240):
    """Write a tiny color-cycling synthetic video for the test.
    Markers in the pose data don't have to land on anything in
    particular — we're testing overlay mechanics, not visual
    accuracy."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        # Some opencv builds don't have mp4v; fall back to MJPG/.avi
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
        assert writer.isOpened(), (
            "No usable video codec; can't run smoke test"
        )
    for i in range(n_frames):
        frame = np.full((h, w, 3), fill_value=20, dtype=np.uint8)
        # Bright moving rectangle so a human eyeballing the test
        # output can confirm playback works
        x0 = (i * 4) % (w - 30)
        frame[h // 2 - 15:h // 2 + 15, x0:x0 + 30] = (50, 200, 100)
        writer.write(frame)
    writer.release()
    return path


def _make_synthetic_pose(path, n_frames=60, n_markers=15, with_var=True):
    """Flat-column parquet matching v2 smoother output schema."""
    rng = np.random.default_rng(0)
    markers = [
        "back2", "back1", "back3", "lateral_left", "lateral_right",
        "center", "back4", "neck", "headmid",
        "nose", "ear_left", "ear_right",
        "tailbase", "tailmid", "tailend",
    ][:n_markers]
    cols = {}
    for k, m in enumerate(markers):
        cols[f"{m}_x"] = (
            100.0 + 20.0 * np.sin(np.arange(n_frames) * 0.1 + k)
            + rng.normal(0, 0.5, n_frames)
        )
        cols[f"{m}_y"] = (
            120.0 + 15.0 * np.cos(np.arange(n_frames) * 0.1 + k)
            + rng.normal(0, 0.5, n_frames)
        )
        cols[f"{m}_p"] = np.full(n_frames, 0.95)
        if with_var:
            cols[f"{m}_var_x"] = np.full(n_frames, 1.0)
            cols[f"{m}_var_y"] = np.full(n_frames, 1.0)
    pd.DataFrame(cols).to_parquet(path, index=False)


def _ensure_app():
    """Get-or-create singleton QApplication for the test."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def case1_video_source(td: Path):
    from mufasa.tools.pose_video_overlay import VideoSource
    vid = td / "v.mp4"
    actual_path = _make_synthetic_video(vid, fps=30, n_frames=60)
    src = VideoSource(actual_path)
    assert src.n_frames == 60, f"frames: {src.n_frames}"
    assert abs(src.fps - 30.0) < 0.5
    assert src.width == 320 and src.height == 240
    src.close()
    return actual_path


def case2_random_seek(actual_video_path):
    from mufasa.tools.pose_video_overlay import VideoSource
    src = VideoSource(actual_video_path)
    f0 = src.read(0)
    f30 = src.read(30)
    f59 = src.read(59)
    f_oob = src.read(100)
    assert f0 is not None and f0.shape == (240, 320, 3)
    assert f30 is not None and not np.array_equal(f0, f30), (
        "seek to frame 30 returned same content as frame 0"
    )
    assert f59 is not None
    assert f_oob is None, "out-of-bounds should return None"
    src.close()


def case3_construct_viewer(actual_video_path, td: Path):
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path, n_frames=60, with_var=True)
    raw_path = td / "raw.parquet"
    _make_synthetic_pose(raw_path, n_frames=60, with_var=False)

    smoothed = _load_pose_file(pose_path)
    raw = _load_pose_file(raw_path)
    assert smoothed.variances is not None
    assert raw.variances is None

    # Smoothed-only
    src = VideoSource(actual_video_path)
    win_s = OverlayViewer(
        video=src, smoothed=smoothed, raw=None,
        start_frame=0,
    )
    assert win_s.scene_obj.show_smoothed is True
    assert win_s.scene_obj.show_raw is False
    src.close()

    # Smoothed + raw
    src = VideoSource(actual_video_path)
    win_sr = OverlayViewer(
        video=src, smoothed=smoothed, raw=raw,
        start_frame=10,
    )
    assert win_sr.scrubber.value() == 10
    assert win_sr.scene_obj.show_smoothed is True
    assert win_sr.scene_obj.show_raw is True
    src.close()


def case4_advance_frames(actual_video_path, td: Path):
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)

    # Position dot for back1 should change between frames
    dot_b1 = win.scene_obj._smoothed_dots["back1"]
    win._set_frame(0)
    p0 = (dot_b1.pos().x(), dot_b1.pos().y())
    win._set_frame(20)
    p20 = (dot_b1.pos().x(), dot_b1.pos().y())
    assert p0 != p20, (
        "Marker did not move between frame 0 and 20: "
        f"{p0} vs {p20}"
    )
    # Step button-equivalent
    win.step(+1)
    assert win.scrubber.value() == 21
    win.step(-5)
    assert win.scrubber.value() == 16
    src.close()


def case5_layer_toggles(actual_video_path, td: Path):
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    raw_path = td / "raw.parquet"
    _make_synthetic_pose(pose_path)
    _make_synthetic_pose(raw_path, with_var=False)
    smoothed = _load_pose_file(pose_path)
    raw = _load_pose_file(raw_path)

    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=raw)

    # Toggle smoothed off
    win.cb_smoothed.setChecked(False)
    assert win.scene_obj.show_smoothed is False
    s_dot = next(iter(win.scene_obj._smoothed_dots.values()))
    win._set_frame(5)  # re-evaluate visibility
    assert s_dot.isVisible() is False

    # Toggle skeleton off
    win.cb_skeleton.setChecked(False)
    assert win.scene_obj.show_skeleton is False
    win._set_frame(6)
    line = next(iter(win.scene_obj._skeleton_lines_raw.values()))
    assert line.isVisible() is False

    src.close()


def case6_pose_offset(actual_video_path, td: Path):
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    # Pose file shorter than video, with offset 5: pose row 0
    # corresponds to video frame 5.
    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path, n_frames=20)
    smoothed = _load_pose_file(pose_path)

    src = VideoSource(actual_video_path)  # 60 frames
    win = OverlayViewer(
        video=src, smoothed=smoothed, raw=None,
        pose_offset=5, start_frame=5,
    )
    dot_b1 = win.scene_obj._smoothed_dots["back1"]
    # Video frame 5 → pose row 0
    win._set_frame(5)
    pos_at_video5 = (dot_b1.pos().x(), dot_b1.pos().y())
    # Now compare against what pose row 0 actually is in the
    # data — should match the dot position.
    pose_row_0_xy = (
        float(smoothed.positions[0, smoothed.markers.index("back1"), 0]),
        float(smoothed.positions[0, smoothed.markers.index("back1"), 1]),
    )
    assert abs(pos_at_video5[0] - pose_row_0_xy[0]) < 1e-3
    assert abs(pos_at_video5[1] - pose_row_0_xy[1]) < 1e-3

    # Frame 4 (before pose starts) — markers should be hidden
    win._set_frame(4)
    assert dot_b1.isVisible() is False, (
        "Marker should hide for video frames before pose data starts"
    )

    src.close()


def case7_past_end_of_pose(actual_video_path, td: Path):
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    # Pose covers only the first 30 frames; video has 60.
    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path, n_frames=30)
    smoothed = _load_pose_file(pose_path)

    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)
    dot = win.scene_obj._smoothed_dots["back1"]
    win._set_frame(50)
    assert dot.isVisible() is False, (
        "Marker should hide once we go past pose end"
    )
    # Earlier frame should still show
    win._set_frame(10)
    assert dot.isVisible() is True
    src.close()


def case8_zoom_in_out(actual_video_path, td: Path):
    """Keyboard zoom in then out leaves transform near identity
    (modulo float roundoff). Verifies _keyboard_zoom path."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)
    win.show()  # required for fitInView to size correctly
    win.resize(800, 600)

    initial_scale = win.view.transform().m11()
    # Zoom in 3 times
    for _ in range(3):
        win._keyboard_zoom(1.25)
    after_zoom_in = win.view.transform().m11()
    assert after_zoom_in > initial_scale * 1.9, (
        f"3 zoom-ins should ~1.95× the scale; "
        f"got {after_zoom_in / initial_scale:.3f}×"
    )
    # Zoom back out 3 times
    for _ in range(3):
        win._keyboard_zoom(1 / 1.25)
    after_round_trip = win.view.transform().m11()
    assert abs(after_round_trip - initial_scale) < 1e-6 * initial_scale, (
        f"zoom in×3 then out×3 should return to start; "
        f"got {after_round_trip} vs {initial_scale}"
    )
    src.close()


def case9_zoom_clamped_at_limits(actual_video_path, td: Path):
    """Zoom requests beyond MIN/MAX_SCALE limits are no-ops.
    Prevents runaway zoom when the user hammers Ctrl++."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)
    win.show()
    win.resize(800, 600)

    # Hammer zoom-in until clamped
    for _ in range(100):
        win._keyboard_zoom(1.25)
    cur = win.view.transform().m11()
    assert cur <= win.view.MAX_SCALE * 1.001, (
        f"zoom-in should clamp at MAX_SCALE={win.view.MAX_SCALE}; "
        f"got {cur}"
    )
    # And zoom-out until clamped
    for _ in range(200):
        win._keyboard_zoom(1 / 1.25)
    cur = win.view.transform().m11()
    assert cur >= win.view.MIN_SCALE * 0.999, (
        f"zoom-out should clamp at MIN_SCALE={win.view.MIN_SCALE}; "
        f"got {cur}"
    )
    src.close()


def case10_reset_view(actual_video_path, td: Path):
    """Reset (Ctrl+0) restores fit-to-scene from any zoom state."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)
    win.show()
    win.resize(800, 600)

    # Apply several zooms
    for _ in range(5):
        win._keyboard_zoom(1.5)
    zoomed_scale = win.view.transform().m11()
    # Reset
    win._reset_view()
    reset_scale = win.view.transform().m11()
    assert reset_scale != zoomed_scale, (
        "reset should change the transform"
    )
    # Reset should fit the scene rect; the resulting scale
    # depends on viewport size, but it must be a reasonable
    # finite positive number, not the post-zoom scale.
    assert reset_scale > 0
    assert win.view.user_zoomed is False, (
        "reset_view should clear user_zoomed flag"
    )
    src.close()


def case11_speed_combo_initial_state(actual_video_path, td: Path):
    """Speed combo is created with the expected presets and
    starts at 1.0×."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer, PLAYBACK_SPEED_PRESETS,
        DEFAULT_PLAYBACK_SPEED,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)

    assert win.speed_combo.count() == len(PLAYBACK_SPEED_PRESETS)
    # Verify each preset's data
    for i, expected in enumerate(PLAYBACK_SPEED_PRESETS):
        got = win.speed_combo.itemData(i)
        assert float(got) == expected, (
            f"preset[{i}]: expected {expected}, got {got}"
        )
    # Default selection at 1.0×
    cur_data = win.speed_combo.itemData(win.speed_combo.currentIndex())
    assert float(cur_data) == DEFAULT_PLAYBACK_SPEED
    assert win.play_speed == DEFAULT_PLAYBACK_SPEED
    src.close()


def case12_combo_pick_changes_speed(actual_video_path, td: Path):
    """Picking a preset from the combo updates play_speed and
    the timer interval."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)

    interval_at_1x = win.timer.interval()
    # Find the index for 4.0×
    idx_4x = win.speed_combo.findData(4.0)
    assert idx_4x >= 0
    # Simulate user picking 4× (activated signal — programmatic)
    win._on_speed_combo_picked(idx_4x)
    assert win.play_speed == 4.0
    interval_at_4x = win.timer.interval()
    # 4× speed → 4× faster ticks → ~1/4 the interval
    assert interval_at_4x < interval_at_1x / 3, (
        f"4× should shorten interval; was {interval_at_1x}, "
        f"now {interval_at_4x}"
    )
    src.close()


def case13_keyboard_speed_syncs_combo(actual_video_path, td: Path):
    """Keyboard +/- updates play_speed AND the combo display
    stays in sync."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)

    # Faster (×2): 1.0 → 2.0 — this is a preset, combo should
    # land on the 2.0 row by exact match.
    win.set_speed(win.play_speed * 2)
    assert win.play_speed == 2.0
    cur_data = win.speed_combo.itemData(win.speed_combo.currentIndex())
    assert float(cur_data) == 2.0, (
        f"combo not synced: {cur_data} vs play_speed={win.play_speed}"
    )

    # Reset (0): 2.0 → 1.0
    win.set_speed(1.0)
    assert win.play_speed == 1.0
    cur_data = win.speed_combo.itemData(win.speed_combo.currentIndex())
    assert float(cur_data) == 1.0
    src.close()


def case14_custom_speed_typed(actual_video_path, td: Path):
    """Typing a custom (non-preset) speed and confirming updates
    play_speed; bad input is rejected and combo reverts."""
    _ensure_app()
    from mufasa.tools.pose_video_overlay import (
        VideoSource, OverlayViewer,
    )
    from mufasa.tools.pose_viewer import _load_pose_file

    pose_path = td / "smoothed.parquet"
    _make_synthetic_pose(pose_path)
    smoothed = _load_pose_file(pose_path)
    src = VideoSource(actual_video_path)
    win = OverlayViewer(video=src, smoothed=smoothed, raw=None)

    # Custom value 1.5
    win.speed_combo.setEditText("1.5")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 1.5) < 1e-9
    # Combo display should now read "1.5×"
    assert win.speed_combo.currentText() == "1.5×"

    # Tolerated suffixes: ×, x, X
    win.speed_combo.setEditText("3×")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 3.0) < 1e-9
    win.speed_combo.setEditText("0.75x")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 0.75) < 1e-9

    # Bad input: speed unchanged, combo reverts to "0.75×"
    win.speed_combo.setEditText("garbage")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 0.75) < 1e-9
    assert win.speed_combo.currentText() == "0.75×"

    # Negative or zero: unchanged
    win.speed_combo.setEditText("-2")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 0.75) < 1e-9
    win.speed_combo.setEditText("0")
    win._on_speed_combo_edited()
    assert abs(win.play_speed - 0.75) < 1e-9

    # Out-of-range clamped (high → MAX, low → MIN)
    from mufasa.tools.pose_video_overlay import (
        MAX_PLAYBACK_SPEED, MIN_PLAYBACK_SPEED,
    )
    win.speed_combo.setEditText("100")
    win._on_speed_combo_edited()
    assert win.play_speed == MAX_PLAYBACK_SPEED
    win.speed_combo.setEditText("0.001")
    win._on_speed_combo_edited()
    assert win.play_speed == MIN_PLAYBACK_SPEED
    src.close()


def main() -> int:
    if not _has_qt:
        print("PySide6 not installed; skipping smoke test")
        return 0
    if not _has_cv2:
        print("OpenCV (cv2) not installed; skipping smoke test")
        return 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        actual_video_path = case1_video_source(td)
        case2_random_seek(actual_video_path)
        case3_construct_viewer(actual_video_path, td)
        case4_advance_frames(actual_video_path, td)
        case5_layer_toggles(actual_video_path, td)
        case6_pose_offset(actual_video_path, td)
        case7_past_end_of_pose(actual_video_path, td)
        case8_zoom_in_out(actual_video_path, td)
        case9_zoom_clamped_at_limits(actual_video_path, td)
        case10_reset_view(actual_video_path, td)
        case11_speed_combo_initial_state(actual_video_path, td)
        case12_combo_pick_changes_speed(actual_video_path, td)
        case13_keyboard_speed_syncs_combo(actual_video_path, td)
        case14_custom_speed_typed(actual_video_path, td)
    print("smoke_pose_video_overlay: 14/14 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
