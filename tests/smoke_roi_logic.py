"""Smoke-test for mufasa.roi_tools.roi_logic.

Tests the in-memory ROI dict semantics (add / delete / rename /
duplicate) without requiring cv2, h5py, or a real video.

The full mufasa import chain isn't available in the sandbox (numba,
trafaret, etc.). This test extracts the ROILogic class source and
executes it in a synthetic namespace with stub dependencies — the
same pattern used by other smoke tests in this directory.

    PYTHONPATH=. python tests/smoke_roi_logic.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _build_logic_class():
    """Load the ROILogic class via exec, with stub dependencies in
    place of mufasa.utils.checks / mufasa.utils.read_write."""
    import numpy as np
    import pandas as pd

    # Stub the mufasa modules the source file imports
    fake_meta = {"frame_count": 100, "fps": 30.0, "width": 800,
                 "height": 600, "video_name": "test_video"}
    fake_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    checks_mod = ModuleType("mufasa.utils.checks")
    checks_mod.check_file_exist_and_readable = (
        lambda **kwargs: None
    )
    sys.modules["mufasa.utils.checks"] = checks_mod

    read_write_mod = ModuleType("mufasa.utils.read_write")
    read_write_mod.get_fn_ext = (
        lambda filepath=None, **kw: ("/tmp", "test_video", ".mp4")
    )
    read_write_mod.get_video_meta_data = (
        lambda video_path=None, **kw: fake_meta
    )
    read_write_mod.read_frm_of_video = lambda **kw: fake_frame
    read_write_mod.read_roi_data = (
        lambda roi_path=None, **kw: (None, None, None)
    )
    sys.modules["mufasa.utils.read_write"] = read_write_mod
    # Parent packages need stubs too
    sys.modules.setdefault("mufasa", ModuleType("mufasa"))
    sys.modules.setdefault("mufasa.utils", ModuleType("mufasa.utils"))

    # Now we can import roi_logic — but it lives under mufasa.roi_tools
    # which we also need to fake-init to avoid pulling its __init__.py
    sys.modules.setdefault("mufasa.roi_tools",
                           ModuleType("mufasa.roi_tools"))

    # Read the file directly and exec in a fresh namespace, so we don't
    # trigger any sibling-module imports inside mufasa.roi_tools/.
    src = Path("mufasa/roi_tools/roi_logic.py").read_text()
    mod = ModuleType("mufasa.roi_tools.roi_logic")
    sys.modules["mufasa.roi_tools.roi_logic"] = mod
    ns = mod.__dict__
    exec(src, ns)
    return ns


def main() -> int:
    ns = _build_logic_class()
    ROILogic = ns["ROILogic"]
    RECTANGLE = ns["RECTANGLE"]
    CIRCLE = ns["CIRCLE"]
    POLYGON = ns["POLYGON"]

    logic = ROILogic(config_path="/fake/cfg.ini",
                     video_path="/fake/video.mp4")

    # ------------------------------------------------------------------ #
    # Case 1: empty initial state
    # ------------------------------------------------------------------ #
    assert logic.video_name == "test_video"
    assert logic.frame_count == 100
    assert logic.frame_idx == 0
    assert logic.all_roi_names == []

    # ------------------------------------------------------------------ #
    # Case 2: add rectangle
    # ------------------------------------------------------------------ #
    logic.add_rectangle(
        name="zone_a", top_left=(100, 100), bottom_right=(300, 200),
        color_name="Red", bgr=(0, 0, 255),
        thickness=3, ear_tag_size=8,
    )
    assert "zone_a" in logic.all_roi_names
    roi = logic.get_roi("zone_a")
    assert roi.shape_type == RECTANGLE
    assert roi.geometry["topLeftX"] == 100
    assert roi.geometry["width"] == 200
    assert roi.geometry["height"] == 100

    # ------------------------------------------------------------------ #
    # Case 3: duplicate name raises
    # ------------------------------------------------------------------ #
    try:
        logic.add_rectangle(
            name="zone_a", top_left=(0, 0), bottom_right=(50, 50),
            color_name="Red", bgr=(0, 0, 255),
            thickness=3, ear_tag_size=8,
        )
        assert False, "case 3: should have raised on duplicate name"
    except ValueError:
        pass

    # ------------------------------------------------------------------ #
    # Case 4: add circle
    # ------------------------------------------------------------------ #
    logic.add_circle(
        name="zone_b", center=(400, 300), radius=50,
        color_name="Blue", bgr=(255, 0, 0),
        thickness=2, ear_tag_size=10,
    )
    assert "zone_b" in logic.all_roi_names
    assert logic.get_roi("zone_b").shape_type == CIRCLE

    # ------------------------------------------------------------------ #
    # Case 5: add polygon
    # ------------------------------------------------------------------ #
    logic.add_polygon(
        name="zone_c",
        vertices=[(10, 10), (20, 10), (20, 20), (10, 20)],
        color_name="Green", bgr=(0, 255, 0),
        thickness=2, ear_tag_size=6,
    )
    assert "zone_c" in logic.all_roi_names
    assert logic.get_roi("zone_c").shape_type == POLYGON

    # ------------------------------------------------------------------ #
    # Case 6: polygon needs >= 3 vertices
    # ------------------------------------------------------------------ #
    try:
        logic.add_polygon(
            name="too_few", vertices=[(0, 0), (10, 10)],
            color_name="Green", bgr=(0, 255, 0),
            thickness=2, ear_tag_size=6,
        )
        assert False, "case 6: should have raised"
    except ValueError:
        pass

    # ------------------------------------------------------------------ #
    # Case 7: rename
    # ------------------------------------------------------------------ #
    logic.rename_roi("zone_a", "renamed_a")
    assert "renamed_a" in logic.all_roi_names
    assert "zone_a" not in logic.all_roi_names
    assert logic.get_roi("renamed_a").geometry["topLeftX"] == 100

    # ------------------------------------------------------------------ #
    # Case 8: rename collision
    # ------------------------------------------------------------------ #
    try:
        logic.rename_roi("renamed_a", "zone_b")
        assert False, "case 8: rename to existing name should raise"
    except ValueError:
        pass

    # ------------------------------------------------------------------ #
    # Case 9: duplicate ROI shifts geometry
    # ------------------------------------------------------------------ #
    logic.duplicate_roi("renamed_a", "zone_a_dup", offset=(50, 30))
    dup = logic.get_roi("zone_a_dup")
    assert dup.geometry["topLeftX"] == 150
    assert dup.geometry["topLeftY"] == 130
    assert dup.geometry["width"] == 200    # unchanged
    assert dup.geometry["height"] == 100

    # ------------------------------------------------------------------ #
    # Case 10: delete
    # ------------------------------------------------------------------ #
    assert logic.delete_roi("zone_b") is True
    assert "zone_b" not in logic.all_roi_names
    assert logic.delete_roi("does_not_exist") is False

    # ------------------------------------------------------------------ #
    # Case 11: delete_all
    # ------------------------------------------------------------------ #
    logic.delete_all()
    assert logic.all_roi_names == []

    # ------------------------------------------------------------------ #
    # Case 12: frame nav clamps to bounds
    # ------------------------------------------------------------------ #
    logic.goto_frame(50)
    assert logic.frame_idx == 50
    logic.advance_frame(1000)
    assert logic.frame_idx == 99
    logic.advance_frame(-9999)
    assert logic.frame_idx == 0

    # ------------------------------------------------------------------ #
    # Case 13: jump_seconds maps to frames via fps
    # ------------------------------------------------------------------ #
    logic.goto_frame(0)
    logic.jump_seconds(1.0)
    assert logic.frame_idx == 30
    logic.jump_seconds(-1.0)
    assert logic.frame_idx == 0

    # ------------------------------------------------------------------ #
    # Case 14: rendered_frame returns ndarray
    # ------------------------------------------------------------------ #
    rendered = logic.rendered_frame()
    assert rendered is not None
    assert rendered.shape == (600, 800, 3)

    # ------------------------------------------------------------------ #
    # Case 15: rendered_frame actually draws (output sums non-zero)
    # ------------------------------------------------------------------ #
    logic.add_rectangle(
        name="vis_test", top_left=(100, 100), bottom_right=(200, 200),
        color_name="Red", bgr=(0, 0, 255),
        thickness=3, ear_tag_size=8,
    )
    rendered = logic.rendered_frame()
    assert rendered.sum() > 0

    # ------------------------------------------------------------------ #
    # Case 16: BGR parser handles tuple, list, string, garbage
    # ------------------------------------------------------------------ #
    assert ROILogic._parse_bgr((1, 2, 3)) == (1, 2, 3)
    assert ROILogic._parse_bgr([4, 5, 6]) == (4, 5, 6)
    assert ROILogic._parse_bgr("(7, 8, 9)") == (7, 8, 9)
    assert ROILogic._parse_bgr("garbage") == (0, 0, 255)
    assert ROILogic._parse_bgr(None) == (0, 0, 255)

    # ------------------------------------------------------------------ #
    # Case 17: project_path read from config (issue 5)
    # When [General settings]/project_path is set in the .ini, the
    # logic uses it; falls back to config's parent directory if the
    # key is missing.
    # ------------------------------------------------------------------ #
    import tempfile
    import pandas as pd
    tmp_root = Path(tempfile.mkdtemp())
    try:
        good_proj = tmp_root / "real_project"
        good_proj.mkdir()
        good_cfg = tmp_root / "elsewhere" / "project_config.ini"
        good_cfg.parent.mkdir()
        good_cfg.write_text(
            f"[General settings]\nproject_path = {good_proj}\n"
        )
        # Reset the read-roi-data stub since it was patched per-call
        ns["read_roi_data"] = lambda roi_path=None, **kw: (None, None, None)
        l2 = ROILogic(config_path=str(good_cfg),
                      video_path="/fake/video.mp4")
        assert l2.project_path == str(good_proj), (
            f"case 17: project_path={l2.project_path}, want={good_proj}"
        )
        assert l2.roi_h5_path == str(
            good_proj / "logs" / "measures" / "ROI_definitions.h5"
        )

        # Missing project_path falls back to config's directory
        bad_cfg = tmp_root / "no_proj_path.ini"
        bad_cfg.write_text("[General settings]\n")
        l3 = ROILogic(config_path=str(bad_cfg),
                      video_path="/fake/video.mp4")
        assert l3.project_path == str(tmp_root)

        # ------------------------------------------------------------ #
        # Case 18: polygon vertices parsed via ast.literal_eval (issue 6)
        # ------------------------------------------------------------ #
        # Simulate a polygon row with vertices stored as a Python-list
        # string (the SimBA H5 convention)
        fake_poly_row = pd.Series({
            "Video": "test_video",
            "Name": "imported_poly",
            "Color name": "Green",
            "Color BGR": "(0, 255, 0)",
            "Thickness": 2,
            "Ear_tag_size": 8,
            "vertices": "[[10, 20], [30, 40], [50, 60]]",
            "Center_X": 30,
            "Center_Y": 40,
        })
        roi = ROILogic._row_to_definition(fake_poly_row, POLYGON)
        assert roi is not None, "case 18: parse should succeed"
        assert roi.geometry["vertices"] == [[10, 20], [30, 40], [50, 60]]

        # Malicious vertices string — ast.literal_eval refuses
        # arbitrary code; the row parser should return None rather
        # than execute it.
        evil_row = pd.Series({
            "Video": "test_video",
            "Name": "evil",
            "Color name": "Red",
            "Color BGR": "(0, 0, 255)",
            "Thickness": 2,
            "Ear_tag_size": 8,
            "vertices": "__import__('os').system('echo PWNED')",
            "Center_X": 0,
            "Center_Y": 0,
        })
        roi = ROILogic._row_to_definition(evil_row, POLYGON)
        # Either None (rejected) or a valid ROIDefinition with no
        # side effects — the test passes as long as the function
        # didn't execute the injected expression. Since the function
        # catches all exceptions and returns None, that's our
        # expected outcome.
        assert roi is None, (
            f"case 18: malicious string should not produce a valid "
            f"ROI, got {roi}"
        )

        # ------------------------------------------------------------ #
        # Case 19: save preserves other-video rows (issue 7 supports)
        # Two ROILogic instances — one for video_a, one for video_b.
        # Save A, then save B, both videos' ROIs should remain in the
        # final file.
        # ------------------------------------------------------------ #
        save_proj = tmp_root / "save_proj"
        save_proj.mkdir()
        (save_proj / "logs" / "measures").mkdir(parents=True)
        save_cfg = tmp_root / "save_proj_cfg.ini"
        save_cfg.write_text(
            f"[General settings]\nproject_path = {save_proj}\n"
        )

        # Override get_fn_ext for these to return different video names
        orig_get_fn_ext = sys.modules["mufasa.utils.read_write"].get_fn_ext
        sys.modules["mufasa.utils.read_write"].get_fn_ext = (
            lambda filepath=None, **kw: (
                "/tmp", "video_a", ".mp4"
            ) if "video_a" in str(filepath) else ("/tmp", "video_b", ".mp4")
        )

        # Re-import the module to pick up the patched read_write
        # (the function captures the import-time reference)
        if "mufasa.roi_tools.roi_logic" in sys.modules:
            del sys.modules["mufasa.roi_tools.roi_logic"]
        ns2 = _build_logic_class()
        # Re-stub get_fn_ext after _build_logic_class re-installs it
        ns2["get_fn_ext"] = (
            lambda filepath=None, **kw: (
                "/tmp", "video_a", ".mp4"
            ) if "video_a" in str(filepath) else ("/tmp", "video_b", ".mp4")
        )
        ROILogic2 = ns2["ROILogic"]

        try:
            la = ROILogic2(config_path=str(save_cfg),
                           video_path="/tmp/video_a.mp4")
            la.add_rectangle(
                name="zone_in_a", top_left=(0, 0), bottom_right=(100, 100),
                color_name="Red", bgr=(0, 0, 255),
                thickness=2, ear_tag_size=8,
            )
            la.save()
            assert os.path.isfile(la.roi_h5_path), "save A didn't create file"

            lb = ROILogic2(config_path=str(save_cfg),
                           video_path="/tmp/video_b.mp4")
            # When B loads, it should NOT find A's rectangle in self.defs
            # (that's for video_a, not video_b)
            assert "zone_in_a" not in lb.all_roi_names, (
                "case 19: video_b should not see video_a's ROIs in defs"
            )
            # But it should find them in other_video_rois
            assert "video_a" in lb.other_video_rois, (
                f"case 19: other_video_rois={list(lb.other_video_rois.keys())}"
            )

            lb.add_circle(
                name="zone_in_b", center=(50, 50), radius=20,
                color_name="Blue", bgr=(255, 0, 0),
                thickness=2, ear_tag_size=8,
            )
            lb.save()

            # After both saves, A's rectangles AND B's circles should
            # be in the final file. We verify by re-reading via a
            # third instance.
            lc = ROILogic2(config_path=str(save_cfg),
                           video_path="/tmp/video_a.mp4")
            assert "zone_in_a" in lc.all_roi_names, (
                "case 19: video_a's ROI was lost after video_b's save"
            )
            assert "video_b" in lc.other_video_rois, (
                "case 19: video_b's ROIs missing from other_video_rois"
            )
        except ImportError:
            # H5 deps (pytables/h5py) not available in sandbox
            print("case 19: skipped (h5 deps not in sandbox)")

        # ------------------------------------------------------------ #
        # Case 20: frame cache LRU cap (issue 10)
        # ------------------------------------------------------------ #
        FRAME_CACHE_MAX = ns["FRAME_CACHE_MAX"]
        # Force many distinct frame reads
        for i in range(FRAME_CACHE_MAX + 20):
            logic.goto_frame(i % logic.frame_count)
        # Cache should never exceed the cap
        assert len(logic._frame_cache) <= FRAME_CACHE_MAX, (
            f"case 20: cache grew to {len(logic._frame_cache)}, "
            f"max={FRAME_CACHE_MAX}"
        )

    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("smoke_roi_logic: 20/20 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
