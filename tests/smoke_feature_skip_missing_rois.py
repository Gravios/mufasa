"""Smoke-test for ROI-missing-videos skip behavior in feature_subsets.

Replicates the filtering logic that decides which videos to skip vs.
run when some videos have incomplete ROI coverage. Doesn't import
the full feature_subsets module (pulls heavy deps); runs the same
logic on a synthetic scenario.

    PYTHONPATH=. python tests/smoke_feature_skip_missing_rois.py
"""
from __future__ import annotations

import os
import sys
import tempfile


def _filter_videos(video_names, roi_dict_keys, check_result):
    """Mirror the logic from feature_subsets.py.

    video_names: list of all video names being processed
    roi_dict_keys: set of video names that appear in the per-video
                   ROI dict (i.e. have at least one ROI defined)
    check_result: either True (no missing ROIs) or
                  (False, {video: [missing_roi_names...]})
    Returns: (target_video_names, videos_skipped_dict)
    """
    videos_with_complete_rois = set()
    videos_skipped = {}
    if check_result is True:
        videos_with_complete_rois = set(video_names)
    else:
        _, missing_rois = check_result
        for vname, missing_list in missing_rois.items():
            if not missing_list:
                videos_with_complete_rois.add(vname)
            else:
                videos_skipped[vname] = list(missing_list)

    videos_with_no_rois_at_all = [
        v for v in video_names if v not in roi_dict_keys
    ]
    for v in videos_with_no_rois_at_all:
        videos_skipped[v] = ["(no ROIs defined for this video)"]

    target_video_names = [
        v for v in video_names
        if v in roi_dict_keys and v in videos_with_complete_rois
    ]
    return target_video_names, videos_skipped


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: every video has every ROI — keep all
    # ------------------------------------------------------------------ #
    videos = ["v1", "v2", "v3"]
    roi_keys = {"v1", "v2", "v3"}
    target, skipped = _filter_videos(videos, roi_keys, True)
    assert set(target) == {"v1", "v2", "v3"}, f"case 1 target: {target}"
    assert skipped == {}, f"case 1 skipped: {skipped}"

    # ------------------------------------------------------------------ #
    # Case 2: one video missing one ROI — skip it, keep the rest
    # ------------------------------------------------------------------ #
    videos = ["v1", "v2", "v3"]
    roi_keys = {"v1", "v2", "v3"}
    check = (False, {"v1": [], "v2": [], "v3": ["zone_a"]})
    target, skipped = _filter_videos(videos, roi_keys, check)
    assert target == ["v1", "v2"], f"case 2 target: {target}"
    assert skipped == {"v3": ["zone_a"]}, f"case 2 skipped: {skipped}"

    # ------------------------------------------------------------------ #
    # Case 3: matches the user's actual error — most videos missing
    # the same two ROIs, one missing a different one
    # ------------------------------------------------------------------ #
    videos = ["v1", "v2", "v3", "v4", "v5"]
    roi_keys = {"v1", "v2", "v3", "v4", "v5"}
    check = (False, {
        "v1": ["platform", "nose-poke"],
        "v2": ["platform", "nose-poke"],
        "v3": [],   # this one has all required ROIs
        "v4": ["window"],
        "v5": ["platform", "nose-poke"],
    })
    target, skipped = _filter_videos(videos, roi_keys, check)
    assert target == ["v3"], f"case 3 target: {target}"
    assert set(skipped.keys()) == {"v1", "v2", "v4", "v5"}
    assert skipped["v1"] == ["platform", "nose-poke"]
    assert skipped["v4"] == ["window"]

    # ------------------------------------------------------------------ #
    # Case 4: video missing from ROI dict entirely (zero ROIs defined)
    # ------------------------------------------------------------------ #
    videos = ["v1", "v2", "v3"]
    roi_keys = {"v1", "v2"}   # v3 has no ROIs at all
    check = (False, {"v1": [], "v2": []})
    target, skipped = _filter_videos(videos, roi_keys, check)
    assert target == ["v1", "v2"]
    assert "v3" in skipped
    assert skipped["v3"] == ["(no ROIs defined for this video)"]

    # ------------------------------------------------------------------ #
    # Case 5: NO videos have complete ROI coverage — empty target
    # (caller should raise on this; we just verify the filter)
    # ------------------------------------------------------------------ #
    videos = ["v1", "v2"]
    roi_keys = {"v1", "v2"}
    check = (False, {"v1": ["a"], "v2": ["b"]})
    target, skipped = _filter_videos(videos, roi_keys, check)
    assert target == [], f"case 5 target: {target}"
    assert len(skipped) == 2

    # ------------------------------------------------------------------ #
    # Case 6: ordering preserved — target_video_names follows the
    # original video_names order, not roi_keys iteration order
    # ------------------------------------------------------------------ #
    videos = ["beta", "alpha", "gamma"]
    roi_keys = {"alpha", "beta", "gamma"}
    target, _ = _filter_videos(videos, roi_keys, True)
    assert target == ["beta", "alpha", "gamma"], f"case 6: {target}"

    # ------------------------------------------------------------------ #
    # Case 7: video has empty missing list AND is in roi_keys — keep
    # ------------------------------------------------------------------ #
    videos = ["v1"]
    roi_keys = {"v1"}
    check = (False, {"v1": []})
    target, skipped = _filter_videos(videos, roi_keys, check)
    assert target == ["v1"]
    assert skipped == {}

    # ------------------------------------------------------------------ #
    # Case 8: CSV log writes correctly (smoke test the file-writing
    # part by reproducing the same csv writer pattern)
    # ------------------------------------------------------------------ #
    import csv
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_log_path = os.path.join(tmpdir, "skipped_videos_missing_rois.csv")
        skipped = {
            "v1": ["platform", "nose-poke"],
            "v2": ["window"],
        }
        with open(skip_log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video", "missing_rois"])
            for v, missing in sorted(skipped.items()):
                writer.writerow([v, ";".join(missing)])
        # Read it back and verify
        with open(skip_log_path) as fh:
            content = fh.read()
        assert "video,missing_rois" in content
        assert "v1,platform;nose-poke" in content
        assert "v2,window" in content

    # ------------------------------------------------------------------ #
    # Case 9: skipped CSV stays well-formed when ROI names contain
    # characters that could confuse a naive joiner (commas, quotes)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_log_path = os.path.join(tmpdir, "skip.csv")
        skipped = {"v1": ["zone, with comma", 'quoted "name"']}
        with open(skip_log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video", "missing_rois"])
            for v, missing in sorted(skipped.items()):
                writer.writerow([v, ";".join(missing)])
        # csv module handles quoting; reading back via csv should
        # round-trip
        with open(skip_log_path) as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["video", "missing_rois"]
        assert rows[1][0] == "v1"
        # The ; separator is preserved within the (csv-quoted) cell
        assert "zone, with comma" in rows[1][1]
        assert 'quoted "name"' in rows[1][1]

    print("smoke_feature_skip_missing_rois: 9/9 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
