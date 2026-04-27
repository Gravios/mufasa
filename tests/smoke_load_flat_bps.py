"""Smoke-test for _load_flat_bps in mufasa.ui_qt.forms.pose_cleanup.

Validates the helper used by EgocentricAlignmentForm to populate its
body-part dropdowns. Headless — no PySide6 import needed.

    PYTHONPATH=. python tests/smoke_load_flat_bps.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    # Import without triggering the rest of the form module's QtWidgets
    # imports — _load_flat_bps is pure Python, so we import the file
    # directly via importlib.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_pose_cleanup",
        Path("mufasa/ui_qt/forms/pose_cleanup.py").resolve(),
    )
    # We can't actually execute the module without PySide6.
    # Instead: read the function source and exec it in a minimal env.
    src = Path("mufasa/ui_qt/forms/pose_cleanup.py").read_text()
    # Locate the function definition
    needle = "def _load_flat_bps("
    start = src.find(needle)
    assert start >= 0, "_load_flat_bps not found"
    # Walk forward line by line; the function ends when we find an
    # unindented line that isn't blank and isn't a continuation.
    lines = src[start:].splitlines(keepends=True)
    func_lines = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln[0] in " \t":
            func_lines.append(ln)
        else:
            break
    func_src = "".join(func_lines)

    # Provide the imports the helper needs
    ns = {}
    import configparser
    from pathlib import Path as P
    ns["configparser"] = configparser
    ns["Path"] = P
    exec(func_src, ns)
    _load_flat_bps = ns["_load_flat_bps"]

    tmp = Path(tempfile.mkdtemp())
    try:
        # ------------------------------------------------------------ #
        # Case 1: project with one-per-line bp_names returns the list
        # in order
        # ------------------------------------------------------------ #
        proj = tmp / "p1" / "project_folder"
        bp_dir = proj / "logs" / "measures" / "pose_configs" / "bp_names"
        bp_dir.mkdir(parents=True)
        (bp_dir / "project_bp_names.csv").write_text(
            "nose\nheadmid\near_left\nback1\ntailbase\n"
        )
        cfg = proj / "project_config.ini"
        cfg.write_text(
            f"[General settings]\nproject_path = {proj}\n"
        )
        out = _load_flat_bps(str(cfg))
        assert out == ["nose", "headmid", "ear_left", "back1", "tailbase"], (
            f"case 1: {out}"
        )

        # ------------------------------------------------------------ #
        # Case 2: comma-separated single row (legacy format) parses
        # correctly with trailing empties dropped
        # ------------------------------------------------------------ #
        proj2 = tmp / "p2" / "project_folder"
        bp_dir2 = proj2 / "logs" / "measures" / "pose_configs" / "bp_names"
        bp_dir2.mkdir(parents=True)
        (bp_dir2 / "project_bp_names.csv").write_text(
            "Mouse1_left_ear,Mouse1_right_ear,Mouse1_nose,,,,\n"
        )
        cfg2 = proj2 / "project_config.ini"
        cfg2.write_text(f"[General settings]\nproject_path = {proj2}\n")
        out2 = _load_flat_bps(str(cfg2))
        assert out2 == ["Mouse1_left_ear", "Mouse1_right_ear",
                        "Mouse1_nose"], f"case 2: {out2}"

        # ------------------------------------------------------------ #
        # Case 3: missing project_path returns []
        # ------------------------------------------------------------ #
        cfg3 = tmp / "no_project_path.ini"
        cfg3.write_text("[General settings]\n")
        out3 = _load_flat_bps(str(cfg3))
        assert out3 == [], f"case 3: {out3}"

        # ------------------------------------------------------------ #
        # Case 4: missing bp_names file returns []
        # ------------------------------------------------------------ #
        proj4 = tmp / "p4" / "project_folder"
        proj4.mkdir(parents=True)
        cfg4 = proj4 / "project_config.ini"
        cfg4.write_text(f"[General settings]\nproject_path = {proj4}\n")
        out4 = _load_flat_bps(str(cfg4))
        assert out4 == [], f"case 4: {out4}"

        # ------------------------------------------------------------ #
        # Case 5: empty bp_names file returns []
        # ------------------------------------------------------------ #
        proj5 = tmp / "p5" / "project_folder"
        bp_dir5 = proj5 / "logs" / "measures" / "pose_configs" / "bp_names"
        bp_dir5.mkdir(parents=True)
        (bp_dir5 / "project_bp_names.csv").write_text("\n\n\n")
        cfg5 = proj5 / "project_config.ini"
        cfg5.write_text(f"[General settings]\nproject_path = {proj5}\n")
        out5 = _load_flat_bps(str(cfg5))
        assert out5 == [], f"case 5: {out5}"

        print("smoke_load_flat_bps: 5/5 cases passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
