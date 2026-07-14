"""
tests/smoke_122gt_marker_rename_pathfix.py
==========================================

Patch 122gt — fix "Save model" crash: 'str' object has no attribute 'parent'.

rename_markers set ``cp = os.fspath(config_path)`` (a str) and passed it to
write_project_toml(path: Path, ...) / project_paths_from_config, which do
``path.parent`` — so Save model raised ``'str' object has no attribute
'parent'`` before writing anything. Fixed by using ``cp = Path(config_path)``.
The read_df/write_df import (which needs h5py) is now lazy — only when there
are pose files to rewrite.

Regression test: run the full rename_markers orchestrator against a real
temp v1 project.toml (+ [skeleton]); it must not crash and must rename
body_parts and skeleton edges.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    from mufasa.model.marker_rename import rename_markers
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        read_project_toml,
        read_skeleton,
        write_project_toml,
        write_skeleton,
    )

    root = Path(tempfile.mkdtemp())
    cp = root / "project.toml"
    write_project_toml(cp, {
        "project_layout_version": PROJECT_LAYOUT_VERSION,
        "pose": {"body_parts": ["nose", "ear_left", "ear_right", "tailbase"],
                 "file_type": "parquet"},
    })
    write_skeleton(cp, nodes=["nose", "ear_left", "ear_right", "tailbase"],
                   edges=[("nose", "ear_left"), ("ear_left", "ear_right"),
                          ("ear_right", "tailbase")])

    rmap = {"ear_left": "left_ear", "ear_right": "right_ear"}

    # dry run + real run must not raise (regression for the .parent crash)
    crashed = False
    try:
        rename_markers(cp, rmap, dry_run=True)
        rename_markers(cp, rmap)
    except Exception as e:  # noqa: BLE001
        crashed = True
        print(f"  (raised: {type(e).__name__}: {e})")
    check("rename_markers runs without the 'str'/.parent crash", not crashed)

    d = read_project_toml(cp)
    check("project.toml body_parts renamed",
          d["pose"]["body_parts"] == ["nose", "left_ear", "right_ear", "tailbase"])

    sk = read_skeleton(cp)
    check("skeleton edges renamed",
          ("nose", "left_ear") in sk["edges"]
          and ("left_ear", "right_ear") in sk["edges"]
          and ("right_ear", "tailbase") in sk["edges"])

    src = (REPO / "mufasa" / "model" / "marker_rename.py").read_text()
    check("cp is a Path (not os.fspath str); read_write import is lazy",
          "cp = Path(config_path)" in src
          and "from mufasa.utils.read_write import read_df, write_df" in src
          and src.index("if pose_files:") < src.index(
              "from mufasa.utils.read_write import read_df, write_df"))

    print(f"smoke_122gt_marker_rename_pathfix: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
