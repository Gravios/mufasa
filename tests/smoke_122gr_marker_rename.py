"""
tests/smoke_122gr_marker_rename.py
==================================

Patch 122gr — Model modifications tab: rename markers + Save model.

A new sidebar page ("Model modifications") hosts a rename form: edit each
marker's new name, press "Save model", and the change propagates to
project.toml [pose].body_parts, the [skeleton] (nodes AND edges — the
marker-connector relationships follow the rename), and the imported pose
parquets.

Pure rename logic is unit-tested here; the form/page are checked
structurally (Qt not importable in the sandbox).
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd

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
    spec = importlib.util.spec_from_file_location(
        "mr", REPO / "mufasa" / "model" / "marker_rename.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)

    bps = ["nose", "ear_left", "ear_right", "tailbase"]

    ok = True
    for bad, needle in [({"foo": "bar"}, "unknown"),
                        ({"ear_left": "nose"}, "collide"),
                        ({"ear_left": "x", "ear_right": "x"}, "same name"),
                        ({"ear_left": ""}, "empty")]:
        try:
            mr.validate_rename_map(bps, bad)
            ok = False
        except ValueError as e:
            ok = ok and needle in str(e)
    mr.validate_rename_map(bps, {"ear_left": "ear_right", "ear_right": "ear_left"})  # swap ok
    check("validate rejects unknown/collision/dup/empty; allows swaps", ok)

    check("apply_rename_to_names preserves order",
          mr.apply_rename_to_names(bps, {"ear_left": "left_ear"})
          == ["nose", "left_ear", "ear_right", "tailbase"])

    check("skeleton edges follow the rename",
          mr.apply_rename_to_skeleton([["nose", "ear_left"], ["ear_left", "ear_right"]],
                                      {"ear_left": "left_ear"})
          == [("nose", "left_ear"), ("left_ear", "ear_right")])

    flat = pd.DataFrame(columns=["nose_x", "nose_y", "nose_p",
                                 "ear_left_x", "ear_left_y", "ear_left_p"])
    fout = mr.rename_pose_columns(flat, {"ear_left": "left_ear"})
    mi = pd.MultiIndex.from_tuples([
        ("IMPORTED_POSE", "IMPORTED_POSE", "ear_left_x"),
        ("IMPORTED_POSE", "IMPORTED_POSE", "nose_x"),
    ])
    mout = mr.rename_pose_columns(pd.DataFrame(columns=mi), {"ear_left": "left_ear"})
    check("pose columns renamed (flat + IMPORTED_POSE multi-index)",
          list(fout.columns)[3:] == ["left_ear_x", "left_ear_y", "left_ear_p"]
          and [t[2] for t in mout.columns] == ["left_ear_x", "nose_x"])

    check("rename_markers orchestrator exposed",
          hasattr(mr, "rename_markers"))

    form = (REPO / "mufasa" / "ui_qt" / "forms" / "model_modifications.py").read_text()
    check("form: 'Rename markers' title + 'Save model' button + rename_markers call",
          'title = "Rename markers"' in form and "Save model" in form
          and "rename_markers(" in form and "showEvent" in form)

    page = (REPO / "mufasa" / "ui_qt" / "pages" / "model_modifications_page.py").read_text()
    app = (REPO / "mufasa" / "ui_qt" / "workbench_app.py").read_text()
    check("page 'Model modifications' built + wired into workbench_app",
          'add_page("Model modifications"' in page
          and "build_model_modifications_page" in app)

    print(f"smoke_122gr_marker_rename: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
