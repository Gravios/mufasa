"""Smoke test for patch 122km — re-import the skeleton model.

User: "under import data I would like to be able to reimport the model. I
am considering adding the eyes and maybe the shoulders to improve rigid
and soft body solutions." — "The skeleton model".

A project with the 15-marker rat model and explicit Kalman v2 segments
takes a retrained model's pose file (FreeDLC parquet with an embedded
skeleton) that adds eyes and shoulders and drops tail_V32:

* the plan: added / removed markers, the file's skeleton, eyes → 'head',
  shoulders → 'back' (nearest marker on the new skeleton), the emptied
  segment, a removed marker still named in a stage setting;
* applying: project.toml backed up; body_parts, [skeleton], segments
  written; the emptied segment dropped (none of its children orphaned);
  other tables untouched; the Kalman v2 layout builds with the new markers;
* a file without a skeleton (DeepLabCut CSV) keeps the project's edges; a
  new marker without a skeleton path to a segment is reported;
* segments left alone when asked; an unchanged model offers nothing to do;
* the Data Import page's "Re-import skeleton model" section.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


RAT = ["head_nose", "head_mid", "head_left", "head_right", "head_back", "back_T4", "back_T8", "back_L2",
       "back_L6", "back_V2", "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32"]
EDGES = [("head_nose", "head_mid"), ("head_mid", "head_left"), ("head_mid", "head_right"),
         ("head_mid", "head_back"), ("head_back", "back_T4"), ("back_T4", "back_T8"), ("back_T8", "back_L2"),
         ("back_L2", "back_L6"), ("back_L6", "back_V2"), ("back_L2", "hip_left"), ("back_L2", "hip_right"),
         ("back_V2", "tail_V6"), ("tail_V6", "tail_V18"), ("tail_V18", "tail_V32")]
SEGMENTS = [
    {"name": "back", "markers": ["back_T8", "back_T4", "back_L6", "back_L2", "hip_left", "hip_right"]},
    {"name": "back_rear", "parent": "back", "markers": ["back_V2"]},
    {"name": "neck", "parent": "back", "markers": ["head_back"]},
    {"name": "head", "parent": "neck", "markers": ["head_mid", "head_nose", "head_left", "head_right"]},
    {"name": "tail_1", "parent": "back_rear", "markers": ["tail_V6"]},
    {"name": "tail_2", "parent": "tail_1", "markers": ["tail_V18"]},
    {"name": "tail_3", "parent": "tail_2", "markers": ["tail_V32"]},
]


def make_project(root: Path) -> Path:
    from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
    root.mkdir(parents=True)
    ProjectPaths(root).ensure_skeleton()
    cfg = root / PROJECT_CONFIG_FILENAME
    write_project_toml(cfg, {
        "project_layout_version": 1,
        "project": {"name": "rat"},
        "stages": {"outlier_correction": {"none_location_bp_1": "tail_V32", "none_location_bp_2": "head_mid"}},
        "pose": {"file_type": "parquet", "animal_count": 1, "body_parts": RAT, "segments": SEGMENTS},
        "skeleton": {"nodes": RAT, "edges": [list(e) for e in EDGES]},
        "classifiers": {"targets": ["Rear", "Groom"]},
    })
    return cfg


def fdlc_parquet(path: Path, markers, edges) -> Path:
    from mufasa.pose_importers.pose_parquet import write_parquet_with_pose_metadata
    df = pd.DataFrame([(f, "rat", b, 1.0, 2.0, .9) for f in range(3) for b in markers],
                      columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])
    write_parquet_with_pose_metadata(df, path, list(markers), list(edges))
    return path


def dlc_csv(path: Path, markers) -> Path:
    head = ["scorer"] + ["DLC"] * 3 * len(markers)
    bps = ["bodyparts"] + [m for m in markers for _ in range(3)]
    coords = ["coords"] + ["x", "y", "likelihood"] * len(markers)
    rows = [head, bps, coords] + [[str(i)] + ["1.0"] * 3 * len(markers) for i in range(3)]
    path.write_text("\n".join(",".join(r) for r in rows) + "\n")
    return path


def main() -> int:
    from mufasa.data_processors.kalman_pose_smoother_v2 import layout_from_config
    from mufasa.model.skeleton_reimport import apply_reimport, plan_reimport
    from mufasa.project_layout import read_project_toml

    new = [m for m in RAT if m != "tail_V32"] + ["eye_left", "eye_right", "shoulder_left", "shoulder_right"]
    new_edges = [e for e in EDGES if "tail_V32" not in e] + [
        ("eye_left", "head_mid"), ("eye_right", "head_mid"), ("eye_left", "head_nose"),
        ("shoulder_left", "back_T4"), ("shoulder_right", "back_T4")]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = make_project(td / "p1")
        pose = fdlc_parquet(td / "new_model.parquet", new, new_edges)
        plan = plan_reimport(cfg, pose)
        check("plan: eyes and shoulders added, tail_V32 removed",
              plan.added == ["eye_left", "eye_right", "shoulder_left", "shoulder_right"] and plan.removed == ["tail_V32"])
        check("plan: the file's skeleton", plan.edges_from_file and len(plan.new_edges) == len(new_edges))
        check("plan: eyes join 'head', shoulders join 'back' (nearest marker on the new skeleton)",
              plan.segment_joins == {"eye_left": "head", "eye_right": "head",
                                     "shoulder_left": "back", "shoulder_right": "back"})
        check("plan: emptied segment and a removed marker named in a stage setting reported",
              plan.emptied_segments == ["tail_3"]
              and plan.references == {"tail_V32": ["stages.outlier_correction.none_location_bp_1"]}
              and "tail_3" in plan.text() and "none_location_bp_1" in plan.text())

        backup = apply_reimport(cfg, plan)
        d = read_project_toml(cfg)
        segs = {s["name"]: s for s in d["pose"]["segments"]}
        check("apply: project.toml backed up in logs/", backup.parent == cfg.parent / "logs"
              and read_project_toml(backup)["pose"]["body_parts"] == RAT)
        check("apply: body_parts and [skeleton] from the file",
              d["pose"]["body_parts"] == new and len(d["skeleton"]["edges"]) == len(new_edges)
              and d["skeleton"]["nodes"] == new)
        check("apply: segments extended, emptied 'tail_3' dropped",
              segs["head"]["markers"][-2:] == ["eye_left", "eye_right"]
              and segs["back"]["markers"][-2:] == ["shoulder_left", "shoulder_right"]
              and "tail_3" not in segs and all(s.get("parent") in segs or s.get("parent") is None for s in segs.values()))
        check("apply: other tables untouched",
              d["classifiers"]["targets"] == ["Rear", "Groom"]
              and d["stages"]["outlier_correction"]["none_location_bp_2"] == "head_mid")
        lay = layout_from_config(str(cfg))
        check(f"Kalman v2 layout builds with the new markers ({lay.n_markers} markers)",
              lay.n_markers == len(new) and "head" in [s.name for s in lay.segments])

        # a middle segment emptied: its children move to its parent
        cfg2 = make_project(td / "p2")
        mid = [m for m in RAT if m != "back_V2"]
        p2 = plan_reimport(cfg2, fdlc_parquet(td / "m2.parquet", mid, [e for e in EDGES if "back_V2" not in e]))
        apply_reimport(cfg2, p2)
        s2 = {s["name"]: s for s in read_project_toml(cfg2)["pose"]["segments"]}
        check("emptied middle segment: its child re-attached to its parent",
              "back_rear" not in s2 and s2["tail_1"]["parent"] == "back")

        # no skeleton in the file; a marker with no path to a segment; segments left alone
        cfg3 = make_project(td / "p3")
        p3 = plan_reimport(cfg3, dlc_csv(td / "dlc.csv", RAT + ["eye_left"]))
        check("file without a skeleton: the project's edges kept; the new marker reported as not placed",
              not p3.edges_from_file and len(p3.new_edges) == len(EDGES) and p3.unplaced == ["eye_left"]
              and "smoother will ignore" in p3.text())
        apply_reimport(cfg3, plan_reimport(cfg3, pose), extend_segments=False)
        s3 = {s["name"]: s for s in read_project_toml(cfg3)["pose"]["segments"]}
        check("extend_segments=False: new markers stay out of the segments",
              "eye_left" not in s3["head"]["markers"])

        cfg4 = make_project(td / "p4")
        same = plan_reimport(cfg4, fdlc_parquet(td / "same.parquet", RAT, EDGES))
        check("unchanged model: nothing to change", not same.changed and "nothing to change" in same.text())

        # the form
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.ui_qt.forms.skeleton_reimport import SkeletonReimportForm
        cfg5 = make_project(td / "p5")
        f = SkeletonReimportForm(config_path=str(cfg5))
        f.file.setText(str(pose))
        f.preview()
        shown = f.report.toPlainText()
        ok_preview = "added: eye_left" in shown and f.apply_btn.isEnabled()
        f.apply(confirm=False)
        check("form: preview, then Apply writes project.toml",
              ok_preview and read_project_toml(cfg5)["pose"]["body_parts"] == new and not f.apply_btn.isEnabled())
        f.file.setText(str(td / "missing.parquet"))
        f.preview()
        check("form: a missing file is reported", "not found" in f.report.toPlainText() and not f.apply_btn.isEnabled())

    page = (REPO / "mufasa" / "ui_qt" / "pages" / "data_import_page.py").read_text()
    check("Data Import page: 'Re-import skeleton model' section",
          'page.add_section("Re-import skeleton model",' in page and "SkeletonReimportForm" in page)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122km_skeleton_reimport: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
