"""
tests/smoke_122bt_tier1_dispatcher_additions.py
=================================================

Patch 122bt: Tier 1 of the Tk → Qt consolidation plan. Adds
missing routes/backends to the three dispatcher forms:

VisualizationForm (29 routes; mp backends added)
  + ROI overlay (per video)             — add backend_mp
  + ROI feature overlay (per video)     — add backend_mp
  + Classifier validation clips         — add backend_mp

AnalysisForm (11 → 12 routes)
  + Directing toward body-part — statistics
    (DirectingAnimalsToBodyPartAnalyzer — companion to the
     already-routed DirectingOtherAnimalsAnalyzer)

ConverterForm (10 → 13 routes)
  + SLEAP (CSV) → YOLO keypoints   (Sleap2Yolo — CSV variant
    of the existing .slp + H5 routes)
  + DLC (multi-animal CSV) → YOLO keypoints  (MultiDLC2Yolo —
    CSV/folder variant of the existing H5 multi-animal route)
  + Labelme (keypoints) → DLC      (Labelme2DLC — reverse of
    the existing DLC → Labelme route)

Coverage
--------
1. ROUTES tables have the expected counts:
   VisualizationForm >= 29, AnalysisForm >= 12, ConverterForm >= 13.
2. The new mp backends are referenced in the visualizations
   route table.
3. The new AnalysisForm route exists and references
   DirectingAnimalsToBodyPartAnalyzer.
4. The three new ConverterForm routes exist and reference
   the correct backend factories.
5. All targeted backend modules exist and define the expected
   class names (regression guard against module renames).
6. All mufasa/**/*.py files parse cleanly.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def routes_source(path: Path) -> str:
    """Return the raw source of the ROUTES list literal."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                target = node.targets[0].id
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target = node.target.id
                value = node.value
        if target == "ROUTES":
            return ast.unparse(value) if value else ""
    return ""


def routes_count(path: Path) -> int:
    """Count the number of top-level elements in the ROUTES list."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                target = node.targets[0].id
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target = node.target.id
                value = node.value
        if (target == "ROUTES"
                and isinstance(value, (ast.List, ast.Tuple))):
            return len(value.elts)
    return 0


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    viz_path = pkg / "ui_qt" / "forms" / "visualizations.py"
    ana_path = pkg / "ui_qt" / "forms" / "analysis.py"
    con_path = pkg / "ui_qt" / "forms" / "data_import.py"

    viz_src = routes_source(viz_path)
    ana_src = routes_source(ana_path)
    con_src = routes_source(con_path)

    # ==================================================================
    # 1. Route counts (resilient floors)
    # ==================================================================
    check(
        f"VisualizationForm has >= 29 routes "
        f"(got {routes_count(viz_path)})",
        routes_count(viz_path) >= 29,
    )
    check(
        f"AnalysisForm has >= 12 routes "
        f"(got {routes_count(ana_path)})",
        routes_count(ana_path) >= 12,
    )
    check(
        f"ConverterForm has >= 13 routes "
        f"(got {routes_count(con_path)})",
        routes_count(con_path) >= 13,
    )

    # ==================================================================
    # 2. New mp backends in VisualizationForm ROUTES
    # ==================================================================
    new_mp_classes = [
        "ROIPlotMultiprocess",
        "ROIfeatureVisualizerMultiprocess",
        "ClassifierValidationClipsMultiprocess",
    ]
    for cls in new_mp_classes:
        check(
            f"VisualizationForm ROUTES references mp class "
            f"'{cls}'",
            cls in viz_src,
        )

    # ==================================================================
    # 3. AnalysisForm route for DirectingAnimalsToBodyPartAnalyzer
    # ==================================================================
    check(
        "AnalysisForm ROUTES has a route labeled "
        "'Directing toward body-part — statistics'",
        "Directing toward body-part" in ana_src,
    )
    check(
        "AnalysisForm ROUTES references "
        "DirectingAnimalsToBodyPartAnalyzer",
        "DirectingAnimalsToBodyPartAnalyzer" in ana_src,
    )

    # ==================================================================
    # 4. New ConverterForm routes
    # ==================================================================
    new_converter_classes = [
        ("Sleap2Yolo",  "SLEAP (CSV)"),
        ("MultiDLC2Yolo", "DLC (multi-animal CSV)"),
        ("Labelme2DLC", "Labelme (keypoints)"),
    ]
    for cls, label in new_converter_classes:
        check(
            f"ConverterForm ROUTES references '{cls}' class",
            cls in con_src,
        )
        check(
            f"ConverterForm ROUTES has a route source labeled "
            f"'{label}'",
            label in con_src,
        )

    # ==================================================================
    # 5. All targeted backend modules + classes exist
    # ==================================================================
    backends = [
        ("mufasa/plotting/roi_plotter_mp.py",
         "ROIPlotMultiprocess"),
        ("mufasa/plotting/ROI_feature_visualizer_mp.py",
         "ROIfeatureVisualizerMultiprocess"),
        ("mufasa/plotting/clf_validator_mp.py",
         "ClassifierValidationClipsMultiprocess"),
        ("mufasa/data_processors/directing_animal_to_bodypart.py",
         "DirectingAnimalsToBodyPartAnalyzer"),
        ("mufasa/third_party_label_appenders/transform/"
         "sleap_csv_to_yolo.py", "Sleap2Yolo"),
        ("mufasa/third_party_label_appenders/transform/"
         "dlc_multi_to_yolo.py", "MultiDLC2Yolo"),
        ("mufasa/third_party_label_appenders/transform/"
         "labelme_to_dlc.py", "Labelme2DLC"),
    ]
    for path_str, cls in backends:
        p = REPO_ROOT / path_str
        check(
            f"Backend module {path_str} exists",
            p.exists(),
        )
        if p.exists():
            check(
                f"Module {p.name} defines class '{cls}'",
                f"class {cls}" in p.read_text(),
            )

    # ==================================================================
    # 6. All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122bt_tier1_dispatcher_additions: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
