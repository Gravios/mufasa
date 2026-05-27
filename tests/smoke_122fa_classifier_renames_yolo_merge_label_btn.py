"""
tests/smoke_122fa_classifier_renames_yolo_merge_label_btn.py
==============================================================

Patch 122fa — three small UX polish items requested by the user
on Tue May 26, 2026:

  > Change tab names (Train classifier -> Train), (Validate
  > classifier -> Validate), (Run inference -> Inference). Merge
  > Yolo pose train and inference under a tab called YOLO pose.
  >
  > In Annotation : Frame labeling, clicking "Label" should take
  > me to the appropriate video folder. The Label button should
  > be an appropriate size.

Three independent changes batched into one patch because they're
all small UI tweaks following the 122ey restructure:

1. Classifier tab renames (3 pages): use short verb-form labels
   matching the project's button-label style (Label, Run, etc.).

2. YOLO pose merge: was two pages (Train + Inference); user
   requested they share one tab with two sections.

3. Annotation Label button:
   - Click opens the file dialog in the project's video folder
     (was OS-default last-used location).
   - Button stays at natural width instead of stretching across
     the form (HBoxLayout + addStretch).

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/pages/classifier_page.py:
* build_train_classifier_page: add_page("Train classifier")
  → add_page("Train").
* build_validate_classifier_page: add_page("Validate classifier")
  → add_page("Validate").
* build_run_inference_page: add_page("Run inference")
  → add_page("Inference").
* build_yolo_train_page + build_yolo_inference_page → MERGED into
  build_yolo_pose_page, which registers add_page("YOLO pose")
  with TWO sections inside.

mufasa/ui_qt/workbench_app.py:
* Updated imports: build_yolo_train_page + build_yolo_inference_page
  removed; build_yolo_pose_page added.
* Updated build calls: 6 → 5 page builders.

mufasa/section_provenance.py SECTIONS:
* classifier_train.page: "Train classifier" → "Train".
* classifier_run.page: "Run inference" → "Inference".

mufasa/ui_qt/frame_labeller.py::launch_frame_labeller:
* Resolves the project's video_dir via v1_project_paths() and
  uses it as the QFileDialog start directory. Falls back to ""
  (OS default) if resolution fails.

mufasa/ui_qt/forms/annotation.py::FrameLabellingLauncher.build:
* Label button: setMinimumWidth(140); wrapped in QHBoxLayout
  with addStretch() so it stays at natural size, right-aligned
  (matches OperationForm.run_btn placement convention).

Coverage
--------
Classifier renames (4 checks):
1.  build_train_classifier_page calls add_page("Train").
2.  build_validate_classifier_page calls add_page("Validate").
3.  build_run_inference_page calls add_page("Inference").
4.  build_yolo_pose_page calls add_page("YOLO pose") with two
    sections (YOLO pose — train, YOLO pose — inference).

SECTIONS contract (2 checks):
5.  SECTIONS['classifier_train'].page == 'Train'.
6.  SECTIONS['classifier_run'].page == 'Inference'.

Annotation Label button (2 checks):
7.  launch_frame_labeller resolves video_dir via v1_project_paths
    and uses it as QFileDialog start dir (substring search for
    'v1_project_paths' import + 'start_dir' kwarg use).
8.  FrameLabellingLauncher's Label button has setMinimumWidth(140)
    and is wrapped in an HBoxLayout with addStretch (matches the
    OperationForm.run_btn placement convention).

Cross-patch invariants (3 checks):
9.  122ez state preserved: SECTIONS['egocentric'].detect_path
    is callable.
10. 122ex state preserved: EgocentricAlignmentForm.section_id
    == 'egocentric'.
11. Parse-clean.
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


def _find_add_page_calls(src: str) -> dict[str, list[str]]:
    """Map function name → list of add_page string args."""
    tree = ast.parse(src)
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        pages = []
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add_page"
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)):
                pages.append(sub.args[0].value)
        if pages:
            out[node.name] = pages
    return out


def _find_add_section_calls_in_fn(
    src: str, fn_name: str,
) -> list[str]:
    """Return the section titles registered by a specific build_* fn."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == fn_name):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in (
                            "add_section", "add_section_widget")
                        and sub.args
                        and isinstance(sub.args[0], ast.Constant)):
                    out.append(sub.args[0].value)
            break
    return out


def main() -> int:
    cp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
              / "classifier_page.py").read_text()
    pages = _find_add_page_calls(cp_src)

    # 1-3. Tab renames
    cases = [
        ("build_train_classifier_page",    "Train"),
        ("build_validate_classifier_page", "Validate"),
        ("build_run_inference_page",       "Inference"),
    ]
    for fn, expected in cases:
        actual = pages.get(fn, [])
        check(
            f"{fn} calls add_page('{expected}') (post-122fa "
            f"short verb-form rename)",
            actual == [expected],
            detail=(f"got {actual!r}"),
        )

    # 4. YOLO merge
    yolo_pages = pages.get("build_yolo_pose_page", [])
    yolo_sections = _find_add_section_calls_in_fn(
        cp_src, "build_yolo_pose_page",
    )
    check(
        "build_yolo_pose_page calls add_page('YOLO pose') with "
        "TWO sections inside (post-122fa merge of yolo train + "
        "yolo inference into one tab)",
        yolo_pages == ["YOLO pose"]
        and yolo_sections == [
            "YOLO pose — train", "YOLO pose — inference",
        ],
        detail=(
            f"page={yolo_pages!r} sections={yolo_sections!r}"
        ),
    )

    # 5-6. SECTIONS contract.
    from mufasa.section_provenance import SECTIONS
    ct = SECTIONS["classifier_train"]
    check(
        "SECTIONS['classifier_train'].page == 'Train' "
        "(post-122fa rename)",
        ct.page == "Train",
        detail=(f"got {ct.page!r}"),
    )

    cr = SECTIONS["classifier_run"]
    check(
        "SECTIONS['classifier_run'].page == 'Inference' "
        "(post-122fa rename)",
        cr.page == "Inference",
        detail=(f"got {cr.page!r}"),
    )

    # 7. launch_frame_labeller starts file dialog in video_dir.
    fl_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_labeller.py").read_text()
    check(
        "launch_frame_labeller resolves the project's video_dir "
        "via v1_project_paths and uses it as the QFileDialog "
        "start directory (user request: 'clicking Label should "
        "take me to the appropriate video folder')",
        "v1_project_paths" in fl_src
        and "start_dir" in fl_src,
    )

    # 8. Label button sizing.
    ann_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "annotation.py").read_text()
    # AST: find FrameLabellingLauncher.build and look for
    # setMinimumWidth(140) + an HBoxLayout assignment.
    ann_tree = ast.parse(ann_src)
    sizing_ok = False
    for cls in ast.walk(ann_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "FrameLabellingLauncher"):
            for fn in cls.body:
                if not (isinstance(fn, ast.FunctionDef)
                        and fn.name == "build"):
                    continue
                fn_src = ast.unparse(fn)
                if ("setMinimumWidth(140)" in fn_src
                        and "QHBoxLayout(" in fn_src
                        and "addStretch" in fn_src):
                    sizing_ok = True
                break
            break
    check(
        "FrameLabellingLauncher.build wraps the Label button in "
        "an HBoxLayout with addStretch and sets a min width — "
        "natural-sized, right-aligned (matches OperationForm."
        "run_btn placement; user request: 'the Label button "
        "should be an appropriate size')",
        sizing_ok,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    egospec = SECTIONS["egocentric"]
    check(
        "122ez state preserved: SECTIONS['egocentric']."
        "detect_path is callable",
        callable(egospec.detect_path),
    )

    pc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "pose_cleanup.py").read_text()
    pc_tree = ast.parse(pc_src)
    egocentric_sid = None
    for cls in ast.walk(pc_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "EgocentricAlignmentForm"):
            for m in cls.body:
                if isinstance(m, ast.Assign):
                    for tgt in m.targets:
                        if (isinstance(tgt, ast.Name)
                                and tgt.id == "section_id"
                                and isinstance(m.value, ast.Constant)):
                            egocentric_sid = m.value.value
    check(
        "122ex state preserved: EgocentricAlignmentForm."
        "section_id == 'egocentric'",
        egocentric_sid == "egocentric",
    )

    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fa_classifier_renames_yolo_merge_label_btn: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
