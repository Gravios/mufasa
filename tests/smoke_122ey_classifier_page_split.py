"""
tests/smoke_122ey_classifier_page_split.py
=============================================

Patch 122ey — split the monolithic ``Classifier`` page into six
standalone sidebar pages, one per former section.

User request (Mon May 25, 2026):

> the Classifier tab should just have each section split into
> its own tab.

The split surfaces the dependency order the workflow actually
follows:

  Features
  Manage classifiers          ← Classifier setup (pre-Annotation)
  Annotation                  ← existing position
  Train classifier            ← train from labelled data
  Validate classifier         ← out-of-sample check
  Run inference               ← apply trained model
  YOLO pose — train           ← independent YOLO workflow
  YOLO pose — inference
  Analysis

The QWI-4 ordering constraint (classifier setup BEFORE Annotation)
is preserved — it now lives on the build_manage_classifiers_page
call. The operational sections (train/validate/inference) come AFTER
Annotation because they consume labelled data.

What this patch changed
-----------------------
* mufasa/ui_qt/pages/classifier_page.py: rewritten. The single
  build_classifier_page function was removed; replaced with six
  new build_*_page functions. Each registers ONE workbench page
  with ONE section.

* mufasa/ui_qt/workbench_app.py: the single
  build_classifier_page() call became six calls in workflow order.
  Annotation moved into the middle of the cluster (between Manage
  and the operational tabs).

* mufasa/section_provenance.py SECTIONS: the page= attribute on
  the two classifier-related entries was updated:
  - classifier_train: "Classifier" → "Train classifier"
  - classifier_run: "Classifier" → "Run inference"
  These now match the new standalone pages' names.

* tests/smoke_122el_section_binding_audit.py: classifier_run's
  expected page changed from "Classifier" to "Run inference".

* tests/smoke_122d0_workbench_page_order.py: the QWI-4 ordering
  invariant now applies to build_manage_classifiers_page instead
  of the now-gone build_classifier_page.

Coverage
--------
Page rewrite (3 checks):
1.  classifier_page.py exposes all 6 new build_*_page functions
    AND the legacy build_classifier_page is gone (clean rename,
    no compatibility shim — workbench_app.py was the only caller).
2.  Each new page-builder registers exactly one section with the
    matching title.
3.  workbench_app.py calls all 6 new builders in the correct
    order (Manage before Annotation; the rest after).

SECTIONS contract (2 checks):
4.  SECTIONS['classifier_train'].page == 'Train classifier'.
5.  SECTIONS['classifier_run'].page == 'Run inference'.

Sidebar order (1 check):
6.  Annotation appears between Manage classifiers and Train
    classifier in workbench_app.py registration order.

Cross-patch invariants (4 checks):
7.  122ex state preserved: EgocentricAlignmentForm.section_id ==
    "egocentric".
8.  122ew state preserved: get_all_statuses uses _resolve_run_at.
9.  Parse-clean.
10. 122do baseline.
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


def main() -> int:
    cp_path = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "classifier_page.py")
    cp_src = cp_path.read_text()
    cp_tree = ast.parse(cp_src)

    expected_builders = {
        "build_manage_classifiers_page",
        "build_train_classifier_page",
        "build_validate_classifier_page",
        "build_run_inference_page",
        # Patch 122fa — was build_yolo_train_page + build_yolo_inference_page
        # (two pages); user requested they be merged into one "YOLO pose"
        # tab with two sections inside.
        "build_yolo_pose_page",
    }
    actual_builders = {
        node.name for node in ast.walk(cp_tree)
        if isinstance(node, ast.FunctionDef)
    }
    legacy_present = "build_classifier_page" in actual_builders
    new_present = expected_builders.issubset(actual_builders)
    check(
        "classifier_page.py exposes 5 build_*_page functions "
        "(post-122fa YOLO merge) AND no longer defines the "
        "legacy build_classifier_page (clean structure)",
        new_present and not legacy_present,
        detail=(
            f"new_present={new_present} "
            f"legacy_present={legacy_present} "
            f"actual={sorted(actual_builders)}"
        ),
    )

    # 2. Each builder registers sections with the matching titles.
    # Most pages have one section; YOLO pose has two after 122fa.
    expected_sections = {
        "build_manage_classifiers_page": ["Manage classifiers"],
        "build_train_classifier_page":   ["Train classifier"],
        "build_validate_classifier_page": ["Validate classifier"],
        "build_run_inference_page":      ["Run inference"],
        "build_yolo_pose_page":          ["YOLO pose — train",
                                          "YOLO pose — inference"],
    }
    one_section_each = True
    mismatches = []
    for fn in cp_tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name not in expected_sections:
            continue
        expected_titles = expected_sections[fn.name]
        section_calls = []
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in (
                        "add_section", "add_section_widget")):
                # First arg is the section title
                if (node.args
                        and isinstance(node.args[0], ast.Constant)):
                    section_calls.append(node.args[0].value)
        if section_calls != expected_titles:
            one_section_each = False
            mismatches.append(
                f"{fn.name}: titles {section_calls!r} ≠ expected "
                f"{expected_titles!r}"
            )
    check(
        "Each build_*_page function registers the expected sections "
        "(most one each; YOLO pose has two post-122fa merge)",
        one_section_each,
        detail=("; ".join(mismatches[:3])),
    )

    # 3. workbench_app.py calls all 5 builders.
    wba_src = (REPO_ROOT / "mufasa" / "ui_qt"
               / "workbench_app.py").read_text()
    wba_tree = ast.parse(wba_src)
    called_builders = set()
    for node in ast.walk(wba_tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in expected_builders):
            called_builders.add(node.func.id)
    check(
        "workbench_app.py invokes all 5 build_*_page functions "
        "(post-122fa YOLO merge reduced the count from 6 to 5)",
        expected_builders == called_builders,
        detail=(
            f"missing: {sorted(expected_builders - called_builders)}; "
            f"extra: {sorted(called_builders - expected_builders)}"
        ),
    )

    # -----------------------------------------------------------------
    # SECTIONS contract
    # -----------------------------------------------------------------
    from mufasa.section_provenance import SECTIONS
    ct = SECTIONS["classifier_train"]
    check(
        "SECTIONS['classifier_train'].page == 'Train' "
        "(post-122fa rename from 'Train classifier')",
        ct.page == "Train",
        detail=(f"got {ct.page!r}"),
    )

    cr = SECTIONS["classifier_run"]
    check(
        "SECTIONS['classifier_run'].page == 'Inference' "
        "(post-122fa rename from 'Run inference')",
        cr.page == "Inference",
        detail=(f"got {cr.page!r}"),
    )

    # -----------------------------------------------------------------
    # Sidebar order
    # -----------------------------------------------------------------
    lines = wba_src.split("\n")
    manage_line = None
    annotation_line = None
    train_line = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("build_manage_classifiers_page("):
            if manage_line is None:
                manage_line = i
        elif stripped.startswith("build_annotation_page("):
            if annotation_line is None:
                annotation_line = i
        elif stripped.startswith("build_train_classifier_page("):
            if train_line is None:
                train_line = i
    check(
        "Sidebar order: Manage classifiers → Annotation → Train "
        "classifier (workflow dependency order; the QWI-4 'setup "
        "before annotation' invariant is preserved)",
        (manage_line is not None
         and annotation_line is not None
         and train_line is not None
         and manage_line < annotation_line < train_line),
        detail=(
            f"manage={manage_line} "
            f"annotation={annotation_line} "
            f"train={train_line}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
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
        "122ex state preserved: EgocentricAlignmentForm.section_id "
        "== 'egocentric'",
        egocentric_sid == "egocentric",
    )

    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    check(
        "122ew state preserved: get_all_statuses delegates to "
        "_resolve_run_at",
        "_resolve_run_at" in sp_src,
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

    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122ey_classifier_page_split: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
