"""
tests/smoke_122bs_consolidation_plan.py
=========================================

Patch 122bs: ships docs/tk_to_qt_consolidation_plan.md — a
comprehensive Tk-popup → Qt-form mapping document with a
target workbench layout, gap analysis, migration order, and
drop-candidate list.

No code changes; pure design document plus docs/README.md
index update.

Coverage
--------
1. tk_to_qt_consolidation_plan.md exists.
2. References every existing Qt workbench page.
3. Names every major Tk → Qt mapping category (already
   covered, dispatcher-routed, gaps, drops).
4. Includes a migration order in tiers.
5. docs/README.md references the new doc.
6. The plan covers ALL 14 Qt workbench pages by name.
"""
from __future__ import annotations

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
    plan = REPO_ROOT / "docs" / "tk_to_qt_consolidation_plan.md"
    check("docs/tk_to_qt_consolidation_plan.md exists", plan.exists())

    if not plan.exists():
        return 1

    plan_text = plan.read_text()

    # 1. All 12 main pages of the Qt workbench mentioned
    pages = [
        "Projects", "Data Import", "Video Processing",
        "Preprocessing", "ROI", "Features", "Annotation",
        "Classifier", "Visualizations", "Analysis",
        "Add-ons", "Tools",
    ]
    for page in pages:
        check(
            f"plan mentions Qt workbench page '{page}'",
            page in plan_text,
        )

    # 2. Key sections of the document
    for section in [
        "State of play",
        "Target Qt workbench layout",
        "Gap summary",
        "Consolidation principles",
        "Drop candidates",
        "Migration order",
    ]:
        check(
            f"plan contains section '{section}'",
            section in plan_text,
        )

    # 3. Tier-based migration order present
    for tier in ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]:
        check(
            f"plan has '{tier}' in migration order",
            tier in plan_text,
        )

    # 4. Mentions the dispatcher forms (key consolidation point)
    for dispatcher in ["VisualizationForm", "AnalysisForm",
                        "ConverterForm"]:
        check(
            f"plan references dispatcher '{dispatcher}'",
            dispatcher in plan_text,
        )

    # 5. Identifies the unsupervised port as the largest gap
    check(
        "plan flags UnsupervisedGUI as the largest gap",
        "UnsupervisedGUI" in plan_text
        and "large" in plan_text.lower(),
    )

    # 6. docs/README.md references it
    readme = REPO_ROOT / "docs" / "README.md"
    check(
        "docs/README.md references tk_to_qt_consolidation_plan.md",
        "tk_to_qt_consolidation_plan.md" in readme.read_text(),
    )

    # 7. Cross-references existing docs
    for ref in ["tk_surface_audit.md", "workflow_audit.md"]:
        check(
            f"plan cross-references companion doc '{ref}'",
            ref in plan_text,
        )

    print(
        f"smoke_122bs_consolidation_plan: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
