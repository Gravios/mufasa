"""
tests/smoke_122dk_testing_workflow_and_roi_proposal.py
========================================================

Patch 122dk: doc-only patch adding (1) testing workflow tailored
to parquet + cage-video data and (2) ROI tool audit + enhancement
proposal.

What this patch landed
----------------------
1. ``docs/testing_workflow.md`` — end-to-end test plan with 8
   steps + sanity-check checklist + risk-path watch list.
   Includes Step 0 parquet-format identification (Mufasa internal
   vs DLC 3.0+ parquet vs custom shape) so users don't try to
   import data that's already in the right shape.

2. ``docs/roi_enhancements_proposal.md`` — honest audit of ROI
   tool state (works / fixed / not implemented). Two design
   proposals for the gaps surfaced by real-user testing:

   - Apply to selected videos (subset-apply) — solves the
     multi-condition pain. ~280 LoC. Suggested as 122dk-future.
   - Drag-to-adjust placed ROIs — UX improvement within the
     existing QPainter framework (not a QGraphicsScene rewrite).
     ~280 LoC. Suggested as 122dl-future.

3. ``docs/README.md`` index updated to include both docs.

Coverage
--------
1.  testing_workflow.md exists.
2.  Workflow covers Step 0 parquet-format identification.
3.  Workflow covers v1 project creation (Step 1).
4.  Workflow covers ROI testing (Step 4).
5.  Workflow has the sanity-check checklist.
6.  Workflow has the "bugs to watch for" section flagging
    MARS/TRK sentinels and 122dc/122d9/122da/122de paths.
7.  roi_enhancements_proposal.md exists.
8.  Proposal has the audit table flagging subset-apply and
    drag-to-adjust as NOT IMPLEMENTED.
9.  Proposal documents the QPainter-vs-QGraphicsScene decision
    (no full rewrite).
10. Proposal 1 (subset-apply) has UI mock + backend API +
    cost estimate.
11. Proposal 2 (drag-to-adjust) has implementation outline +
    cost estimate + risk assessment.
12. docs/README.md indexes both new docs.
13. All mufasa/**/*.py parse cleanly (no code changes; check
    anyway).
"""
from __future__ import annotations

import ast
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
    pkg = REPO_ROOT / "mufasa"
    workflow = REPO_ROOT / "docs" / "testing_workflow.md"
    proposal = REPO_ROOT / "docs" / "roi_enhancements_proposal.md"
    docs_index = REPO_ROOT / "docs" / "README.md"

    # 1. Workflow doc exists
    check(
        "docs/testing_workflow.md exists",
        workflow.exists(),
    )

    if workflow.exists():
        wf = workflow.read_text()
        # 2. Step 0 parquet-format identification
        check(
            "Workflow covers Step 0 parquet-format identification "
            "(Mufasa internal / DLC 3.0+ / custom)",
            "Step 0" in wf
            and "parquet format" in wf.lower()
            and "Mufasa" in wf
            and "DLC 3.0" in wf,
        )
        # 3. v1 project creation
        check(
            "Workflow covers v1 project creation (Step 1)",
            "Step 1" in wf
            and "project.toml" in wf,
        )
        # 4. ROI testing
        check(
            "Workflow has Step 4 dedicated to ROI testing",
            "Step 4" in wf
            and "ROI" in wf
            and "Apply to all" in wf
            and "two conditions" in wf,
        )
        # 5. Sanity-check checklist
        check(
            "Workflow has the sanity-check checklist",
            "sanity-check checklist" in wf.lower()
            or "Quick sanity-check" in wf,
        )
        # 6. Bugs-to-watch section
        check(
            "Workflow has 'bugs to watch for' section flagging "
            "MARS/TRK sentinels and 122dc/122d9/122da/122de paths",
            "MARS" in wf
            and "TRK" in wf
            and ("122dc" in wf or "visualizations" in wf.lower())
            and ("122d9" in wf or "Apply-all" in wf)
            and ("122de" in wf or "PixelCalibration" in wf),
        )

    # 7. Proposal doc exists
    check(
        "docs/roi_enhancements_proposal.md exists",
        proposal.exists(),
    )

    if proposal.exists():
        pp = proposal.read_text()
        # 8. Audit table flags the two gaps
        check(
            "Proposal's audit table flags subset-apply + "
            "drag-to-adjust as NOT IMPLEMENTED",
            "NOT IMPLEMENTED" in pp
            and "subset" in pp.lower()
            and "drag" in pp.lower(),
        )
        # 9. QPainter-vs-QGraphicsScene decision documented
        check(
            "Proposal documents the QPainter-vs-QGraphicsScene "
            "decision (no full rewrite)",
            "QPainter" in pp
            and "QGraphicsScene" in pp
            and ("not the right path" in pp.lower()
                 or "no full rewrite" in pp.lower()
                 or "not a qgraphicsscene rewrite" in pp.lower()),
        )
        # 10. Proposal 1 — subset-apply: UI mock + backend + cost
        check(
            "Proposal 1 has UI mock + backend API + cost estimate",
            "Apply to selected" in pp
            and ("multiply_ROIs" in pp)
            and "LoC" in pp,
        )
        # 11. Proposal 2 — drag-to-adjust: outline + cost + risk
        check(
            "Proposal 2 has implementation outline + cost + risk "
            "assessment",
            "_DrawMode.SELECT" in pp
            and "_hit_test" in pp
            and "Risk" in pp,
        )

    # 12. docs/README indexes both
    if docs_index.exists():
        idx = docs_index.read_text()
        check(
            "docs/README.md indexes testing_workflow.md AND "
            "roi_enhancements_proposal.md",
            "testing_workflow.md" in idx
            and "roi_enhancements_proposal.md" in idx,
        )

    # 13. Parse-clean (no code changes; verify nothing broke)
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=parse_errors[0] if parse_errors else "",
    )

    print(
        f"smoke_122dk_testing_workflow_and_roi_proposal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
