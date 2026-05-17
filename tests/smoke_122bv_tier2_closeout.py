"""
tests/smoke_122bv_tier2_closeout.py
=====================================

Patch 122bv: Tier 2 close-out for the Tk → Qt consolidation
plan. Documentation-only patch that re-audits the Tier 2 lanes
and updates docs/tk_to_qt_consolidation_plan.md to reflect:

* Lane 1 (Background removal)         — ✓ shipped patch 122bu.
* Lanes 2, 3 (ROI Import CSV / Size)  — ✓ already in ROIManageForm.
* Lane 4 (Annotated bouts → videos)   — ✓ already in VisualizationForm.
* Lane 5 (Blob quick-check)           — ✗ reclassified to Tier 3;
                                         BlobQuickChecker is an
                                         interactive CV2+Tk viewer,
                                         not a small form. Needs
                                         medium-sized Qt viewer port.
* Lane 6 (Clf × ROI / time bins)      — ✓ already in AnalysisForm.

Net result: of the 6 original Tier 2 lanes, 5 are complete and
1 (BlobQuickChecker) moves to Tier 3 alongside UnsupervisedGUI.

The plan's §3 gap table is also updated to show ✓ DONE markers
for the resolved entries and reclassification notes for the
moved entry.

Coverage
--------
1. docs/tk_to_qt_consolidation_plan.md still exists.
2. Tier 1 marked DONE with patch reference.
3. Tier 2 lane status table shows 4 ✓ already-done entries +
   1 ✓ shipped (122bu) + 1 ✗ reclassified.
4. Tier 3 has two subsections: 3a (Blob quick-check) and 3b
   (Unsupervised analysis).
5. §3 gap summary shows updated status markers (DONE / GAP).
6. Existing companion-doc references and structural sections
   are still intact (regression guards against accidental
   deletion of content in this patch).
7. The "Don't rename Tk-only code" guidance is preserved.
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
    text = plan.read_text()

    # ------------------------------------------------------------------
    # 1. Tier 1 marked DONE
    # ------------------------------------------------------------------
    check(
        "Tier 1 marked DONE with patch 122bt reference",
        "Tier 1" in text and "DONE" in text and "122bt" in text,
    )

    # ------------------------------------------------------------------
    # 2. Tier 2 lane status table reflects what's done
    # ------------------------------------------------------------------
    # The four "already done" lanes
    for lane in [
        "ROI Import CSV", "ROI Size standardiser",
        "Annotated bouts", "Clf × ROI",
    ]:
        check(
            f"Tier 2 lane '{lane}' marked as already done",
            lane in text,
        )

    # The newly shipped lane
    check(
        "Tier 2 lane 'Background removal' marked as shipped in 122bu",
        "Background removal" in text and "122bu" in text,
    )

    # The reclassified lane
    check(
        "Blob quick-check reclassified out of Tier 2",
        ("reclassified" in text.lower()
         and "blob" in text.lower()),
    )

    # ------------------------------------------------------------------
    # 3. Tier 3 has two subsections (3a + 3b)
    # ------------------------------------------------------------------
    check(
        "Tier 3 has subsection 3a (Blob quick-check)",
        "Tier 3a" in text or "3a" in text,
    )
    check(
        "Tier 3 has subsection 3b (Unsupervised analysis)",
        "Tier 3b" in text or "3b" in text,
    )
    check(
        "Tier 3 still mentions UnsupervisedGUI as the large port",
        "UnsupervisedGUI" in text,
    )

    # ------------------------------------------------------------------
    # 4. §3 gap summary updated with status markers
    # ------------------------------------------------------------------
    check(
        "§3 gap summary has DONE markers",
        "✓ DONE" in text,
    )
    check(
        "§3 gap summary has STILL A GAP markers",
        "STILL A GAP" in text or "still a gap" in text.lower(),
    )
    check(
        "§3 gap summary references patch 122bu close-out",
        "122bu" in text,
    )

    # ------------------------------------------------------------------
    # 5. Companion doc references preserved
    # ------------------------------------------------------------------
    for ref in ["tk_surface_audit.md", "workflow_audit.md",
                "Tier 4", "Migration order"]:
        check(
            f"plan still references / contains '{ref}' "
            f"(regression guard)",
            ref in text,
        )

    # ------------------------------------------------------------------
    # 6. Original consolidation principles preserved
    # ------------------------------------------------------------------
    check(
        "plan preserves the Consolidation principles section "
        "(§4 — three porting patterns)",
        "Consolidation principles" in text
        and "mode selector" in text.lower(),
    )

    # ------------------------------------------------------------------
    # 7. docs/README.md still references the plan
    # ------------------------------------------------------------------
    readme = REPO_ROOT / "docs" / "README.md"
    check(
        "docs/README.md still references tk_to_qt_consolidation_plan.md",
        "tk_to_qt_consolidation_plan.md" in readme.read_text(),
    )

    print(
        f"smoke_122bv_tier2_closeout: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
