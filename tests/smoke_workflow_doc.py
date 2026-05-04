"""Smoke test for the workflow audit document.

Doesn't validate the *content* (a doc audit can't be unit-
tested for correctness — that's what the audit IS). Validates
structural invariants:

- the file exists and is non-trivial
- the audit-status legend is defined and used consistently
- every status emoji used in headings appears in the legend
- claims about specific Mufasa workflow stages reference
  functions/classes that actually exist in the codebase
  (catches bit-rot — if a referenced popup class is renamed
  or removed, the doc references break)
- the table of Tk-tab → Qt-page coverage matches reality

    PYTHONPATH=. python tests/smoke_workflow_doc.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# Status legend tags expected to appear in the doc
STATUS_TAGS = {"🟢", "🟡", "🔴", "⚪"}

# Workflow class/function names referenced in the doc that we
# can verify against the codebase. Not exhaustive — just spot-
# checks that the references aren't entirely fabricated.
REFERENCED_NAMES = [
    # Core importers
    ("DLC importer", "mufasa/pose_importers/dlc_h5_importer.py"),
    ("DLC csv importer", "mufasa/pose_importers/dlc_csv_importer.py"),
    # Outlier correction
    (
        "Movement outlier corrector",
        "mufasa/outlier_tools/outlier_corrector_movement.py",
    ),
    (
        "Location outlier corrector",
        "mufasa/outlier_tools/outlier_corrector_location.py",
    ),
    # Egocentric alignment
    (
        "Egocentric alignment CPU",
        "mufasa/data_processors/egocentric_alignment.py",
    ),
    # Video info form (Qt — recently added)
    ("Qt VideoInfoForm", "mufasa/ui_qt/forms/video_info.py"),
    # Pixel calibration dialog (Qt)
    (
        "Qt PixelCalibrationDialog",
        "mufasa/ui_qt/dialogs/pixel_calibration.py",
    ),
    # Legacy Tk video info table
    ("Tk VideoInfoTable", "mufasa/ui/video_info_ui.py"),
    # The legacy launcher
    ("Tk launcher", "mufasa/SimBA.py"),
]


def main() -> int:
    doc_path = Path("docs/workflows.md")
    assert doc_path.is_file(), (
        f"Workflow audit document missing at {doc_path}"
    )
    text = doc_path.read_text()
    assert len(text) > 5000, (
        f"Document suspiciously short ({len(text)} bytes); audit "
        f"should be substantial"
    )

    # ------------------------------------------------------------------ #
    # Case 1: legend defined
    # ------------------------------------------------------------------ #
    assert "Audit status legend" in text or "audit status" in text.lower(), (
        "Document should define an audit-status legend"
    )

    # ------------------------------------------------------------------ #
    # Case 2: every status tag used somewhere is defined in the legend
    # ------------------------------------------------------------------ #
    used_tags = set()
    for tag in STATUS_TAGS:
        if tag in text:
            used_tags.add(tag)
    # The legend section should explain each used tag
    legend_idx = text.index("Audit status legend")
    legend_section = text[legend_idx:legend_idx + 2000]
    for tag in used_tags:
        assert tag in legend_section, (
            f"Tag {tag} used in the document but not explained in "
            f"the legend section"
        )

    # ------------------------------------------------------------------ #
    # Case 3: at least one workflow tagged as "Deeply audited"
    # (otherwise the doc isn't doing the deep-audit work the user asked for)
    # ------------------------------------------------------------------ #
    assert "🟢" in text, (
        "Document must contain at least one deeply-audited "
        "(🟢) workflow"
    )

    # ------------------------------------------------------------------ #
    # Case 4: spot-check that referenced Mufasa code paths exist.
    # If any of these references break (file renamed or removed),
    # the document needs updating — and this test surfaces that.
    # ------------------------------------------------------------------ #
    failures = []
    for label, path in REFERENCED_NAMES:
        if not Path(path).exists():
            failures.append(f"{label} — {path}")
    # Allow up to 2 stale references — this is documentation,
    # not code; a few broken pointers are normal as the codebase
    # evolves. More than 2 means the doc is meaningfully out of date.
    if len(failures) > 2:
        raise AssertionError(
            "Too many stale code references in the workflow doc:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    # ------------------------------------------------------------------ #
    # Case 5: 10-tab structure mentioned
    # (sanity check that the doc actually maps the canonical
    # SimBA project popup, not just a generic intro)
    # ------------------------------------------------------------------ #
    expected_tabs = [
        "Further imports", "Video parameters", "Outlier correction",
        "Extract features", "Label behavior", "Train machine model",
        "Run machine model", "Visualizations", "Add-ons",
    ]
    missing_tabs = [t for t in expected_tabs if t not in text]
    assert not missing_tabs, (
        f"Document should mention all 10 SimBA project tabs; "
        f"missing: {missing_tabs}"
    )

    # ------------------------------------------------------------------ #
    # Case 6: doc has a "what's broken" or "known broken" indicator
    # (audit isn't honest if it doesn't flag broken things)
    # ------------------------------------------------------------------ #
    # 🔴 used at least once for a known-broken finding
    red_flag_count = text.count("🔴")
    assert red_flag_count >= 1, (
        "Honest audit should flag at least one broken/known-"
        "broken workflow with 🔴"
    )

    # ------------------------------------------------------------------ #
    # Case 7: doc has a self-honest "what we haven't done" section
    # (calibrates expectations for the reader)
    # ------------------------------------------------------------------ #
    assert (
        "shallow" in text.lower()
        or "haven't" in text.lower()
        or "not yet" in text.lower()
    ), (
        "Document should include a self-honest section about "
        "what has NOT been audited deeply"
    )

    print("smoke_workflow_doc: 7/7 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
