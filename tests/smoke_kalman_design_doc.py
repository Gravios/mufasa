"""Smoke test for the Kalman smoother design document.

Like smoke_workflow_doc.py, this doesn't validate the design
itself (a design doc can't be unit-tested for correctness)
— it validates structural invariants:

- doc exists, is substantial (>5KB), and is markdown
- the canonical "Status: design document, no implementation
  yet" header is present
- math notation sections are present (state space,
  measurement model, spatial prior)
- prior art section references the specific published works
  the doc engages with (Lightning Pose EKS, Anipose, etc.)
- recommendations + open questions are present (so the doc
  serves as a decision-driver, not just a description)

    PYTHONPATH=. python tests/smoke_kalman_design_doc.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    doc = Path("docs/kalman_smoother_design.md")
    assert doc.is_file(), (
        f"Kalman smoother design document missing at {doc}"
    )
    text = doc.read_text()
    assert len(text) > 5000, (
        f"Document suspiciously short ({len(text)} bytes); "
        f"design doc should be substantial"
    )

    # Status banner — explicit "design only" or "no implementation"
    # so future readers don't expect to find an implementation
    # (matches the doc's status section). Allow various phrasings
    # since the doc evolves.
    status_indicators = [
        "design document",
        "no implementation yet",
        "no code",
        "design only",
        "design committed",
    ]
    has_status = any(s in text.lower() for s in status_indicators)
    assert has_status, (
        f"Document should clearly identify its status (design "
        f"document, no implementation yet, etc.). Looked for: "
        f"{status_indicators}"
    )

    # Math sections — the doc has to contain the actual
    # mathematical formulation, not just hand-waving
    expected_math_concepts = [
        "Kalman",          # the canonical algorithm
        "RTS",             # Rauch-Tung-Striebel smoother
        "state space",     # the framework
        "covariance",      # the spatial prior structure
        "likelihood",      # the variance source
        "Mahalanobis",     # the spatial prior loss
        "egocentric",      # the triplet frame
        "triplet",         # the structural unit
    ]
    missing = [c for c in expected_math_concepts
               if c.lower() not in text.lower()]
    assert not missing, (
        f"Document should cover key mathematical concepts; "
        f"missing: {missing}"
    )

    # Prior art — must engage with specific named works
    expected_prior_art = [
        "Lightning Pose",   # closest precedent (EKS)
        "Anipose",          # limb-length / pictorial structures
        "DeepLabCut",       # baseline tool
    ]
    missing_priors = [p for p in expected_prior_art
                      if p not in text]
    assert not missing_priors, (
        f"Document should reference relevant published prior "
        f"art; missing: {missing_priors}"
    )

    # Decision points — the doc has to surface design choices,
    # not just describe one final answer. Either with explicit
    # recommendations (when undecided) or committed-answers
    # (after Gravio commits). Either way, the doc must show
    # decisions being explicitly discussed.
    decision_markers = (
        text.count("**Recommendation**")
        + text.count("✅ COMMITTED")
        + text.count("Committed:")
    )
    assert decision_markers >= 3, (
        f"Document should call out at least 3 design decisions "
        f"either as recommendations (pre-commit) or committed "
        f"answers (post-commit); found {decision_markers}"
    )

    # Open questions OR committed answers section — the doc must
    # explicitly track decisions, either pre- or post-commit
    has_open = ("Open design questions" in text
                or "open design" in text.lower())
    has_committed = ("Decisions committed" in text
                     or "committed" in text.lower())
    assert has_open or has_committed, (
        "Document should track design decisions explicitly, "
        "either as open questions (pre-commit) or as committed "
        "answers (post-commit)"
    )

    # Honesty section — what we're certain of vs. asserting
    assert (
        "What I'm certain" in text
        or "certain of" in text.lower()
        or "Risks and unknowns" in text
    ), (
        "Document should distinguish what's mathematically "
        "certain from what's being asserted on plausibility"
    )

    # Validation strategy mentioned — design has to address
    # how we'd know it works
    assert (
        "validation" in text.lower() or "validate" in text.lower()
    ), (
        "Document should address how the smoother would be "
        "validated against ground truth"
    )

    print("smoke_kalman_design_doc: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
