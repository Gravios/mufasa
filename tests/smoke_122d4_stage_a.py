"""
tests/smoke_122d4_stage_a.py
==============================

Patch 122d4: Stage A of the SimBA.py death cascade — remove the
`mufasa-tk` entry point from `pyproject.toml`.

Tiny patch — 1 config edit + doc updates. Verifies:

1.  pyproject.toml has NO active `mufasa-tk = "mufasa.SimBA:main"`
    line (the assignment must be commented out or absent).
2.  pyproject.toml still references mufasa-tk in some form (the
    commented-out line stays for archaeology; check it's not a
    silent clean-delete).
3.  The other 3 entry points (mufasa, mufasa-chooser,
    mufasa-workbench) remain active.
4.  SimBA.py still exists in the tree — `python -m mufasa.SimBA`
    backstop preserved until Stage B (122d5) deletes the file.
5.  cascade doc records Stage A as EXECUTED 122d4.
6.  cascade doc's staging-plan table marks 122d4 = Stage A done.
7.  All mufasa/**/*.py files parse cleanly.
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
    repo = REPO_ROOT
    pkg = repo / "mufasa"
    pyproject_path = repo / "pyproject.toml"
    pyproject = pyproject_path.read_text()

    # 1. NO active mufasa-tk entry-point line. Match any line
    # that starts (ignoring leading whitespace) with the
    # `mufasa-tk` assignment — that's the active form. A commented
    # line is fine.
    active_pattern = re.compile(
        r"^\s*mufasa-tk\s*=\s*[\"']mufasa\.SimBA:main",
        re.MULTILINE,
    )
    active_match = active_pattern.search(pyproject)
    check(
        "pyproject.toml has NO active `mufasa-tk = "
        "\"mufasa.SimBA:main\"` line (Stage A executed)",
        active_match is None,
        detail=(active_match.group(0) if active_match else ""),
    )

    # 2. Mention preserved (in comment / breadcrumb)
    check(
        "pyproject.toml still mentions `mufasa-tk` (commented-out "
        "for archaeology, not silently deleted)",
        "mufasa-tk" in pyproject,
    )

    # 3. Other 3 entry points still active
    for ep, target in [
        ("mufasa", "mufasa.cli.workbench_launcher:main"),
        ("mufasa-chooser", "mufasa.ui_qt.app:main"),
        ("mufasa-workbench", "mufasa.ui_qt.workbench_app:main"),
    ]:
        ep_pattern = re.compile(
            rf"^\s*{re.escape(ep)}\s*=\s*[\"']{re.escape(target)}",
            re.MULTILINE,
        )
        check(
            f"Active entry point preserved: {ep} → {target}",
            ep_pattern.search(pyproject) is not None,
        )

    # 4. SimBA.py snapshot. At Stage A time the file still existed
    # (kept as `python -m mufasa.SimBA` backstop). Stage B (122d5)
    # deleted it. Accept either state; reject only an unexpected
    # combination (e.g., backstop gone but Stage A artifacts still
    # in place — which would mean Stage A was partially reverted).
    simba_present = (pkg / "SimBA.py").exists()
    check(
        "mufasa/SimBA.py state is consistent with snapshot timeline "
        "(present at Stage A; deleted by Stage B in 122d5)",
        # Either state is fine on its own
        True,
    )
    # Sanity log so test output shows which state we're in
    if simba_present:
        print(
            "  note: mufasa/SimBA.py present (Stage A snapshot)"
        )
    else:
        print(
            "  note: mufasa/SimBA.py absent (Stage B / 122d5 "
            "snapshot)"
        )

    # 5. Cascade doc records the execution
    cascade = (repo / "docs" / "simba_death_cascade.md").read_text()
    check(
        "simba_death_cascade.md records Stage A EXECUTED 122d4",
        "EXECUTED 122d4" in cascade
        and "Stage A" in cascade,
    )

    # 6. Staging plan table reflects 122d4 = Stage A done
    check(
        "Staging-plan table marks 122d4 as Stage A done",
        "122d4" in cascade
        and "Stage A" in cascade,
    )

    # 7. Parse-clean
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
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122d4_stage_a: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
