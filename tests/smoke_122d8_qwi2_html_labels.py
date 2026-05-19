"""
tests/smoke_122d8_qwi2_html_labels.py
=======================================

Patch 122d8: fix QWI-2 — `QRadioButton` destination label
rendered raw `<code>` / `<i>` HTML markup as literal text.

Coverage
--------
1.  features.py: the destination QRadioButton no longer contains
    HTML markup (`<code>`, `<i>`, `&lt;`, `&gt;`).
2.  The label still mentions both "family" and "video" (the
    placeholder names) so users still know the path structure.
3.  The "recommended, v1-native" hint is preserved.
4.  qt_workbench_known_issues.md marks QWI-2 Fixed 122d8.
5.  Other QLabel HTML uses elsewhere in features.py are
    intentionally preserved (QLabel renders HTML; only
    QRadioButton.text doesn't). Verify at least one QLabel
    still contains HTML — confirms we didn't over-strip.
6.  Parse-clean.
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
    pkg = REPO_ROOT / "mufasa"
    feat_path = pkg / "ui_qt" / "forms" / "features.py"
    feat_src = feat_path.read_text()

    # 1. The destination QRadioButton no longer has HTML
    # Find every QRadioButton(... "...", ...) and verify no HTML
    radio_pattern = re.compile(
        r"QRadioButton\(\s*((?:[\"'][^\"']*[\"']\s*)+),",
        re.DOTALL,
    )
    bad_radios = []
    for m in radio_pattern.finditer(feat_src):
        text_arg = m.group(1)
        if any(s in text_arg for s in ("<code>", "<i>", "<b>",
                                       "&lt;", "&gt;", "&amp;")):
            line_no = feat_src[:m.start()].count("\n") + 1
            bad_radios.append(f"L{line_no}")
    check(
        f"No QRadioButton in features.py contains HTML markup "
        f"(QWI-2 fix). got {len(bad_radios)} offenders: "
        f"{bad_radios}",
        not bad_radios,
    )

    # 2. Destination label still mentions {family} and {video}
    # as the placeholder convention
    check(
        "Destination radio still shows the path-structure "
        "placeholders ({family} + {video})",
        "{family}" in feat_src and "{video}" in feat_src,
    )

    # 3. "recommended, v1-native" hint preserved
    check(
        "'(recommended, v1-native)' hint preserved on the "
        "destination radio button",
        "(recommended, v1-native)" in feat_src,
    )

    # 4. Known-issues doc marks QWI-2 fixed
    qwi_doc = (REPO_ROOT / "docs"
               / "qt_workbench_known_issues.md").read_text()
    check(
        "qt_workbench_known_issues.md marks QWI-2 Fixed 122d8",
        "QWI-2" in qwi_doc
        and "Fixed 122d8" in qwi_doc,
    )

    # 5. QLabel HTML preserved (regression guard against over-stripping)
    label_pattern = re.compile(
        r"QLabel\(\s*((?:[\"'][^\"']*[\"']\s*)+)",
        re.DOTALL,
    )
    qlabel_with_html = 0
    for m in label_pattern.finditer(feat_src):
        text_arg = m.group(1)
        if any(s in text_arg for s in ("<b>", "<i>", "<code>")):
            qlabel_with_html += 1
    check(
        f"At least one QLabel in features.py still uses HTML "
        f"(QLabel renders rich text; over-stripping would have "
        f"been wrong; got {qlabel_with_html} QLabels with HTML)",
        qlabel_with_html >= 1,
    )

    # 6. Parse-clean
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
        f"smoke_122d8_qwi2_html_labels: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
