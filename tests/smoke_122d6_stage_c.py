"""
tests/smoke_122d6_stage_c.py
==============================

Patch 122d6: Stage C of the SimBA.py death cascade — tail
deletion of the last Tk-coupled files. 3 files: the 2 scoped in
122cx (tkinter_functions, pop_up_mixin) plus 1 newly-exposed
orphan (mixins/unsupervised_mixin).

Coverage
--------

Stage C deletions:
1.  mufasa/ui/tkinter_functions.py gone.
2.  mufasa/mixins/pop_up_mixin.py gone.
3.  mufasa/mixins/unsupervised_mixin.py gone (newly discovered
    in 122d6 — orphan after Stage B cleared the unsupervised/
    cluster but mixins/unsupervised_mixin wasn't swept by Stage B
    because it lives in mixins/, not unsupervised/).

Survivors confirmed:
4.  mufasa/ui/px_to_mm_ui.py still exists (last Qt-uses-Tk
    dependency; Tier-4 follow-on).
5.  mufasa/ui/__init__.py still exists (package marker; ui/ now
    contains just px_to_mm_ui.py + __init__).
6.  utils/confirm.py still exists — the lazy import is now
    intentional headless-fallback (the try/except ImportError
    routes to stdin).
7.  mufasa/mixins/network_mixin.py still exists — pre-existing
    orphan, NOT exposed by the cascade. Disposition deferred.

Integrity:
8.  No surviving file imports `mufasa.mixins.pop_up_mixin`.
9.  No surviving file imports `mufasa.mixins.unsupervised_mixin`.
10. The only remaining importer of mufasa.ui.tkinter_functions
    is utils/confirm.py, AND it's a deferred (function-scope)
    import wrapped in try/except (graceful headless fallback).
11. All mufasa/**/*.py parse cleanly.
12. The complete `mufasa/ui/` directory now contains only the
    px_to_mm_ui.py survivor + __init__.

Doc updates:
13. cascade doc records Stage C as EXECUTED 122d6.
14. cascade doc notes the unsupervised_mixin discovery and the
    network_mixin pre-existing-orphan disposition.
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

    # 1-3. Stage C deletions
    check(
        "mufasa/ui/tkinter_functions.py is gone",
        not (pkg / "ui" / "tkinter_functions.py").exists(),
    )
    check(
        "mufasa/mixins/pop_up_mixin.py is gone",
        not (pkg / "mixins" / "pop_up_mixin.py").exists(),
    )
    check(
        "mufasa/mixins/unsupervised_mixin.py is gone "
        "(newly discovered orphan in 122d6)",
        not (pkg / "mixins" / "unsupervised_mixin.py").exists(),
    )

    # 4-7. Survivors
    # Patch 122de note: ui/px_to_mm_ui.py + ui/__init__.py were
    # later deleted (122de ported the cv2-based calibration UI to
    # the existing Qt PixelCalibrationDialog and purged the empty
    # mufasa/ui/ directory). Relax these pins to accept either the
    # post-122d6 state (files present) OR the post-122de state
    # (files gone). The intent of the check — "Stage C cascade
    # didn't accidentally remove px_to_mm_ui.py" — is preserved as
    # long as either state holds; only an unexpected partial state
    # (e.g., __init__.py gone but px_to_mm_ui.py present) would
    # signal a bug.
    px_present = (pkg / "ui" / "px_to_mm_ui.py").exists()
    init_present = (pkg / "ui" / "__init__.py").exists()
    check(
        "ui/px_to_mm_ui.py state consistent (either Stage C "
        "survivor or 122de cleanup; not a partial state)",
        px_present == init_present,
    )
    check(
        "utils/confirm.py still exists",
        (pkg / "utils" / "confirm.py").exists(),
    )
    check(
        "mufasa/mixins/network_mixin.py still exists "
        "(pre-existing orphan; NOT exposed by the cascade; "
        "separate disposition)",
        (pkg / "mixins" / "network_mixin.py").exists(),
    )

    # 8. No surviving file imports pop_up_mixin
    pum_hits = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.mixins.pop_up_mixin":
                    pum_hits.append(
                        f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No surviving file imports mufasa.mixins.pop_up_mixin",
        not pum_hits,
        detail=", ".join(pum_hits[:3]),
    )

    # 9. No surviving file imports unsupervised_mixin
    um_hits = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.mixins.unsupervised_mixin":
                    um_hits.append(
                        f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No surviving file imports mufasa.mixins.unsupervised_mixin",
        not um_hits,
        detail=", ".join(um_hits[:3]),
    )

    # 10. tkinter_functions importers state — at Stage C time
    # utils/confirm.py had a deferred (function-scope) lazy import
    # wrapped in try/except. Patch 122de removed that lazy import
    # entirely (the `mufasa.ui` package was deleted; the lazy
    # import could no longer succeed even if attempted). Accept
    # either state.
    tf_hits = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.ui.tkinter_functions":
                    tf_hits.append(
                        (str(f.relative_to(pkg)), node.lineno,
                         node.col_offset))
    if tf_hits:
        # Stage C snapshot: must be exactly utils/confirm.py
        check(
            f"All remaining tkinter_functions importers are "
            f"utils/confirm.py (got {len(tf_hits)}: {tf_hits})",
            len(tf_hits) == 1
            and "utils/confirm.py" in tf_hits[0][0].replace(
                "\\", "/"),
        )
        confirm_hit = tf_hits[0]
        check(
            "utils/confirm.py's import is deferred (function-"
            f"scope) — col_offset > 0; got {confirm_hit}",
            confirm_hit[2] > 0,
        )
    else:
        # Post-122de snapshot: no importers remain at all
        check(
            "No surviving tkinter_functions importers (122de "
            "cleanup removed the last lazy fallback)",
            True,
        )
        # Skip the col_offset check entirely
        check(
            "deferred-import check skipped (no importers to "
            "verify post-122de)",
            True,
        )

    # Bonus: verify confirm.py either still wraps the import in
    # try/except (Stage C state) OR has dropped it entirely
    # (post-122de state). Either is acceptable. Use AST to look
    # for an actual ImportFrom node, NOT a substring match —
    # 122de left a "tkinter_functions" mention in the docstring as
    # historical breadcrumb, which would false-positive a string
    # match.
    if (pkg / "utils" / "confirm.py").exists():
        cf_src = (pkg / "utils" / "confirm.py").read_text()
        cf_tree = ast.parse(cf_src)
        has_tf_import = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "mufasa.ui.tkinter_functions"
            for n in ast.walk(cf_tree)
        )
        if has_tf_import:
            # Stage C state — must still be wrapped in try/except
            check(
                "utils/confirm.py's tkinter_functions import is "
                "wrapped in try/except (Stage C state)",
                "except ImportError" in cf_src,
            )
        else:
            # Post-122de state — import gone entirely. Docstring
            # may still mention it as historical context.
            check(
                "utils/confirm.py has no tkinter_functions import "
                "(122de cleanup; docstring breadcrumb is fine)",
                True,
            )

    # 11. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All surviving mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 12. ui/ directory contents (post-122d6) OR ui/ gone entirely
    # (post-122de). Patch 122de deleted the directory after
    # porting px_to_mm_ui.py's caller to the Qt dialog. Both
    # states are acceptable; reject only unexpected partial states.
    if (pkg / "ui").exists():
        ui_files = sorted([
            f.name for f in (pkg / "ui").iterdir()
            if f.is_file() and f.name.endswith(".py")
        ])
        check(
            f"mufasa/ui/ exists and contains exactly "
            f"__init__.py + px_to_mm_ui.py (got: {ui_files})",
            ui_files == ["__init__.py", "px_to_mm_ui.py"],
        )
    else:
        check(
            "mufasa/ui/ directory deleted entirely (122de cleanup)",
            True,
        )

    # 13. Doc updates
    cascade = (REPO_ROOT / "docs"
               / "simba_death_cascade.md").read_text()
    check(
        "simba_death_cascade.md records Stage C EXECUTED 122d6",
        "EXECUTED 122d6" in cascade
        and "Stage C" in cascade,
    )

    # 14. Notes the unsupervised_mixin + network_mixin findings
    check(
        "cascade doc notes unsupervised_mixin newly-discovered + "
        "network_mixin pre-existing-orphan disposition",
        "unsupervised_mixin" in cascade
        and "network_mixin" in cascade,
    )

    print(
        f"smoke_122d6_stage_c: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
