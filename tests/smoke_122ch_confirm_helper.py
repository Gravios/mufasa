"""
tests/smoke_122ch_confirm_helper.py
=====================================

Patch 122ch: introduces `mufasa/utils/confirm.py` and uses it to
decouple `video_processors/video_processing.py` and
`mixins/train_model_mixin.py` from the Tk surface.

These two files were the two highest-leverage entries in
`backend_audit.md` §4d — they're fundamental backend code (the
video-processing module is imported by ~25 other backend modules;
the train-model mixin is the substrate for every Sklearn /
Tensorflow training pipeline). Removing Tk from them removes a
viral coupling.

The new helper:

* `confirm_two_option(question, option_one, option_two, title)`
  — UI-agnostic confirmation. Returns one of the two option
  strings.
* Default implementation lazy-imports Tk; falls back to stdin if
  Tk unavailable; falls back to `option_one` if stdin
  unavailable.
* Qt code can reassign `mufasa.utils.confirm.confirm_two_option`
  at workbench startup to use `QMessageBox.question`.
* Tests can reassign to `lambda **_: "YES"` for auto-confirm.

Coverage
--------
1. mufasa/utils/confirm.py exists.
2. confirm_two_option is exported (module-level attribute).
3. _default_confirm is defined.
4. _stdin_confirm fallback is defined.
5. confirm_two_option default points to _default_confirm.
6. confirm.py's Tk import is lazy (inside function body, not at
   module load).
7. video_processing.py no longer imports TwoOptionQuestionPopUp.
8. video_processing.py no longer imports from
   mufasa.ui.tkinter_functions at module load.
9. video_processing.py imports confirm_two_option.
10. video_processing.py extract_frames branch calls
    confirm_two_option (not TwoOptionQuestionPopUp(...).selected_option).
11. train_model_mixin.py no longer imports TwoOptionQuestionPopUp.
12. train_model_mixin.py no longer imports from
    mufasa.ui.tkinter_functions at module load.
13. train_model_mixin.py imports confirm_two_option.
14. train_model_mixin.py meta-config branch calls
    confirm_two_option.
15. Backend module-level Tk-importer count is ≤ 23 (regression
    guard — was 25 pre-122ch).
16. backend_audit.md §4d items 11 + 12 marked DONE in 122ch.
17. All mufasa/**/*.py files parse cleanly.

Behavioural verification: confirm.py CAN be imported and the
default function CAN be called (with a monkey-patched stdin) in
the sandbox — Tk isn't available so the stdin fallback fires.
This is the only patch in recent memory where a behavioral test
is actually feasible alongside the AST audit.
"""
from __future__ import annotations

import ast
import io
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


def _module_level_imports(tree: ast.Module, target: str) -> bool:
    """True iff `target` is imported at module level."""
    return any(
        isinstance(n, ast.ImportFrom) and n.module == target
        for n in tree.body
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # mufasa/utils/confirm.py
    # ==================================================================
    confirm_path = pkg / "utils" / "confirm.py"
    check("mufasa/utils/confirm.py exists", confirm_path.exists())
    if not confirm_path.exists():
        return 1
    confirm_src = confirm_path.read_text()
    confirm_tree = ast.parse(confirm_src)

    # _default_confirm, _stdin_confirm, confirm_two_option
    funcs = {n.name for n in confirm_tree.body
             if isinstance(n, ast.FunctionDef)}
    check(
        "_default_confirm function defined",
        "_default_confirm" in funcs,
    )
    check(
        "_stdin_confirm fallback function defined",
        "_stdin_confirm" in funcs,
    )

    # confirm_two_option module-level assignment
    has_binding = False
    for n in confirm_tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "confirm_two_option":
                    has_binding = True
    check(
        "confirm_two_option module-level binding present",
        has_binding,
    )
    check(
        "confirm_two_option default points to _default_confirm",
        "confirm_two_option = _default_confirm" in confirm_src,
    )

    # Tk import is lazy (inside _default_confirm body, not at module
    # level)
    has_top_tk = _module_level_imports(
        confirm_tree, "mufasa.ui.tkinter_functions")
    check(
        "confirm.py does NOT import Tk at module level "
        "(import is lazy inside _default_confirm)",
        not has_top_tk,
    )
    check(
        "confirm.py has a deferred Tk import somewhere in the file "
        "(in _default_confirm body)",
        "from mufasa.ui.tkinter_functions" in confirm_src
        and "TwoOptionQuestionPopUp" in confirm_src,
    )

    # ==================================================================
    # video_processing.py decoupled
    # ==================================================================
    vp_src = (pkg / "video_processors" /
              "video_processing.py").read_text()
    vp_tree = ast.parse(vp_src)
    check(
        "video_processing.py no longer references "
        "TwoOptionQuestionPopUp as an importable name",
        "from mufasa.ui.tkinter_functions import TwoOptionQuestionPopUp"
        not in vp_src,
    )
    check(
        "video_processing.py does NOT import Tk at module level",
        not _module_level_imports(
            vp_tree, "mufasa.ui.tkinter_functions"),
    )
    check(
        "video_processing.py imports confirm_two_option from "
        "mufasa.utils.confirm",
        "from mufasa.utils.confirm import confirm_two_option" in vp_src,
    )
    # The new call site
    check(
        "video_processing.py uses confirm_two_option at the "
        "EXTRACT ALL FRAMES branch",
        "confirm_two_option(" in vp_src
        and "EXTRACT ALL FRAMES" in vp_src,
    )

    # ==================================================================
    # train_model_mixin.py decoupled
    # ==================================================================
    tm_src = (pkg / "mixins" / "train_model_mixin.py").read_text()
    tm_tree = ast.parse(tm_src)
    check(
        "train_model_mixin.py no longer imports TwoOptionQuestionPopUp",
        "from mufasa.ui.tkinter_functions import TwoOptionQuestionPopUp"
        not in tm_src,
    )
    check(
        "train_model_mixin.py does NOT import Tk at module level",
        not _module_level_imports(
            tm_tree, "mufasa.ui.tkinter_functions"),
    )
    check(
        "train_model_mixin.py imports confirm_two_option",
        "from mufasa.utils.confirm import confirm_two_option" in tm_src,
    )
    check(
        "train_model_mixin.py meta-config branch calls confirm_two_option",
        "confirm_two_option(" in tm_src
        and "META CONFIG FILE ERROR" in tm_src,
    )

    # ==================================================================
    # Backend module-level Tk-importer count regression guard
    # ==================================================================
    module_level_count = 0
    for f in pkg.rglob("*.py"):
        if any(p in f.parts for p in ("ui", "ui_qt")):
            continue
        if f.name == "SimBA.py":
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        if _module_level_imports(t, "mufasa.ui.tkinter_functions"):
            module_level_count += 1
    check(
        f"Backend module-level Tk-importer count is ≤ 23 "
        f"(got {module_level_count}; was 25 pre-122ch)",
        module_level_count <= 23,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4d marks both decouplings DONE in 122ch",
        audit.count("DONE in patch 122ch") >= 2,
    )

    # ==================================================================
    # Behavioural smoke: confirm.py CAN be imported and called
    # ==================================================================
    # In the sandbox Tk isn't available, so the default callback
    # falls through to the stdin path. Monkey-patch stdin to a
    # known response, verify the helper returns the expected option.
    try:
        # Force import + behavioral test.
        import mufasa.utils.confirm as _cf  # noqa: E402

        # Replace input() temporarily
        import builtins as _b
        orig_input = _b.input
        _b.input = lambda *_a, **_kw: "2"  # user picks option_two
        try:
            choice = _cf.confirm_two_option(
                question="test", option_one="A", option_two="B",
                title="t")
        finally:
            _b.input = orig_input
        check(
            "behavioral: confirm_two_option returns chosen option "
            "string via stdin fallback",
            choice == "B",
            detail=f"got {choice!r}",
        )
    except Exception as exc:
        check(
            "behavioral: confirm.py importable + callable",
            False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    # ==================================================================
    # All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122ch_confirm_helper: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
