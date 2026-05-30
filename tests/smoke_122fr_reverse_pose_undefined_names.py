"""
tests/smoke_122fr_reverse_pose_undefined_names.py
=================================================

Patch 122fr — resolve the recoverable undefined-name crashes in
pose_processors/reverse_pose.py surfaced by the 122fp audit.

User request: "continue" (autonomous backlog selection — next was the
deferred reverse_pose functional bug from the 122fp audit).

CONTEXT / WHY
=============
The 122fp audit found three undefined names in reverse_pose.py that
NameError the moment their branch runs. 122fp deferred all three
("need the deleted defs or a domain decision"). On closer tracing two
are recoverable from in-repo evidence, one is not:

  * "9" branch -> extract_features_wotarget_9   (UNDEFINED)
        The canonical pose->extractor table (mufasa.utils.lookups) maps
        "9": ExtractFeaturesFrom9bps, and reverse_pose ALREADY imports
        ExtractFeaturesFrom9bps but never used it (dead import). Both
        facts pin the fix: call ExtractFeaturesFrom9bps(config_path),
        matching every sibling branch ("16"->From16bps, etc.). This is
        evidence-based, not a guess.

  * check_that_two_dfs_are_equal_len  (UNDEFINED)
        No equivalent exists in mufasa.utils.checks. The name + call
        site fully pin the contract (two equal-length series before a
        per-column assignment in reappend_targets), so the helper was
        RESTORED in checks.py following the established convention
        (raise_error param, Union[None, bool] return, CountError with
        source=<fn>.__name__) and imported in reverse_pose.

  * "987" branch -> extract_features_wotarget_14_from_16  (UNDEFINED)
        LEFT AS-IS. "987" is not a key in the canonical lookups table
        and no in-repo extractor matches; the intended semantics are
        unknown. Restoring it would be a behavioural guess. An in-code
        NOTE marks it deliberate; it still NameErrors by design if
        reached.

WHAT THIS PATCH LANDED
======================
mufasa/utils/checks.py
* new check_that_two_dfs_are_equal_len(df_1, df_2, file_path_1,
  file_path_2, col_name=None, raise_error=True) -> Union[None, bool].

mufasa/pose_processors/reverse_pose.py
* import check_that_two_dfs_are_equal_len from mufasa.utils.checks.
* "9" branch -> ExtractFeaturesFrom9bps(self.config_path) (import now
  live, not dead).
* "987" branch annotated with a deliberate-non-fix NOTE.

NEW SMOKE: smoke_122fr_reverse_pose_undefined_names.py (8 checks)
"""

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


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _func_node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def main() -> int:
    checks_src = _read("mufasa/utils/checks.py")
    rev_src = _read("mufasa/pose_processors/reverse_pose.py")

    # --- restored helper exists with the call-site signature -------------
    fn = _func_node(checks_src, "check_that_two_dfs_are_equal_len")
    check("check_that_two_dfs_are_equal_len is defined in checks.py", fn is not None)
    if fn is not None:
        arg_names = [a.arg for a in fn.args.args]
        check(
            "helper signature matches the reverse_pose call site",
            arg_names == ["df_1", "df_2", "file_path_1", "file_path_2", "col_name", "raise_error"],
            detail=str(arg_names),
        )
        body_src = ast.unparse(fn)
        check(
            "helper raises CountError via raise_error convention",
            "CountError" in body_src and "raise_error" in arg_names,
        )
    else:
        check("helper signature matches the reverse_pose call site", False)
        check("helper raises CountError via raise_error convention", False)

    # --- reverse_pose imports the helper ---------------------------------
    imported = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "mufasa.utils.checks"
        and any(a.name == "check_that_two_dfs_are_equal_len" for a in n.names)
        for n in ast.walk(ast.parse(rev_src))
    )
    check("reverse_pose imports check_that_two_dfs_are_equal_len", imported)

    # --- "9" branch now calls ExtractFeaturesFrom9bps --------------------
    cf = _func_node(rev_src, "create_features")
    cf_src = ast.unparse(cf) if cf else ""
    check(
        "create_features '9' branch calls ExtractFeaturesFrom9bps",
        "ExtractFeaturesFrom9bps(self.config_path)" in cf_src
        and "extract_features_wotarget_9" not in rev_src,
    )

    # --- "987" branch deliberately left (documented) ---------------------
    check(
        "'987' branch left unresolved with a documented NOTE",
        "extract_features_wotarget_14_from_16" in rev_src
        and "NOTE (122fr)" in rev_src,
    )

    # --- the 9bp import is no longer dead --------------------------------
    check(
        "ExtractFeaturesFrom9bps import is now used",
        rev_src.count("ExtractFeaturesFrom9bps") >= 2,  # import + call site
    )

    # --- everything still parses -----------------------------------------
    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"all mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fr_reverse_pose_undefined_names: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
