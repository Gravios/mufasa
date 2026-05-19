"""
tests/smoke_122dc_visualizations_v1_routing.py
================================================

Patch 122dc: fix the last deferred hardwired-paths bug —
`ui_qt/forms/visualizations.py` per-route source-dir resolution
hardcoded `<root>/csv/<subdir>` and failed on v1 projects.

The fix adds a module-level `_resolve_viz_source_dir` helper +
`_VIZ_SOURCE_V1_MAP` dict that maps each route's `data_*_source`
value to the right v1 directory under `derived/` (with latest-
run-or-parent semantics matching `ConfigReader._apply_v1_path_overrides`).

Coverage
--------
1.  visualizations.py defines `_VIZ_SOURCE_V1_MAP` dict at module
    scope.
2.  visualizations.py defines `_resolve_viz_source_dir` helper at
    module scope.
3.  Helper signature accepts `config_path`, `project_root`,
    `source_name`.
4.  Helper uses `is_run_id` to detect run-id subdirs.
5.  All actually-used route source names in this file are covered
    by `_VIZ_SOURCE_V1_MAP` (no fallback path needed for the
    routes that ship today).
6.  `target()` no longer hardcodes `_P(proj) / "csv" / subdir` —
    uses the helper instead.
7.  v1 map covers the documented stages: machine_results,
    outlier_corrected_movement_location, features_extracted,
    directing_data.
8.  Each v1 mapping points under `sources/` or `derived/` —
    never under legacy `csv/`.
9.  Helper's legacy branch produces `<root>/csv/<source_name>/`
    (preserves pre-122dc behaviour for legacy projects).
10. Audit doc marks visualizations.py:1233 as fixed in 122dc.
11. Parse-clean.
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
    viz_path = pkg / "ui_qt" / "forms" / "visualizations.py"
    viz_src = viz_path.read_text()
    viz_tree = ast.parse(viz_src)

    # Module-level names
    module_names = {
        n.target.id if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        else (n.targets[0].id
              if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              else None)
        for n in viz_tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
    }
    module_names.discard(None)
    func_names = {n.name for n in viz_tree.body
                  if isinstance(n, ast.FunctionDef)}

    # 1. _VIZ_SOURCE_V1_MAP is module-level
    check(
        "`_VIZ_SOURCE_V1_MAP` dict defined at module scope",
        "_VIZ_SOURCE_V1_MAP" in module_names,
    )

    # 2-3. _resolve_viz_source_dir helper exists with right signature
    helper = next(
        (n for n in viz_tree.body
         if isinstance(n, ast.FunctionDef)
         and n.name == "_resolve_viz_source_dir"),
        None,
    )
    check("`_resolve_viz_source_dir` helper defined", helper is not None)
    if helper is not None:
        all_args = {
            *(a.arg for a in helper.args.args),
            *(a.arg for a in helper.args.kwonlyargs),
        }
        check(
            "Helper signature accepts config_path, project_root, "
            "source_name",
            {"config_path", "project_root", "source_name"} <= all_args,
        )

    # 4. Helper uses is_run_id
    helper_src = ast.unparse(helper) if helper else ""
    check(
        "Helper uses `is_run_id` to detect run-id subdirs "
        "(matches ConfigReader semantics)",
        "is_run_id" in helper_src,
    )

    # 5. All used route source names are in the v1 map
    sources_used = set(
        re.findall(r"data_path[s]?_source=['\"]([^'\"]+)['\"]",
                   viz_src)
    )
    map_match = re.search(
        r"_VIZ_SOURCE_V1_MAP[^}]+\}", viz_src, re.DOTALL,
    )
    map_keys = set(re.findall(
        r'["\'](\w+)["\']\s*:', map_match.group(0)
    )) if map_match else set()
    uncovered = sources_used - map_keys
    check(
        f"All {len(sources_used)} actually-used route sources "
        f"are in the v1 map (no uncovered fallback path needed). "
        f"Uncovered: {sorted(uncovered)}",
        not uncovered,
    )

    # 6. target() no longer hardcodes `_P(proj) / "csv" / subdir`
    target_body = ""
    for node in ast.walk(viz_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "target":
            # The one inside VisualizationForm; multiple `target`s
            # may exist if there are other classes — take the first
            # that contains data_paths_source logic
            body = ast.unparse(node)
            if "data_paths_source" in body:
                target_body = body
                break
    check(
        "`target` method no longer hardcodes "
        "`_P(proj) / \"csv\" / subdir` for src_dir",
        not re.search(
            r"_P\(proj\)\s*/\s*['\"]csv['\"]\s*/\s*subdir",
            target_body,
        ),
    )
    check(
        "`target` method calls `_resolve_viz_source_dir`",
        "_resolve_viz_source_dir" in target_body,
    )

    # 7. v1 map covers the 4 documented stages
    for stage in [
        "machine_results",
        "outlier_corrected_movement_location",
        "features_extracted",
        "directing_data",
    ]:
        check(
            f"v1 map covers stage `{stage}`",
            stage in map_keys,
        )

    # 8. Each v1 mapping value points under sources/ or derived/
    # (never under legacy csv/)
    map_body_match = re.search(
        r"_VIZ_SOURCE_V1_MAP[^=]*=\s*\{(.*?)\}",
        viz_src, re.DOTALL,
    )
    if map_body_match:
        body = map_body_match.group(1)
        check(
            "No v1 mapping value points under legacy `csv/`",
            "'csv'" not in body and '"csv"' not in body,
        )

    # 9. Legacy branch produces <root>/csv/<source_name>/
    check(
        "Helper's legacy branch returns "
        "`project_root / \"csv\" / source_name`",
        "/ 'csv' / source_name" in helper_src
        or '/ "csv" / source_name' in helper_src,
    )

    # 10. Audit doc marks visualizations as fixed
    audit_path = REPO_ROOT / "docs" / "hardwired_paths_audit.md"
    audit_src = audit_path.read_text()
    check(
        "Audit doc records visualizations.py:1233 as fixed in 122dc",
        "visualizations.py" in audit_src
        and "Fixed 122dc" in audit_src,
    )

    # 11. Parse-clean
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
        f"smoke_122dc_visualizations_v1_routing: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
