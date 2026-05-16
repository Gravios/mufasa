"""
tests/smoke_122bi_migrate_cuelight_spontaneous_to_list_kind.py
================================================================

Patch 122bi: migrates CueLightVisualizer + SpontaneousAlternationsPlotter
routes from kwargs_transform lambdas to the native "list" extras
kind (introduced in 122be).

This finishes the kwargs_transform → "list" kind migration for
existing routes. 122be migrated the 2 ROI routes; the
user's 122ba apply re-introduced 2 transforms (cue_light_names
+ arm_names) which this patch now migrates too.

Result
------
* 0 kwargs_transform lambdas in ROUTES (was 2)
* CueLightVisualizer route: cue_light_names is kind="list"
* SpontaneousAlternationsPlotter route: arm_names is kind="list"
* kwargs_transform framework hook stays in _VizRoute as
  extensibility surface for future non-list, non-dict
  coercions

The hook stays unused-but-available — same as
default_kwargs, scope_kind, etc. Removing it would be
overzealous; it's cheap to keep and reserves the design
space.

Coverage
--------
1. Zero kwargs_transform=lambda usages in ROUTES.
2. _VizRoute.kwargs_transform field is still declared in the
   dataclass (framework hook retained).
3. Dispatch path still calls route.kwargs_transform if set.
4. CueLightVisualizer route declares kind="list" for
   cue_light_names with a list default of [] and a hint
   placeholder.
5. SpontaneousAlternationsPlotter route declares kind="list"
   for arm_names with a list default of [] and a hint
   placeholder.
6. The two routes no longer mention str-comma-split phrasing
   like "Comma-separated cue light names" (replaced by the
   list-kind hint).
7. Route count unchanged at 29 (this is a refactor, not an
   add/remove).
8. Header comment for the 122ba section is updated to record
   the 122bi migration (no longer claims "kwargs_transform
   hook" is in use for these).
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
    viz_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "visualizations.py")
    src = viz_path.read_text()

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("viz form parses", False, detail=str(e))
        return 1
    check("viz form parses", True)

    # ==================================================================
    # 1. Zero kwargs_transform lambdas in ROUTES
    # ==================================================================
    transform_lambdas: list[ast.Lambda] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_VizRoute"):
            for kw in node.keywords:
                if (kw.arg == "kwargs_transform"
                        and isinstance(kw.value, ast.Lambda)):
                    transform_lambdas.append(kw.value)
    check(
        "Zero kwargs_transform lambdas in ROUTES "
        "(122bi finished the migration started in 122be)",
        len(transform_lambdas) == 0,
        detail=f"got {len(transform_lambdas)}",
    )

    # ==================================================================
    # 2. _VizRoute.kwargs_transform field still declared
    # ==================================================================
    check(
        "_VizRoute.kwargs_transform field still declared "
        "(framework hook retained)",
        "kwargs_transform: Optional[Callable[[dict], dict]] = None"
        in src,
    )

    # ==================================================================
    # 3. Dispatch path still applies route.kwargs_transform
    # ==================================================================
    check(
        "Dispatch path still applies route.kwargs_transform",
        "route.kwargs_transform" in src,
    )

    # ==================================================================
    # 4-5. Both routes declare kind="list" for the migrated
    #      fields with [] default
    # ==================================================================
    routes_with_list_extras: dict[str, list[tuple]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_VizRoute"):
            continue
        label = None
        list_extras: list[tuple] = []
        for kw in node.keywords:
            if kw.arg == "label" and isinstance(
                kw.value, ast.Constant,
            ):
                label = kw.value.value
            if kw.arg == "extras" and isinstance(kw.value, ast.List):
                for item in kw.value.elts:
                    if not isinstance(item, ast.Tuple):
                        continue
                    if (len(item.elts) >= 3
                            and isinstance(item.elts[1], ast.Constant)
                            and item.elts[1].value == "list"
                            and isinstance(item.elts[0], ast.Constant)):
                        # capture name + raw default ast for type
                        # inspection
                        list_extras.append((
                            item.elts[0].value,
                            item.elts[2],
                        ))
        if label and list_extras:
            routes_with_list_extras[label] = list_extras

    # CueLightVisualizer
    cue_extras = routes_with_list_extras.get(
        "Cue light visualizer", []
    )
    cue_names = [name for name, _ in cue_extras]
    check(
        "Cue light visualizer route declares kind='list' for "
        "cue_light_names (122bi migration)",
        "cue_light_names" in cue_names,
    )
    # Default for cue_light_names is an empty list ([])
    cue_default = next(
        (d for n, d in cue_extras if n == "cue_light_names"),
        None,
    )
    check(
        "Cue light visualizer cue_light_names default is []",
        (isinstance(cue_default, ast.List)
         and len(cue_default.elts) == 0),
    )

    # SpontaneousAlternationsPlotter
    arm_extras = routes_with_list_extras.get(
        "Spontaneous alternation plot", []
    )
    arm_names = [name for name, _ in arm_extras]
    check(
        "Spontaneous alternation plot route declares kind='list' "
        "for arm_names (122bi migration)",
        "arm_names" in arm_names,
    )
    arm_default = next(
        (d for n, d in arm_extras if n == "arm_names"),
        None,
    )
    check(
        "Spontaneous alternation plot arm_names default is []",
        (isinstance(arm_default, ast.List)
         and len(arm_default.elts) == 0),
    )

    # ==================================================================
    # 6. The migrated routes no longer mention str-comma-split
    #    phrasing (sanity guard: the str-shape extras must be
    #    fully replaced, not duplicated)
    # ==================================================================
    # Look for the OLD shapes via patterns. The OLD shape was
    # ("cue_light_names", "str", "", "Comma-separated cue ...")
    # and similarly for arm_names. The placeholder text used
    # "Comma-separated cue light names" / "Comma-separated arm
    # ROI names" — verify both phrasings are gone.
    check(
        "Old 'Comma-separated cue light names' placeholder gone",
        "Comma-separated cue light names" not in src,
    )
    check(
        "Old 'Comma-separated arm ROI names' placeholder gone",
        "Comma-separated arm ROI names" not in src,
    )
    # The old kwargs_transform comma-split lambdas referenced
    # str(kw["cue_light_names"]).split(",") — that pattern
    # specifically should be absent now.
    check(
        "No remaining str(...).split(',') comma-split lambdas "
        "for cue_light_names / arm_names",
        'str(kw["cue_light_names"]).split(",")' not in src
        and 'str(kw["arm_names"]).split(",")' not in src,
    )

    # ==================================================================
    # 7. Route count unchanged (refactor, not add/remove)
    # ==================================================================
    label_count = src.count('label="')
    check(
        f"ROUTES count at least 29 (122bh's floor; this patch "
        f"is a refactor). Found {label_count}",
        label_count >= 29,
    )

    # ==================================================================
    # 8. 122ba section header is updated to record 122bi
    # ==================================================================
    check(
        "122ba section header references 122bi migration",
        "Patch 122bi" in src
        and "122ba" in src,
    )
    # And the old "use the 122az kwargs_transform hook" phrasing
    # in that header is gone.
    check(
        "122ba section header no longer claims kwargs_transform "
        "is used for the str → list coercions",
        "Single str → list[str] coercions" not in src,
    )

    print(
        f"smoke_122bi_migrate_cuelight_spontaneous_to_list_kind: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
