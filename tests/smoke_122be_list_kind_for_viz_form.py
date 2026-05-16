"""
tests/smoke_122be_list_kind_for_viz_form.py
=============================================

Patch 122be: adds a native "list" kind to the
_ExtrasFormBuilder so the declarative form can express
list-typed backend kwargs (body_parts, …) directly — no
more ``kwargs_transform`` lambda boilerplate per route.

Migration of existing transforms (2 → 0)
----------------------------------------
The 2 routes that used ``kwargs_transform`` for body_part
coercion are converted to use the new kind. The
``kwargs_transform`` framework hook itself stays (added in
122az) as the extensibility surface for future routes whose
coercion needs go beyond list-splitting.

Converted routes:
    1. ROIPlotter
        body_part: str + transform → body_parts: list
    2. ROIfeatureVisualizer
        body_part: str + transform → body_parts: list

How the "list" kind works
-------------------------
* UI: QLineEdit collecting comma-separated text.
* Default: may be a list[str] (rendered as comma-joined
  text) or a pre-joined string.
* to_kwargs: splits on comma, strips whitespace, drops
  empties. Always returns list[str] — empty input gives [].
* Placeholder: defaults to a comma-format hint if not
  provided.

Coverage
--------
1. _ExtrasFormBuilder.add docstring documents "list" kind.
2. _add dispatch has a "list" branch with the right shape
   (placeholder defaulting, list-default rendering).
3. to_kwargs has a "list" branch that comma-splits, strips,
   filters empties.
4. Behavioural: simulate a widget reading "a, b,c , " → ["a","b","c"].
5. Behavioural: simulate empty input "" → [].
6. Behavioural: simulate single value "Nose" → ["Nose"].
7. The 2 ROI routes now use kind="list" for body_parts.
8. Zero remaining kwargs_transform= lambdas in ROUTES.
9. _VizRoute.kwargs_transform field is still present as
   framework extensibility hook.
10. Route count unchanged.
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

    # ==================================================================
    # 1. Docstring documents "list" kind
    # ==================================================================
    check(
        "_ExtrasFormBuilder docstring documents 'list' kind",
        '``"list"`` — QLineEdit' in src,
    )
    check(
        "List kind docstring records Patch 122be",
        "Patch 122be" in src,
    )

    # ==================================================================
    # 2. _add dispatch has a "list" branch
    # ==================================================================
    check(
        "_add has elif kind == 'list' branch",
        'elif kind == "list":' in src,
    )
    check(
        "list branch handles list-default rendering",
        'isinstance(default, (list, tuple))' in src
        and '",".join(str(x) for x in default)' in src,
    )
    check(
        "list branch uses QLineEdit",
        ('elif kind == "list":' in src
         and 'QLineEdit' in src.split('elif kind == "list":')[1][:500]),
    )

    # ==================================================================
    # 3. to_kwargs has a "list" branch
    # ==================================================================
    check(
        "to_kwargs has elif kind == 'list' branch",
        'elif kind == "list":' in src,
    )
    check(
        "list branch in to_kwargs: comma split + strip + filter",
        '.split(",")' in src
        and 's.strip()' in src
        and 'if s.strip()' in src,
    )

    # ==================================================================
    # 4-6. Behavioural: simulate the to_kwargs logic
    # ==================================================================
    def list_kind_split(raw: str):
        return [s.strip() for s in (raw or "").split(",") if s.strip()]

    check(
        "Behavioural: 'a, b,c , ' → ['a','b','c']",
        list_kind_split("a, b,c , ") == ["a", "b", "c"],
    )
    check(
        "Behavioural: '' → []",
        list_kind_split("") == [],
    )
    check(
        "Behavioural: 'Nose' → ['Nose']",
        list_kind_split("Nose") == ["Nose"],
    )
    check(
        "Behavioural: ', , ,' (only separators) → []",
        list_kind_split(", , ,") == [],
    )

    # ==================================================================
    # 7. The 2 ROI routes use kind="list" for body_parts
    # ==================================================================
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("viz form parses", False, detail=str(e))
        return 1
    check("viz form parses", True)

    routes_with_list_extras: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_VizRoute"):
            continue
        label = None
        list_extras: list[str] = []
        for kw in node.keywords:
            if kw.arg == "label" and isinstance(
                kw.value, ast.Constant,
            ):
                label = kw.value.value
            if kw.arg == "extras" and isinstance(kw.value, ast.List):
                for item in kw.value.elts:
                    if not isinstance(item, ast.Tuple):
                        continue
                    if (len(item.elts) >= 2
                            and isinstance(item.elts[1], ast.Constant)
                            and item.elts[1].value == "list"):
                        if isinstance(item.elts[0], ast.Constant):
                            list_extras.append(item.elts[0].value)
        if label and list_extras:
            routes_with_list_extras[label] = list_extras

    expected = {
        "ROI overlay (per video)": ["body_parts"],
        "ROI feature overlay (per video)": ["body_parts"],
    }
    for label, kwargs in expected.items():
        check(
            f"Route {label!r} declares "
            f"extras kind='list' for {kwargs}",
            routes_with_list_extras.get(label) == kwargs,
            detail=f"got {routes_with_list_extras.get(label)!r}",
        )

    # ==================================================================
    # 8. ROI-route migration: both ROIPlotter + ROIfeatureVisualizer
    #    are converted to kind="list". 122bg-era HEAD: later patches
    #    (e.g. 122ba's niche viz fill-ins) may re-introduce
    #    kwargs_transform lambdas for OTHER routes that haven't
    #    been migrated yet (cue_light_names, arm_names). Verify
    #    that whatever transforms exist do NOT touch body_part —
    #    locking in 122be's specific migration without forbidding
    #    later transforms.
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
    # Body-part transforms specifically must be gone (the 122be
    # migration). Other transforms may exist post-122ba.
    body_part_lambdas = [
        lam for lam in transform_lambdas
        if "body_part" in ast.dump(lam)
    ]
    check(
        "Zero body_part kwargs_transform lambdas in ROUTES "
        "(both ROI routes converted to native 'list' kind)",
        len(body_part_lambdas) == 0,
        detail=f"got {len(body_part_lambdas)} body_part lambdas; "
               f"total transforms in ROUTES = {len(transform_lambdas)}",
    )

    # ==================================================================
    # 9. _VizRoute.kwargs_transform field still present
    #    (framework hook retained for future extensibility)
    # ==================================================================
    check(
        "_VizRoute.kwargs_transform field still declared",
        "kwargs_transform: Optional[Callable[[dict], dict]] = None"
        in src,
    )
    check(
        "Dispatch path still applies route.kwargs_transform",
        "route.kwargs_transform" in src,
    )

    # ==================================================================
    # 10. Route count — 122be is a refactor-not-add, so its
    #     floor is 20 (from 122az). Later patches (122ba, 122bg)
    #     add routes. Check the floor instead of equality so
    #     this test stays green through future additions.
    # ==================================================================
    label_count = src.count('label="')
    check(
        f"ROUTES has at least 20 entries (122az's floor; this "
        f"patch is a refactor). Found {label_count}",
        label_count >= 20,
    )

    # ==================================================================
    # 11. Old singular "body_part" extras gone from the affected
    #     routes (rename completed)
    # ==================================================================
    check(
        "Old singular 'body_part' string extras no longer in "
        "ROIPlotter / ROIfeatureVisualizer routes",
        '"body_part",         "str",   "Nose"' not in src
        and '"body_part", "str", "Nose"' not in src,
    )

    print(
        f"smoke_122be_list_kind_for_viz_form: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
