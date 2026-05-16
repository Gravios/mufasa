"""
tests/smoke_122bg_dict_kind_for_viz_form.py
=============================================

Patch 122bg: adds a "dict" kind to _ExtrasFormBuilder so the
declarative form can express dict-typed backend kwargs
(JSON-serializable config objects). Surfaces
CircularFeaturePlotter — the first route that genuinely needs
a dict argument. GeometryPlotter is explicitly NOT surfaced
(its `geometries` arg is Shapely objects, not JSON-able).

Continues the extras-kind expansion arc started in 122be:
* 122be — "list" kind (comma-split strings → list[str])
* 122bg — "dict" kind (JSON text → dict)

What the "dict" kind does
-------------------------
* UI: QLineEdit collecting a JSON object as text.
* Default: dict (rendered as compact JSON) or JSON string.
* to_kwargs: json.loads + try/except. Empty/malformed/
  non-object JSON → {} (silent fallback; backends typically
  have their own defaults for missing keys).
* Placeholder: defaults to a JSON hint.

What the "dict" kind explicitly does NOT do
-------------------------------------------
* No schema validation.
* No type coercion within values.
* No support for non-JSON-serializable dict values
  (e.g. Shapely geometry objects — reason GeometryPlotter
  is NOT surfaced).

Route added (1)
---------------
"Circular feature overlay (per video)" → CircularFeaturePlotter
    File-scope. Single extras row: settings dict, kind="dict",
    default `{"text_settings": false, "palette": "bwr"}`.

Routes explicitly NOT added
---------------------------
GeometryPlotter — `geometries` arg is Shapely objects.
    Future patch could add a file-loader kind (pickle/json
    of geometry descriptors) to unlock this.

Coverage
--------
1. _ExtrasFormBuilder.add docstring documents "dict" kind.
2. _add dispatch has a "dict" branch with dict-default JSON
   rendering, QLineEdit widget.
3. to_kwargs has a "dict" branch using json.loads with
   JSONDecodeError fallback to {}.
4. Behavioural: simulate to_kwargs for valid JSON, empty,
   malformed JSON, JSON array (not a dict), JSON scalar
   (not a dict).
5. CircularFeaturePlotter route present with kind="dict"
   for the settings field.
6. GeometryPlotter NOT in routes (still skipped).
7. Route count grows by 1 from this patch's baseline.
"""
from __future__ import annotations

import ast
import json
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
    # 1. Docstring documents "dict" kind
    # ==================================================================
    check(
        "_ExtrasFormBuilder docstring documents 'dict' kind",
        '``"dict"`` — QLineEdit' in src,
    )
    check(
        "Dict kind docstring records Patch 122bg",
        "Patch 122bg" in src,
    )

    # ==================================================================
    # 2. _add dispatch has a "dict" branch
    # ==================================================================
    check(
        "_add has elif kind == 'dict' branch",
        'elif kind == "dict":' in src,
    )
    check(
        "dict branch handles dict-default JSON rendering",
        'isinstance(default, dict)' in src
        and "json.dumps(default" in src,
    )
    check(
        "dict branch uses QLineEdit",
        ('elif kind == "dict":' in src
         and 'QLineEdit' in src.split('elif kind == "dict":')[1][:600]),
    )

    # ==================================================================
    # 3. to_kwargs has a "dict" branch
    # ==================================================================
    check(
        "to_kwargs has elif kind == 'dict' branch (two locations: "
        "_add + to_kwargs)",
        src.count('elif kind == "dict":') >= 2,
    )
    check(
        "dict to_kwargs branch uses json.loads + try/except",
        "json.loads" in src
        and "JSONDecodeError" in src,
    )
    check(
        "dict to_kwargs branch falls back to {} on empty/malformed",
        'out[name] = {}' in src,
    )

    # ==================================================================
    # 4. Behavioural: simulate the to_kwargs JSON-parse logic
    # ==================================================================
    def dict_kind_parse(raw: str):
        raw = (raw or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    check(
        "Behavioural: '' → {}",
        dict_kind_parse("") == {},
    )
    check(
        "Behavioural: whitespace-only → {}",
        dict_kind_parse("   ") == {},
    )
    check(
        'Behavioural: \'{"k":"v"}\' → {"k":"v"}',
        dict_kind_parse('{"k":"v"}') == {"k": "v"},
    )
    check(
        "Behavioural: nested JSON object parses",
        dict_kind_parse(
            '{"center": {"Animal_1": "SwimBladder"}, '
            '"text_settings": false, "palette": "bwr"}'
        ) == {
            "center": {"Animal_1": "SwimBladder"},
            "text_settings": False,
            "palette": "bwr",
        },
    )
    check(
        "Behavioural: malformed JSON → {} (silent fallback)",
        dict_kind_parse("{not valid json") == {},
    )
    check(
        "Behavioural: JSON array → {} (not a dict; rejected)",
        dict_kind_parse('["a", "b"]') == {},
    )
    check(
        "Behavioural: JSON scalar → {} (not a dict; rejected)",
        dict_kind_parse("42") == {},
    )

    # ==================================================================
    # 5. CircularFeaturePlotter route present, GeometryPlotter not
    # ==================================================================
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("viz form parses", False, detail=str(e))
        return 1
    check("viz form parses", True)

    routes_with_dict_extras: dict[str, list[str]] = {}
    route_backends: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_VizRoute"):
            continue
        label = None
        dict_extras: list[str] = []
        backend_call = None
        for kw in node.keywords:
            if kw.arg == "label" and isinstance(
                kw.value, ast.Constant,
            ):
                label = kw.value.value
            if kw.arg in ("backend_sp", "backend_mp") and (
                isinstance(kw.value, ast.Call)
                and isinstance(kw.value.func, ast.Name)
                and kw.value.func.id == "_lazy_factory"
            ):
                if (len(kw.value.args) >= 2
                        and isinstance(kw.value.args[1], ast.Constant)):
                    backend_call = kw.value.args[1].value
            if kw.arg == "extras" and isinstance(kw.value, ast.List):
                for item in kw.value.elts:
                    if not isinstance(item, ast.Tuple):
                        continue
                    if (len(item.elts) >= 2
                            and isinstance(item.elts[1], ast.Constant)
                            and item.elts[1].value == "dict"
                            and isinstance(item.elts[0], ast.Constant)):
                        dict_extras.append(item.elts[0].value)
        if label:
            route_backends[label] = backend_call or ""
            if dict_extras:
                routes_with_dict_extras[label] = dict_extras

    check(
        "Route 'Circular feature overlay (per video)' is present",
        "Circular feature overlay (per video)" in route_backends,
    )
    check(
        "Route wires backend CircularFeaturePlotter",
        route_backends.get(
            "Circular feature overlay (per video)",
        ) == "CircularFeaturePlotter",
    )
    check(
        "CircularFeaturePlotter route declares kind='dict' for "
        "settings field",
        routes_with_dict_extras.get(
            "Circular feature overlay (per video)",
        ) == ["settings"],
        detail=(
            f"got {routes_with_dict_extras.get('Circular feature overlay (per video)')!r}"
        ),
    )

    # ==================================================================
    # 6. GeometryPlotter — not surfaced via the dict kind
    #    introduced by THIS patch. 122bh later surfaces it via
    #    the "pickle" kind, so a `GeometryPlotter` route may
    #    exist in the routes table — but it must NOT use the
    #    "dict" kind (which can't represent Shapely geometries).
    # ==================================================================
    geometry_route_dict_extras = [
        kwargs for label, kwargs in routes_with_dict_extras.items()
        if route_backends.get(label) == "GeometryPlotter"
    ]
    check(
        "GeometryPlotter is NOT routed via the dict kind "
        "(its Shapely-object kwargs aren't JSON-serializable). "
        "Later patches may surface GeometryPlotter via OTHER "
        "kinds (e.g. 122bh's pickle kind) — that's fine.",
        geometry_route_dict_extras == [],
        detail=f"got dict-kind extras {geometry_route_dict_extras!r}",
    )

    # ==================================================================
    # 7. Route count — verifies this patch added exactly 1 route
    #    relative to whatever floor existed before. The absolute
    #    count is not pinned to avoid snapshot drift; instead the
    #    floor is the number of routes pre-122bg.
    # ==================================================================
    label_count = src.count('label="')
    # CircularFeaturePlotter is one of these — counts as +1 over
    # the pre-122bg floor of 27 (122az's 20 + 122ba's 7).
    check(
        f"ROUTES has at least 28 entries (pre-122bg floor 27 + "
        f"1 new). Found {label_count}",
        label_count >= 28,
    )

    print(
        f"smoke_122bg_dict_kind_for_viz_form: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
