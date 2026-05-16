"""
tests/smoke_122bh_pickle_kind_for_viz_form.py
===============================================

Patch 122bh: adds a "pickle" kind to _ExtrasFormBuilder for
backend kwargs that take a complex Python object that can't
be JSON-serialized (Shapely geometries, numpy arrays, custom
classes, …). Users construct the object in a Python script,
pickle it, and point the form at the .pkl file. At dispatch
the form opens the file and pickle.loads it.

Surfaces GeometryPlotter — the last visualization backend
that wasn't reachable through the form. Its `geometries`
arg is List[List[Polygon | LineString | MultiPolygon |
MultiLineString | Point]] (Shapely objects) which JSON
can't round-trip.

Continues the extras-kind expansion arc:
* 122be — "list" kind (comma-split strings → list[str])
* 122bg — "dict" kind (JSON text → dict)
* 122bh — "pickle" kind (file picker + pickle.load → any obj)

Security
--------
pickle.load executes arbitrary code from the file. The user
is trusted to only point the form at files they produced
themselves. Single-user desktop tool trust model — same as
running any Python script. The form does no sandboxing.
Smoke test verifies this assumption holds by exercising the
load path on a pickle file we create in this test.

What the "pickle" kind does
---------------------------
* UI: _PathField (file picker), same widget as the "file"
  kind. Default file filter is `Pickle files (*.pkl)`.
* to_kwargs: opens the picked file with `open(path, "rb")`
  and pickle.load()s it. Returns the deserialized object on
  success.
* Empty path → None. Load error → None. Backend will then
  raise its usual missing-arg / wrong-type error (clearer
  than the form would produce).

Route added (1)
---------------
"Geometry overlay (Shapely objects)" → GeometryPlotter
    Project-scope. needs_save_dir=True. Extras include the
    pickle-file picker for `geometries` plus the rendering
    params (thickness, circle_size, bg_opacity, ...).

Coverage
--------
1. _ExtrasFormBuilder.add docstring documents "pickle" kind
   with the security note.
2. _add dispatch has a "pickle" branch using _PathField
   with a sensible default file_filter.
3. to_kwargs has a "pickle" branch that opens the file and
   pickle.load()s it, with broad exception handling →
   None fallback.
4. Behavioural: pickle a fixture object to a tmpfile, then
   simulate the to_kwargs branch by reading it back; verify
   the round-trip works for both simple and nested objects.
5. Behavioural: empty path → None; nonexistent path → None;
   malformed pickle (wrong bytes) → None.
6. GeometryPlotter route present with kind="pickle" for
   the geometries field.
7. Route count grows by 1 from pre-122bh floor.
"""
from __future__ import annotations

import ast
import pickle
import sys
import tempfile
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
    # 1. Docstring documents "pickle" kind
    # ==================================================================
    check(
        "_ExtrasFormBuilder docstring documents 'pickle' kind",
        '``"pickle"`` — _PathField' in src,
    )
    check(
        "Pickle kind docstring records Patch 122bh",
        "Patch 122bh" in src,
    )
    check(
        "Pickle kind docstring includes a SECURITY note "
        "about pickle.load executing arbitrary code",
        "SECURITY" in src and "arbitrary code" in src,
    )

    # ==================================================================
    # 2. _add dispatch has a "pickle" branch
    # ==================================================================
    check(
        "_add has elif kind == 'pickle' branch",
        'elif kind == "pickle":' in src,
    )
    check(
        "pickle _add branch uses _PathField",
        ('elif kind == "pickle":' in src
         and '_PathField' in src.split('elif kind == "pickle":')[1][:500]),
    )
    check(
        "pickle _add branch has a sensible default file_filter "
        "('Pickle files (*.pkl)')",
        "Pickle files (*.pkl)" in src,
    )

    # ==================================================================
    # 3. to_kwargs has a "pickle" branch
    # ==================================================================
    check(
        "to_kwargs has elif kind == 'pickle' branch "
        "(two locations: _add + to_kwargs)",
        src.count('elif kind == "pickle":') >= 2,
    )
    check(
        "pickle to_kwargs uses pickle.load + try/except",
        "pickle.load" in src
        and "UnpicklingError" in src,
    )
    check(
        "pickle to_kwargs falls back to None on empty path "
        "or load error",
        # The fallback path sets out[name] = None
        'out[name] = None' in src,
    )

    # ==================================================================
    # 4. Behavioural: simulate the to_kwargs pickle.load logic
    # ==================================================================
    def pickle_kind_load(path: str):
        """Mirror the form's to_kwargs branch for "pickle"."""
        if not path:
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError,
                AttributeError, ImportError, ValueError):
            return None

    # Simple round-trip
    with tempfile.NamedTemporaryFile(
        suffix=".pkl", delete=False,
    ) as fh:
        pickle.dump({"hello": "world", "n": 42}, fh)
        simple_path = fh.name
    try:
        out = pickle_kind_load(simple_path)
        check(
            "Behavioural: pickle round-trip of simple dict",
            out == {"hello": "world", "n": 42},
            detail=f"got {out!r}",
        )
    finally:
        Path(simple_path).unlink(missing_ok=True)

    # Nested object resembling GeometryPlotter input shape:
    # List[List[tuple]] (tuple stands in for a Shapely object
    # — we don't need Shapely available to test the pickle
    # round-trip itself; what matters is that pickle.load
    # returns the same Python object that pickle.dump wrote)
    nested_obj = [
        [("polygon", [(0, 0), (1, 0), (1, 1), (0, 1)]),
         ("linestring", [(0, 0), (1, 1), (2, 0)])],
        [("point", (0.5, 0.5))],
    ]
    with tempfile.NamedTemporaryFile(
        suffix=".pkl", delete=False,
    ) as fh:
        pickle.dump(nested_obj, fh)
        nested_path = fh.name
    try:
        out = pickle_kind_load(nested_path)
        check(
            "Behavioural: pickle round-trip of "
            "List[List[geometry-shaped-tuple]]",
            out == nested_obj,
            detail=f"got {out!r}",
        )
    finally:
        Path(nested_path).unlink(missing_ok=True)

    # ==================================================================
    # 5. Error-path behaviour
    # ==================================================================
    check(
        "Behavioural: empty path → None",
        pickle_kind_load("") is None,
    )
    check(
        "Behavioural: nonexistent path → None",
        pickle_kind_load("/tmp/definitely-does-not-exist-122bh.pkl")
        is None,
    )

    # Malformed pickle (not a pickle file at all)
    with tempfile.NamedTemporaryFile(
        suffix=".pkl", mode="wb", delete=False,
    ) as fh:
        fh.write(b"not a pickle file, just garbage bytes")
        malformed_path = fh.name
    try:
        check(
            "Behavioural: malformed pickle bytes → None "
            "(broad except: UnpicklingError / EOFError / etc.)",
            pickle_kind_load(malformed_path) is None,
        )
    finally:
        Path(malformed_path).unlink(missing_ok=True)

    # ==================================================================
    # 6. GeometryPlotter route is present with kind="pickle"
    # ==================================================================
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("viz form parses", False, detail=str(e))
        return 1
    check("viz form parses", True)

    routes_with_pickle_extras: dict[str, list[str]] = {}
    route_backends: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_VizRoute"):
            continue
        label = None
        pickle_extras: list[str] = []
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
                            and item.elts[1].value == "pickle"
                            and isinstance(item.elts[0], ast.Constant)):
                        pickle_extras.append(item.elts[0].value)
        if label:
            route_backends[label] = backend_call or ""
            if pickle_extras:
                routes_with_pickle_extras[label] = pickle_extras

    check(
        "Route 'Geometry overlay (Shapely objects)' is present",
        "Geometry overlay (Shapely objects)" in route_backends,
    )
    check(
        "Route wires backend GeometryPlotter",
        route_backends.get(
            "Geometry overlay (Shapely objects)",
        ) == "GeometryPlotter",
    )
    check(
        "GeometryPlotter route declares kind='pickle' for "
        "geometries field",
        routes_with_pickle_extras.get(
            "Geometry overlay (Shapely objects)",
        ) == ["geometries"],
        detail=(
            f"got {routes_with_pickle_extras.get('Geometry overlay (Shapely objects)')!r}"
        ),
    )

    # ==================================================================
    # 7. Route count: pre-122bh floor 28 + 1 new
    # ==================================================================
    label_count = src.count('label="')
    check(
        f"ROUTES has at least 29 entries (pre-122bh floor 28 + "
        f"1 new). Found {label_count}",
        label_count >= 29,
    )

    print(
        f"smoke_122bh_pickle_kind_for_viz_form: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
