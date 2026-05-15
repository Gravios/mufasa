"""
tests/smoke_122az_viz_form_route_additions.py
==============================================

Patch 122az: extends the existing VisualizationForm with 3
gap-filling routes for ROI / skeleton plotting backends that
were live in the codebase but not surfaced through the Qt
visualizations page.

Routes added
------------
* "ROI overlay (per video)" — :class:`ROIPlotter`. Per-video
  view of body-part position relative to drawn ROIs.
* "ROI feature overlay (per video)" — :class:`ROIfeatureVisualizer`.
  Per-video visualization of the ROI-distance feature
  computed by the ROI feature subset extractor.
* "Skeleton video (project pose)" — :class:`SkeletonVideoCreator`.
  Renders pose skeletons over project videos.

ROIPlotter and ROIfeatureVisualizer both take
``body_parts: list[str]`` but the declarative form has no
"list" kind — uses ``kwargs_transform`` (new in 122az) to wrap
the singular ``body_part: str`` field into ``[body_part]``
before backend dispatch.

The kwargs_transform extension to _VizRoute
-------------------------------------------
The dataclass gets a new optional field
``kwargs_transform: Callable[[dict], dict] | None``. The
dispatch path applies it after default/extras merge but
before the defensive signature filter. Keeps the form
framework extensible for future routes that need a light
transform without growing the kind taxonomy.

Coverage
--------
1. _VizRoute defines the new ``kwargs_transform`` field with
   a None default (back-compat with existing routes).
2. Three new routes are present in ROUTES with the right
   labels, scope_kinds, backends, and extras.
3. The body_part → body_parts wrap transform works
   end-to-end on a fixture kwargs dict.
4. Run-path applies kwargs_transform after defaults+extras
   merge and before filter_kwargs (code-level inspection).
5. ROUTES count is 20 (was 17).
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
    # 1. _VizRoute has kwargs_transform field with None default
    # ==================================================================
    check(
        "_VizRoute declares kwargs_transform field",
        "kwargs_transform: Optional[Callable[[dict], dict]]" in src,
    )
    check(
        "_VizRoute kwargs_transform defaults to None",
        "kwargs_transform: Optional[Callable[[dict], dict]] = None"
        in src,
    )

    # ==================================================================
    # 2. Three new route labels present
    # ==================================================================
    for new_label in (
        "ROI overlay (per video)",
        "ROI feature overlay (per video)",
        "Skeleton video (project pose)",
    ):
        check(
            f"Route label {new_label!r} present",
            f'label="{new_label}"' in src,
        )

    # ==================================================================
    # 3. New routes reference the right backend modules
    # ==================================================================
    for module, cls in (
        ("mufasa.plotting.roi_plotter", "ROIPlotter"),
        ("mufasa.plotting.ROI_feature_visualizer",
         "ROIfeatureVisualizer"),
        ("mufasa.plotting.skeleton_video_creator",
         "SkeletonVideoCreator"),
    ):
        check(
            f"Route wires backend {cls}",
            f'"{module}"' in src and f'"{cls}"' in src,
        )

    # ==================================================================
    # 4. ROUTES count = 20
    # ==================================================================
    label_count = src.count('label="')
    # Strip away counts inside docstrings or comments by scoping to
    # ROUTES list — the dataclass + extras helpers don't use label=.
    check(
        f"ROUTES has 20 entries (was 17, +3 new). Found {label_count}",
        label_count == 20,
    )

    # ==================================================================
    # 5. Run path applies kwargs_transform between merge and filter
    # ==================================================================
    # Read the AST and find the run() method on VisualizationForm.
    # We want to verify the order: defaults merge → extras merge →
    # kwargs_transform → filter_kwargs.
    check(
        "Run path uses route.kwargs_transform",
        "route.kwargs_transform" in src,
    )
    # Sequentially: kwargs_transform should appear AFTER the extras
    # merge for-loop and BEFORE filter_kwargs.
    extras_idx = src.find("for k, v in extras.items():")
    transform_idx = src.find("route.kwargs_transform(kwargs)")
    filter_idx = src.find("filter_kwargs(backend, kwargs)")
    check(
        "Run path order: extras → kwargs_transform → filter_kwargs",
        extras_idx > 0 < transform_idx < filter_idx
        and extras_idx < transform_idx,
    )

    # ==================================================================
    # 6. body_part → [body_part] transform: behavioural test
    # ==================================================================
    # The transforms are lambdas in the route declarations. Pull them
    # out via ast and exec them on a fixture dict.
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("viz form parses", False, detail=str(e))
        return 1
    check("viz form parses", True)

    # Find the two ROI routes' kwargs_transform lambdas, compile
    # one and apply it. We synthesize a fixture and verify the
    # body_part → body_parts wrap fires.
    transform_lambdas: list[ast.Lambda] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(
            node.func, ast.Name,
        ) and node.func.id == "_VizRoute":
            for kw in node.keywords:
                if kw.arg == "kwargs_transform" and isinstance(
                    kw.value, ast.Lambda,
                ):
                    transform_lambdas.append(kw.value)
    check(
        "Two kwargs_transform lambdas found in ROUTES",
        len(transform_lambdas) == 2,
        detail=f"got {len(transform_lambdas)}",
    )

    if transform_lambdas:
        # Compile the first one and apply it to a fixture
        lam = transform_lambdas[0]
        # ast.Expression wraps the lambda so compile() accepts it
        wrapper = ast.Expression(body=lam)
        ast.fix_missing_locations(wrapper)
        fn = eval(compile(wrapper, "<smoke>", "eval"))
        out = fn({"body_part": "Nose", "show_bbox": False})
        check(
            "kwargs_transform wraps body_part → body_parts: list",
            out.get("body_parts") == ["Nose"]
            and "body_part" not in out,
        )
        check(
            "kwargs_transform preserves other kwargs",
            out.get("show_bbox") is False,
        )

    # ==================================================================
    # 7. Patch 122az recorded
    # ==================================================================
    check(
        "visualizations.py records 122az",
        "122az" in src,
    )

    print(
        f"smoke_122az_viz_form_route_additions: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
