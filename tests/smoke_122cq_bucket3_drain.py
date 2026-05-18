"""
tests/smoke_122cq_bucket3_drain.py
====================================

Patch 122cq: reclassify `roi_tools/roi_ui_mixin.py` from §3d
Bucket 3 (Qt-deferred) to Bucket 2 (dies with Tk surface).

The original 122ck audit treated four Qt-side docstring references
to `ROI_ui` as evidence that Qt code consumed the Tk mixin
transitively. The 122cq re-audit shows those references are all
docstrings, not real `ImportFrom` nodes. The real Qt ROI surface
imports `mufasa.roi_tools.roi_logic` directly — a separate
UI-framework-independent module that's been the Qt foundation
since the Qt ROI port shipped.

This patch lands no code deletion. The reclassification is the
honest scope correction: Bucket 3 is DRAINED, and the actual
cluster-deletion (5 files + SimBA.py edit) is staged for a
future patch.

Coverage
--------
1. backend_audit.md §3d Bucket 3 is marked DRAINED in 122cq.
2. §3d Bucket 3 documents the docstring-vs-import error.
3. §3d Bucket 2 now lists `roi_tools/roi_ui_mixin.py`.
4. §3d Bucket 2 now lists `roi_tools/roi_ui.py`.
5. §3d Bucket 2 cluster-shapes section describes the ROI Tk
   cluster (almost-closed, similar to labelling).
6. tk_surface_audit.md has new §2g with the docstring-vs-import
   lesson.
7. §2g lists the four false-positive Qt docstring locations.
8. §2g describes the methodology fix.
9. §7 audit methodology has the strengthened "Only ast.ImportFrom
   counts" directive.
10. AST audit verifies: no Qt file (`ui_qt/**`) has a real
    `from … import ROI_ui` statement.
11. AST audit verifies: only Tk consumers exist for ROI_ui
    (blob_tracker_ui.py + roi_video_table_pop_up.py).
12. AST audit verifies: roi_ui_mixin.py is consumed only by
    roi_ui.py.
13. AST audit verifies: Qt ROI panels (roi_canvas.py +
    roi_define_panel.py) DO import from roi_logic, not from
    roi_ui_mixin.
14. All mufasa/**/*.py files parse cleanly.
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


def _real_importers(symbol: str, pkg: Path) -> list[Path]:
    """Return files that have an ast.ImportFrom node naming
    `symbol`. Excludes the file that defines `symbol`."""
    result = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == symbol:
                        # Exclude the defining file itself —
                        # heuristic: the file whose own __init__
                        # would define the symbol. We compare by
                        # checking if symbol is defined in `f`.
                        own_def = False
                        for top in t.body:
                            if (isinstance(top, (ast.ClassDef,
                                                 ast.FunctionDef))
                                    and top.name == symbol):
                                own_def = True
                                break
                        if not own_def:
                            result.append(f)
                            break  # avoid double-listing
    return result


def _module_imports(target_module: str,
                    pkg: Path,
                    only_in: str = "") -> list[Path]:
    """Return files that have `from {target_module} import …`
    where the file path contains `only_in`."""
    result = []
    for f in pkg.rglob("*.py"):
        if only_in and only_in not in str(f):
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == target_module):
                result.append(f)
                break
    return result


def main() -> int:
    docs_dir = REPO_ROOT / "docs"
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # backend_audit.md updates
    # ==================================================================
    ba = (docs_dir / "backend_audit.md").read_text()
    check(
        "§3d Bucket 3 marked DRAINED in 122cq",
        ("DRAINED 122cq" in ba or "DRAINED in 122cq" in ba)
        and "0 files" in ba,
    )
    check(
        "§3d Bucket 3 documents the docstring-vs-import error",
        "docstrings" in ba
        and ("122ck audit" in ba or "122ck re-audit" in ba
             or "re-re-audit" in ba),
    )
    check(
        "§3d Bucket 2 references roi_tools/roi_ui_mixin.py "
        "(may be strikethroughed if 122cr deletion has happened)",
        # Pin to the durable claim: the file is mentioned in
        # Bucket 2 (either active or strikethrough/DELETED).
        # Was "active entry + reclassified from Bucket 3" pre-122cr;
        # becomes "strikethrough + DELETED 122cr" post-122cr.
        "`roi_tools/roi_ui_mixin.py`" in ba
        and ("reclassified from Bucket 3" in ba
             or "DELETED 122cr" in ba),
    )
    check(
        "§3d Bucket 2 references roi_tools/roi_ui.py",
        "`roi_tools/roi_ui.py`" in ba,
    )
    check(
        "§3d Bucket 2 cluster-shapes describes ROI Tk cluster",
        "ROI Tk cluster" in ba
        and ("almost-closed" in ba.lower()),
    )

    # ==================================================================
    # tk_surface_audit.md §2g + §7 updates
    # ==================================================================
    ta = (docs_dir / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md has new §2g",
        "### 2g. ROI Tk cluster re-audit" in ta,
    )
    check(
        "§2g lists the four false-positive Qt docstring locations",
        "roi_video_table.py" in ta and "roi.py" in ta
        and "all docstrings" in ta,
    )
    check(
        "§2g describes the methodology fix",
        "ast.ImportFrom" in ta
        or "ImportFrom" in ta,
    )
    check(
        "§7 audit methodology has the strengthened "
        "'Only ast.ImportFrom counts' directive",
        "Only `ast.ImportFrom`" in ta
        or "ast.ImportFrom`" in ta
        and "docstring" in ta.lower(),
    )

    # ==================================================================
    # AST verification: no Qt file really imports ROI_ui
    # ==================================================================
    roi_ui_importers = _real_importers("ROI_ui", pkg)
    qt_real_importers = [p for p in roi_ui_importers
                         if "ui_qt" in p.parts]
    check(
        "no Qt file (ui_qt/**) has a real `from … import ROI_ui` "
        f"statement (got {len(qt_real_importers)} Qt importers)",
        qt_real_importers == [],
        detail=", ".join(str(p) for p in qt_real_importers),
    )
    # Pre-deletion: 2 Tk consumers. Post-122cr-deletion: ROI_ui
    # no longer exists; 0 importers expected.
    tk_importers = [p for p in roi_ui_importers
                    if "ui_qt" not in p.parts]
    check(
        f"ROI_ui has either 2 Tk consumers (pre-122cr) or 0 "
        f"consumers (post-122cr-deletion); got "
        f"{len(tk_importers)}",
        len(tk_importers) in (0, 2),
        detail=f"got: {[str(p) for p in tk_importers]}",
    )

    # ROI_mixin: pre-deletion 1 importer (roi_ui.py);
    # post-122cr-deletion 0 importers.
    roi_mixin_importers = _real_importers("ROI_mixin", pkg)
    check(
        f"ROI_mixin has either 1 importer (pre-122cr: roi_ui.py) "
        f"or 0 importers (post-122cr-deletion); got "
        f"{len(roi_mixin_importers)}",
        len(roi_mixin_importers) in (0, 1),
        detail=f"got: {[str(p) for p in roi_mixin_importers]}",
    )

    # Qt panels do import from roi_logic
    qt_logic_importers = _module_imports(
        "mufasa.roi_tools.roi_logic", pkg, only_in="ui_qt")
    check(
        f"Qt panels DO import from roi_logic (got "
        f"{len(qt_logic_importers)} Qt importer(s))",
        len(qt_logic_importers) >= 1
        and any(("roi_canvas" in str(p)
                 or "roi_define_panel" in str(p))
                for p in qt_logic_importers),
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
        f"smoke_122cq_bucket3_drain: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
