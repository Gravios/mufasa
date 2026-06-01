"""
tests/smoke_122gf_pep604_runtime_union_fix.py
=============================================

Patch 122gf — fix runtime TypeError from PEP 604 annotation modernization
on NON-TYPE operands.

BUG (user-reported, app failed to launch):
    File ".../mufasa/utils/read_write.py", line 3345, in <module>
        pool: multiprocessing.Pool | None = None,
    TypeError: unsupported operand type(s) for |: 'method' and 'NoneType'

ROOT CAUSE
==========
Patch 122fx (UP045/UP007) rewrote `Optional[X]` / `Union[...]` to PEP 604
`X | None`. `Optional[X]` never evaluates `X | None`, so it tolerated
operands that are NOT types; PEP 604 syntax DOES evaluate `|` at class/def
definition time in modules WITHOUT `from __future__ import annotations`.
Two operand kinds crash:
  * multiprocessing.Pool  — a bound factory METHOD, not a class
    (`method | None` → TypeError). 30 sites across 6 files.
    (Note multiprocessing.pool.Pool, the real class, isn't even importable
     as `multiprocessing.pool` without an explicit submodule import, so the
     fix is not a class-name swap.)
  * Methods.ERROR.value / Methods.WARNING.value in tools.py — enum VALUES
    (strings) inside `Literal[None | "ERROR" | "WARNING"]` → `None | str`
    TypeError.

FIX
===
Add `from __future__ import annotations` to the 7 affected files. This
makes ALL annotations lazy strings (never evaluated at runtime), so the
`|` is never executed — restoring the pre-122fx behavior uniformly,
independent of operand type. None of the 7 files use runtime annotation
introspection (get_type_hints / dataclass / pydantic), so deferring is
safe. `from __future__ import annotations` is also stable against the ruff
sweep (UP010 never strips `annotations`; isort keeps __future__ first).

Verified NOT crashes (real classes, left as-is): np.ndarray, cp.ndarray,
and cuRF (always RandomForestClassifier from cuml or the sklearn fallback).

AFFECTED FILES (7)
==================
  data_processors/find_animal_blob_location.py   (multiprocessing.Pool)
  mixins/geometry_mixin.py                        (multiprocessing.Pool x24)
  plotting/geometry_plotter.py                    (multiprocessing.Pool)
  plotting/yolo_pose_visualizer.py                (multiprocessing.Pool)
  utils/read_write.py                             (multiprocessing.Pool)
  video_processors/video_processing.py            (multiprocessing.Pool)
  third_party_label_appenders/tools.py            (Methods.*.value)

NEW SMOKE: smoke_122gf_pep604_runtime_union_fix.py (3 checks)
* mufasa/ parses cleanly
* all 7 affected files carry `from __future__ import annotations`
* REGRESSION GUARD: no module-level/def annotation in mufasa/ evaluates
  `multiprocessing.Pool | ...` or an enum `.value | ...` at runtime
  without future-annotations (catches reintroduction).
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

AFFECTED = [
    "mufasa/data_processors/find_animal_blob_location.py",
    "mufasa/mixins/geometry_mixin.py",
    "mufasa/plotting/geometry_plotter.py",
    "mufasa/plotting/yolo_pose_visualizer.py",
    "mufasa/utils/read_write.py",
    "mufasa/video_processors/video_processing.py",
    "mufasa/third_party_label_appenders/tools.py",
]

CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return True
    return False


def ann_nodes(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and n.annotation:
            yield n.annotation
        if isinstance(n, ast.AnnAssign) and n.annotation:
            yield n.annotation
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns:
            yield n.returns


def risky_union_operand(op) -> bool:
    # multiprocessing.Pool (Attribute attr=='Pool' on a multiprocessing-ish base)
    if isinstance(op, ast.Attribute) and op.attr == "Pool":
        return True
    # enum value: X.Y.value
    if isinstance(op, ast.Attribute) and op.attr == "value":
        return True
    return False


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    parse_errors = []
    trees = {}
    for f in sorted(pkg.rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        try:
            trees[f] = ast.parse(src)
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"all mufasa/**/*.py parse cleanly ({len(trees)} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    missing = [
        p for p in AFFECTED
        if not has_future_annotations(ast.parse((REPO_ROOT / p).read_text()))
    ]
    check(
        "all 7 affected files carry `from __future__ import annotations`",
        not missing,
        detail=f"missing in: {missing}",
    )

    # Regression guard: any file with a risky | union operand in an annotation
    # MUST have future-annotations (else it crashes at import).
    offenders = []
    for f, tree in trees.items():
        if has_future_annotations(tree):
            continue
        for ann in ann_nodes(tree):
            for b in ast.walk(ann):
                if isinstance(b, ast.BinOp) and isinstance(b.op, ast.BitOr):
                    if any(risky_union_operand(o) for o in (b.left, b.right)):
                        offenders.append(str(f.relative_to(REPO_ROOT)))
                        break
    check(
        "no runtime-evaluated `multiprocessing.Pool | ...` or `.value | ...` "
        "annotation lacks future-annotations (reintroduction guard)",
        not offenders,
        detail=f"offenders: {sorted(set(offenders))}",
    )

    print(
        f"smoke_122gf_pep604_runtime_union_fix: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
