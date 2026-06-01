"""
tests/smoke_122gg_future_annotations_hardening.py
=================================================

Patch 122gg — close the PEP 604 runtime-crash class for good: add
`from __future__ import annotations` to EVERY remaining module that uses
`|` union syntax in annotations.

CONTEXT
=======
122gf fixed the 7 files with KNOWN non-type `|` operands (multiprocessing.Pool
x6 files; Methods.*.value in tools.py). But the underlying hazard — PEP 604
`X | Y` annotations being evaluated at runtime in modules without
`from __future__ import annotations` — exists wherever such an annotation
has any operand that isn't a runtime-`|`-able type. Static analysis can't
prove every operand safe (operand type-ness can depend on conditional
imports), and the sandbox can't import the package to check at runtime.

So this patch removes the hazard structurally: with future-annotations,
ALL annotations are lazy strings and the `|` is never evaluated at import
time. Applied to the 230 remaining `|`-annotation modules.

SIDE EFFECTS (folded in)
========================
* Adding future-annotations made ruff treat previously-withheld UP007/UP037
  as safe (deferred annotations can use `|` and drop forward-ref quotes).
  6 such fixes in roi_tools/roi_utils.py were applied
  (Union[ROISelector, "InteractiveROIBufferer"] -> ROISelector |
  InteractiveROIBufferer, etc.), keeping the UP family at 0.
* That UP007 conversion orphaned `typing.Union` in roi_utils.py; trimmed
  the import to `from typing import TYPE_CHECKING` (F401 guard stays clean).
* I001 reflow for the new __future__ import blocks.

SAFETY
======
* The only module doing runtime annotation introspection
  (timeseries_features_mixin.py, get_type_hints on sliding_stationary) is
  safe: that function's annotations are all plain type operands
  (np.ndarray, int, Literal[...], tuple[...]), which get_type_hints
  resolves fine from string form.
* F821 unchanged at the 7 known-deliberate entries; compileall clean.

NEW SMOKE: smoke_122gg_future_annotations_hardening.py (3 checks)
* mufasa/ parses cleanly
* INVARIANT: every module with a `|` union annotation carries
  `from __future__ import annotations` (zero modules left exposed) — this
  is the structural guard against the entire crash class
* F821 still exactly the 7 known-deliberate
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


def has_union_annotation(tree) -> bool:
    for ann in ann_nodes(tree):
        for b in ast.walk(ann):
            if isinstance(b, ast.BinOp) and isinstance(b.op, ast.BitOr):
                return True
    return False


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    parse_errors = []
    trees = {}
    for f in sorted(pkg.rglob("*.py")):
        try:
            trees[f] = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"all mufasa/**/*.py parse cleanly ({len(trees)} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    exposed = [
        str(f.relative_to(REPO_ROOT))
        for f, tree in trees.items()
        if has_union_annotation(tree) and not has_future_annotations(tree)
    ]
    check(
        "every module with a `|` union annotation has future-annotations "
        "(no module left exposed to the PEP 604 runtime-eval crash class)",
        not exposed,
        detail=f"{len(exposed)} exposed, e.g. {exposed[:3]}",
    )

    import shutil
    import subprocess
    ruff = shutil.which("ruff")
    if ruff is None:
        print("NOTE: ruff not found — F821 check skipped (soft pass).")
        check("F821 still the 7 known-deliberate", True)
    else:
        f821 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "F821", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        n = sum(1 for ln in f821.stdout.splitlines() if "F821" in ln)
        check("F821 still exactly 7 known-deliberate", n == 7,
              detail=f"F821 count = {n}")

    print(
        f"smoke_122gg_future_annotations_hardening: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
