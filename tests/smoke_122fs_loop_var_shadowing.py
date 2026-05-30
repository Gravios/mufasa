"""
tests/smoke_122fs_loop_var_shadowing.py
=======================================

Patch 122fs — fix the two B020 loop-variable-overrides-iterator sites
flagged in the 122fp audit.

User request: "continue" (autonomous backlog selection). The only
remaining functional item, converters.geometry_to_rle, needs a
pycocotools dependency decision (out of scope to guess), so the next
sandbox-verifiable cleanup was taken instead.

CONTEXT / WHY
=============
Two loops used the same name for both the iterable and the loop target:

  mufasa/mixins/pose_importer_mixin.py
    for x_col, y_col, p_cols in zip(x_cols, y_cols, p_cols):
        df = self.data_df[[x_col, y_col, p_cols]]
  mufasa/plotting/ROI_feature_visualizer_mp.py  (__insert_texts)
    for shape_name, shape_info in shape_info.items():
        shape_color = shape_info["Color BGR"]

These WORK today (zip()/.items() build their iterator before the loop
target rebinds), which is why they were deferred as cosmetic. But ruff
flags B020 because the pattern is a latent footgun: any later edit that
references the iterable again inside or after the loop silently breaks.
Renaming the loop target (and its single in-body use) removes the shadow
with no behavioural change.

WHAT THIS PATCH LANDED
======================
* pose_importer_mixin.py: loop target p_cols -> p_col (iterable p_cols
  unchanged); the one body reference updated.
* ROI_feature_visualizer_mp.py: loop target shape_info -> shape_data
  (param/iterable shape_info unchanged); the one body reference updated.

NEW SMOKE: smoke_122fs_loop_var_shadowing.py (4 checks)
* both sites use the renamed, non-shadowing loop target
* PACKAGE-WIDE drift guard: no `for` loop anywhere in mufasa/ has a
  target name that also appears in its iterable (AST-level B020 proxy)
* package parses
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


def _names(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _shadow_sites(src: str, rel: str) -> list:
    """For-loops whose target names intersect their iterable names (B020)."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if _names(node.target) & _names(node.iter):
                out.append(f"{rel}:{node.lineno}")
    return out


def main() -> int:
    pim = _read("mufasa/mixins/pose_importer_mixin.py")
    check(
        "pose_importer_mixin loop target renamed p_cols->p_col",
        "for x_col, y_col, p_col in zip(x_cols, y_cols, p_cols):" in pim
        and "self.data_df[[x_col, y_col, p_col]]" in pim
        and "for x_col, y_col, p_cols in zip" not in pim,
    )

    rfv = _read("mufasa/plotting/ROI_feature_visualizer_mp.py")
    check(
        "ROI_feature_visualizer loop target renamed shape_info->shape_data",
        "for shape_name, shape_data in shape_info.items():" in rfv
        and 'shape_color = shape_data["Color BGR"]' in rfv
        and "for shape_name, shape_info in shape_info.items():" not in rfv,
    )

    # PACKAGE-WIDE drift guard
    pkg = REPO_ROOT / "mufasa"
    shadows = []
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        rel = str(f.relative_to(REPO_ROOT))
        try:
            shadows.extend(_shadow_sites(f.read_text(encoding="utf-8"), rel))
        except SyntaxError as e:
            parse_errors.append(f"{rel}: {e}")
    check(
        "no for-loop target shadows its iterable anywhere in mufasa/",
        not shadows,
        detail=", ".join(shadows[:5]),
    )
    check(
        f"all mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fs_loop_var_shadowing: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
