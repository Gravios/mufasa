"""
tests/smoke_122fp_audit_runtime_fixes.py
========================================

Patch 122fp — fix nine confirmed runtime crashes surfaced by a
static audit (undefined names + boolean-or except clauses).

User request (Fri May 30, 2026):

> audit the project for errors and style.   (then: "yes" — fix them)

CONTEXT / WHY
=============
The strict smoke suite is AST-extraction only (PySide6/cv2/h5py are
unimportable in the sandbox), so it never *executes* product code and
therefore cannot catch NameError-class bugs or boolean `except`
clauses. A ruff F821/B030 sweep found nine code paths that raise at
runtime the moment they are reached:

  1. model/regression/model.py        fit_xgb -> `xgb_reg` (param is `mdl`)
  2. utils/read_write.py              `gc.collect()` with gc unimported
  3. utils/read_write.py              `get_pose_config_dir.__name__`
                                      (enclosing fn is get_env_pose_config_dir)
  4. scripts/upgrade_simba_keep_configs.py   same get_pose_config_dir slip
  5. mixins/train_model_mixin.py:1705 `except BrokenProcessPool or AttributeError:`
  6. mixins/train_model_mixin.py:2244 `except ValueError or TypeError:`
  7. mixins/network_mixin.py          `G.number_of_nodes()` (var is `graph`)
  8. mixins/plotting_mixin.py         `bg_clr_rgb = bg_img` (var is `bg_clr`)
  9. roi_tools/roi_clf_calculator_mp.py  `self.__class__.__name__` in a
                                      standalone MP worker (sibling at the
                                      line above already uses the fn name)

`except A or B:` is the subtle one: `A or B` collapses to `A`, so the
second type is silently never caught.

WHAT THIS PATCH DID NOT CHANGE
==============================
* reverse_pose.py — three called names (extract_features_wotarget_9,
  extract_features_wotarget_14_from_16, check_that_two_dfs_are_equal_len)
  are defined NOWHERE in the tree. Fixing needs the deleted defs or a
  domain decision on the right replacement; guessing would be worse than
  the honest NameError. Left for a follow-up with project knowledge.
* converters.py — `geometry_to_rle` is commented out at line 51 but
  still called at 121. Restoring it is a real implementation task.
* Two B020 loop-var-overrides-iterator sites (pose_importer_mixin,
  ROI_feature_visualizer_mp) WORK today (zip()/.items() capture the
  iterable before the loop var rebinds); cosmetic, deferred.
* The annotation-only F821s (kalman_v2 Path, project_layout Union) are
  inert under `from __future__ import annotations`.
* The broad style sweep (UP/I/E701/E722/B904/B905 …) is intentionally
  out of scope — separate mechanical commit.

NEW SMOKE: smoke_122fp_audit_runtime_fixes.py (12 checks)
* one check per fix (source-level, AST-validated)
* PACKAGE-WIDE drift guard: no ExceptHandler anywhere binds a BoolOp
  type (locks out re-introduction of the `except A or B:` antipattern)
* all mufasa/**/*.py still parse cleanly
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


def _func_src(src: str, fn_name: str) -> str:
    """ast.unparse of the first top-level/any-level FunctionDef named fn_name."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return ast.unparse(node)
    return ""


def _boolop_except_sites(src: str) -> list:
    """Return line numbers of `except <BoolOp>:` handlers (the A-or-B bug)."""
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.BoolOp):
            hits.append(node.lineno)
    return hits


def main() -> int:
    # --- Fix 1: fit_xgb uses `mdl`, never `xgb_reg` -----------------------
    model_src = _read("mufasa/model/regression/model.py")
    fit_xgb = _func_src(model_src, "fit_xgb")
    check(
        "model.fit_xgb checks instance=mdl (not xgb_reg)",
        "instance=mdl" in fit_xgb and "xgb_reg" not in model_src,
        detail="xgb_reg still present" if "xgb_reg" in model_src else "instance=mdl missing",
    )

    # --- Fix 2: read_write imports gc -------------------------------------
    rw_src = _read("mufasa/utils/read_write.py")
    rw_tree = ast.parse(rw_src)
    imports_gc = any(
        isinstance(n, ast.Import) and any(a.name == "gc" for a in n.names)
        for n in ast.walk(rw_tree)
    )
    check("read_write.py imports gc (gc.collect no longer NameErrors)", imports_gc)

    # --- Fix 3: read_write get_env_pose_config_dir.__name__ ---------------
    check(
        "read_write not-found branch references get_env_pose_config_dir",
        "source=get_env_pose_config_dir.__name__" in rw_src
        and "=get_pose_config_dir.__name__" not in rw_src,
    )

    # --- Fix 4: same slip in the upgrade script ---------------------------
    up_src = _read("scripts/upgrade_simba_keep_configs.py")
    check(
        "upgrade script references get_env_pose_config_dir",
        "source=get_env_pose_config_dir.__name__" in up_src
        and "=get_pose_config_dir.__name__" not in up_src,
    )

    # --- Fixes 5 & 6: train_model_mixin except clauses are tuples ---------
    tmm_src = _read("mufasa/mixins/train_model_mixin.py")
    check(
        "train_model_mixin: (BrokenProcessPool, AttributeError) tuple except",
        "except (BrokenProcessPool, AttributeError):" in tmm_src,
    )
    check(
        "train_model_mixin: (ValueError, TypeError) tuple except",
        "except (ValueError, TypeError):" in tmm_src,
    )
    check(
        "train_model_mixin: no `except A or B:` handlers remain",
        not _boolop_except_sites(tmm_src),
        detail=f"BoolOp except at lines {_boolop_except_sites(tmm_src)}",
    )

    # --- Fix 7: network_mixin uses `graph`, not `G` -----------------------
    nm_src = _read("mufasa/mixins/network_mixin.py")
    check(
        "network_mixin CountError uses graph.number_of_nodes()",
        "graph.number_of_nodes()" in nm_src and "G.number_of_nodes()" not in nm_src,
    )

    # --- Fix 8: plotting_mixin passthrough uses bg_clr --------------------
    pm_src = _read("mufasa/mixins/plotting_mixin.py")
    check(
        "plotting_mixin non-3ch branch assigns bg_clr (not bg_img)",
        "bg_clr_rgb = bg_clr" in pm_src and "bg_clr_rgb = bg_img" not in pm_src,
    )

    # --- Fix 9: roi_clf_calculator_mp worker uses fn name, not self -------
    # (class methods legitimately use self.__class__.__name__; scope the
    #  assertion to the standalone _clf_by_roi_helper worker only.)
    roi_src = _read("mufasa/roi_tools/roi_clf_calculator_mp.py")
    helper_src = _func_src(roi_src, "_clf_by_roi_helper")
    check(
        "roi_clf_calculator_mp worker: no `self` in standalone helper",
        bool(helper_src)
        and "self.__class__.__name__" not in helper_src
        and helper_src.count("source=_clf_by_roi_helper.__name__") >= 2,
    )

    # --- PACKAGE-WIDE drift guard: zero BoolOp except handlers anywhere ---
    pkg = REPO_ROOT / "mufasa"
    boolop_sites = []
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            s = f.read_text(encoding="utf-8")
            for ln in _boolop_except_sites(s):
                boolop_sites.append(f"{f.relative_to(REPO_ROOT)}:{ln}")
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        "no `except A or B:` antipattern anywhere in mufasa/",
        not boolop_sites,
        detail=", ".join(boolop_sites),
    )
    check(
        f"all mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fp_audit_runtime_fixes: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
