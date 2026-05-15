"""
tests/smoke_122ay_mp_worker_migration.py
=========================================

Patch 122ay: migrate the 8 multiprocessing-variant consumers
that were deferred from 122au/av to v1-aware reads.

Without this patch, post-122ax v1-only projects would fail any
MP-based analysis or visualization flow — the legacy CSVs no
longer exist, so the worker's :func:`read_df(file_path, ...)`
calls would raise. This patch closes that regression.

Two distinct migration shapes
-----------------------------
The 8 _mp files split into two patterns:

* **Class-level read (6 files)** — the parallel work is
  downstream (frame-level rendering / chunk processing); the
  read of machine_results happens once per video on the main
  thread, same as the single-core sibling. Migrated identically
  to 122av:
    - plotting/clf_validator_mp.py
    - plotting/gantt_creator_mp.py
    - plotting/heat_mapper_clf_mp.py
    - plotting/path_plotter_mp.py
    - plotting/plot_clf_results_mp.py
      (.reset_index(drop=True).fillna(0) chain preserved)
    - plotting/probability_plot_creator_mp.py

* **Worker-internal read (2 files)** — the read happens inside
  the pickled worker function for video-level parallelism.
  Pattern: thread ``config_path`` into the worker via a new
  keyword argument with a None default (back-compat), then
  pass it through ``functools.partial`` in the dispatcher:
    - data_processors/agg_clf_counter_mp.py
    - roi_tools/roi_clf_calculator_mp.py

  The None default makes the worker fall back to the legacy
  read path so external callers that build their own ``partial``
  without setting config_path don't break.

Coverage
--------
1. Each of the 8 migrated MP files imports the v1 helper at
   code level.
2. Each records 122ay.
3. No remaining raw read_df calls in the class-level read
   sites for the 6 viz _mp files.
4. The 2 worker-internal files thread config_path through
   _func.partial as a keyword argument.
5. Both worker functions accept config_path as a kwarg with
   None default (back-compat shape).
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


CLASS_LEVEL_FILES = [
    REPO_ROOT / "mufasa" / "plotting" / "clf_validator_mp.py",
    REPO_ROOT / "mufasa" / "plotting" / "gantt_creator_mp.py",
    REPO_ROOT / "mufasa" / "plotting" / "heat_mapper_clf_mp.py",
    REPO_ROOT / "mufasa" / "plotting" / "path_plotter_mp.py",
    REPO_ROOT / "mufasa" / "plotting" / "plot_clf_results_mp.py",
    REPO_ROOT / "mufasa" / "plotting"
    / "probability_plot_creator_mp.py",
]

WORKER_INTERNAL_FILES = [
    REPO_ROOT / "mufasa" / "data_processors"
    / "agg_clf_counter_mp.py",
    REPO_ROOT / "mufasa" / "roi_tools" / "roi_clf_calculator_mp.py",
]


def _has_config_path_default_None(src: str,
                                  func_name: str) -> bool:
    """Walk the AST and check whether <func_name> has a
    ``config_path`` parameter with default value None."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != func_name:
            continue
        all_args = list(node.args.args) + list(node.args.kwonlyargs)
        for i, a in enumerate(all_args):
            if a.arg == "config_path":
                # Walk defaults — defaults align with args from the
                # right; kw_defaults align positionally with kwonlyargs.
                if a in node.args.kwonlyargs:
                    kw_idx = node.args.kwonlyargs.index(a)
                    default = node.args.kw_defaults[kw_idx]
                else:
                    pos_idx = node.args.args.index(a)
                    n_defaults = len(node.args.defaults)
                    n_args = len(node.args.args)
                    rev_idx = n_args - pos_idx - 1
                    if rev_idx >= n_defaults:
                        return False
                    default = node.args.defaults[
                        n_defaults - 1 - rev_idx
                    ]
                if isinstance(default, ast.Constant) and default.value is None:
                    return True
                # Older py: Name(id='None')
                if isinstance(default, ast.Name) and default.id == "None":
                    return True
                return False
    return False


def main() -> int:
    # ==================================================================
    # 1. All 8 files import the helper + record 122ay
    # ==================================================================
    for path in CLASS_LEVEL_FILES + WORKER_INTERNAL_FILES:
        check(
            f"{path.name}: exists",
            path.is_file(),
        )
        if not path.is_file():
            continue
        src = path.read_text()
        check(
            f"{path.name}: imports load_machine_results_for_video",
            "load_machine_results_for_video" in src,
        )
        check(
            f"{path.name}: records 122ay",
            "122ay" in src,
        )

    # ==================================================================
    # 2. Class-level reads — no remaining raw read_df calls for
    #    the file_path / data_path / self.data_path read sites.
    # ==================================================================
    for path in CLASS_LEVEL_FILES:
        if not path.is_file():
            continue
        src = path.read_text()
        offending = []
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            compact = stripped.replace(" ", "")
            if (
                "read_df(file_path," in compact
                or "read_df(file_path=file_path" in stripped
                or "read_df(data_path," in compact
                or "read_df(self.data_path" in stripped
                or "read_df(file_path,self." in compact
            ):
                offending.append((lineno, line.strip()))
        check(
            f"{path.name}: no remaining raw read_df calls for "
            "the class-level read site",
            len(offending) == 0,
            detail=(
                "; ".join(f"L{n}: {l}" for n, l in offending)
                if offending else ""
            ),
        )

    # ==================================================================
    # 3. plot_clf_results_mp preserves .reset_index(drop=True).
    #    fillna(0) chain
    # ==================================================================
    pcr_mp_src = (REPO_ROOT / "mufasa" / "plotting"
                  / "plot_clf_results_mp.py").read_text()
    check(
        "plot_clf_results_mp preserves "
        ".reset_index(drop=True).fillna(0) chain",
        ".reset_index(drop=True).fillna(0)" in pcr_mp_src
        and "load_machine_results_for_video" in pcr_mp_src,
    )

    # ==================================================================
    # 4. Worker-internal: workers accept config_path=None and
    #    dispatchers pass it through partial.
    # ==================================================================
    agg_src = (REPO_ROOT / "mufasa" / "data_processors"
               / "agg_clf_counter_mp.py").read_text()
    check(
        "agg_clf_counter_mp: worker _agg_clf_helper has "
        "config_path: str = None",
        _has_config_path_default_None(agg_src, "_agg_clf_helper"),
    )
    check(
        "agg_clf_counter_mp: partial passes config_path=self.config_path",
        "config_path=self.config_path" in agg_src,
    )

    roi_src = (REPO_ROOT / "mufasa" / "roi_tools"
               / "roi_clf_calculator_mp.py").read_text()
    check(
        "roi_clf_calculator_mp: worker _clf_by_roi_helper has "
        "config_path: str = None",
        _has_config_path_default_None(roi_src, "_clf_by_roi_helper"),
    )
    check(
        "roi_clf_calculator_mp: partial passes config_path=self.config_path",
        "config_path=self.config_path" in roi_src,
    )

    # ==================================================================
    # 5. Workers gate the v1 read on config_path is not None
    #    (graceful fallback to legacy read_df when caller didn't set
    #    it — back-compat for external callers building their own
    #    partial)
    # ==================================================================
    check(
        "agg_clf_counter_mp: worker gates v1 read on "
        "`if config_path is not None`",
        "if config_path is not None" in agg_src
        and "load_machine_results_for_video" in agg_src,
    )
    check(
        "roi_clf_calculator_mp: worker gates v1 read on "
        "`if config_path is not None`",
        "if config_path is not None" in roi_src
        and "load_machine_results_for_video" in roi_src,
    )

    print(
        f"smoke_122ay_mp_worker_migration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
