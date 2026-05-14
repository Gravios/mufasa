"""
tests/smoke_122ae6_csv_export.py
=================================

Patch 122ae-6: CSV export backend + Tools-page form.

Behavioural verification of the three backend functions in
:mod:`mufasa.utils.csv_export`:

* :func:`export_features_csv` — reads via load_features_for_video
  (so per-family + wide + legacy CSV all work transparently),
  writes one CSV at <dest>/<video>.csv, returns the path.
* :func:`export_labels_csv` — same shape for classifier labels.
* :func:`export_combined_csv` — features and labels concatenated
  column-wise into the legacy targets_inserted shape; raises
  ValueError on row-count mismatch rather than silently
  mangling shapes.

Also AST-verifies the Qt form
(:class:`ExportToCSVForm`):

* exists with the OperationForm contract (build, collect_args,
  target);
* references the three backend functions;
* dispatch table maps each "What to export" choice to the right
  backend;
* form is wired into the Tools page (tools_page.py imports it
  and adds a section).

And confirms the patch records the 122ae-6 number in both the
backend module and the form.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _make_v1_project(tmp: Path,
                     classifier_targets: list[str] | None = None,
                     ) -> Path:
    toml = tmp / "project.toml"
    targets = classifier_targets or []
    targets_str = ", ".join(f'"{t}"' for t in targets)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae6"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose", "tail"]

        [classifiers]
        targets = [{targets_str}]
    """).strip() + "\n")
    return toml


def _seed_v1_features(tmp: Path, video: str,
                      df: pd.DataFrame) -> None:
    """Write a wide-parquet sidecar to the v1 derived/features/
    location, the same way write_wide_features_v1 would."""
    feat_dir = tmp / "derived" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / f"{video}.parquet", index=False)


def _seed_v1_labels(tmp: Path, video: str,
                    df: pd.DataFrame) -> None:
    """Write a per-video labels parquet at derived/labels/."""
    lab_dir = tmp / "derived" / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(lab_dir / f"{video}.parquet", index=False)


def main() -> int:
    from mufasa.utils.csv_export import (export_combined_csv,
                                         export_features_csv,
                                         export_labels_csv)

    # ==================================================================
    # 1. export_features_csv — v1 wide parquet → CSV
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        _seed_v1_features(tmp, "v_001", pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [10.0, 20.0, 30.0],
        }))
        dest = tmp / "exports"
        path = export_features_csv("v_001", str(toml), str(dest))
        check(
            "features: returns destination path",
            Path(path) == dest / "v_001.csv",
        )
        check(
            "features: CSV file exists on disk",
            Path(path).is_file(),
        )
        # Default include_index=True → leading pad column present.
        # Read raw (no strip) to inspect.
        raw = pd.read_csv(path)
        check(
            "features: include_index=True writes pad column "
            "(SimBA read_df compat)",
            raw.shape[1] == 3
            and {"feat_a", "feat_b"}.issubset(set(raw.columns)),
            detail=f"got columns {list(raw.columns)}",
        )
        # Round-trip the pad strip and check values
        from mufasa.utils.feature_io import _read_legacy
        legacy = _read_legacy(path)
        check(
            "features: round-trip through _read_legacy "
            "(SimBA strip) recovers exact values",
            legacy["feat_a"].tolist() == [1.0, 2.0, 3.0]
            and legacy["feat_b"].tolist() == [10.0, 20.0, 30.0],
        )

    # ==================================================================
    # 1b. export_features_csv with include_index=False
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        _seed_v1_features(tmp, "v_002", pd.DataFrame({
            "feat_a": [1, 2],
        }))
        dest = tmp / "exports"
        path = export_features_csv(
            "v_002", str(toml), str(dest), include_index=False,
        )
        raw = pd.read_csv(path)
        check(
            "features: include_index=False writes no pad column",
            list(raw.columns) == ["feat_a"],
        )

    # ==================================================================
    # 1c. Auto-mkdir of dest_dir
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        _seed_v1_features(tmp, "v_003",
                          pd.DataFrame({"a": [1]}))
        deeply_nested = tmp / "a" / "b" / "c"  # doesn't exist
        path = export_features_csv(
            "v_003", str(toml), str(deeply_nested),
        )
        check(
            "features: deeply-nested dest_dir auto-created",
            Path(path).is_file()
            and deeply_nested.is_dir(),
        )

    # ==================================================================
    # 1d. Missing features → FileNotFoundError propagates
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        raised = False
        try:
            export_features_csv(
                "v_missing", str(toml), str(tmp / "exports"),
            )
        except FileNotFoundError:
            raised = True
        check(
            "features: missing video → FileNotFoundError "
            "propagated from load_features_for_video",
            raised,
        )

    # ==================================================================
    # 2. export_labels_csv
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
        )
        _seed_v1_labels(tmp, "v_004",
            pd.DataFrame({
                "sniff": pd.array([0, 1, 1, 0], dtype="Int64"),
                "rear":  pd.array([0, 0, 1, 1], dtype="Int64"),
            }))
        dest = tmp / "exports"
        path = export_labels_csv("v_004", str(toml), str(dest))
        check(
            "labels: CSV written at expected path",
            Path(path) == dest / "v_004.csv"
            and Path(path).is_file(),
        )
        raw = pd.read_csv(path)
        check(
            "labels: classifier-target columns present",
            {"sniff", "rear"}.issubset(set(raw.columns)),
        )

    # ==================================================================
    # 3. export_combined_csv
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff"],
        )
        _seed_v1_features(tmp, "v_005",
            pd.DataFrame({"feat": [10.0, 20.0, 30.0]}))
        _seed_v1_labels(tmp, "v_005",
            pd.DataFrame({
                "sniff": pd.array([0, 1, 0], dtype="Int64"),
            }))
        dest = tmp / "exports"
        path = export_combined_csv("v_005", str(toml), str(dest))
        check(
            "combined: CSV written",
            Path(path).is_file(),
        )
        # Read raw to count columns (1 pad + 1 feat + 1 sniff = 3)
        raw = pd.read_csv(path)
        check(
            "combined: feature + label columns concatenated",
            "feat" in raw.columns and "sniff" in raw.columns,
        )
        # Strip pad column for value verification
        from mufasa.utils.feature_io import _read_legacy
        stripped = _read_legacy(path)
        check(
            "combined: feature values preserved",
            stripped["feat"].tolist() == [10.0, 20.0, 30.0],
        )
        check(
            "combined: label values preserved (pandas may "
            "read as int64 since no NA present after CSV "
            "round-trip)",
            stripped["sniff"].tolist() == [0, 1, 0],
        )

    # 3b — row-count mismatch raises ValueError
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff"],
        )
        # 4 rows of features, 3 rows of labels
        _seed_v1_features(tmp, "v_006",
            pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0]}))
        _seed_v1_labels(tmp, "v_006",
            pd.DataFrame({
                "sniff": pd.array([0, 1, 0], dtype="Int64"),
            }))
        raised = False
        msg = ""
        try:
            export_combined_csv("v_006", str(toml),
                                str(tmp / "exports"))
        except ValueError as exc:
            raised = True
            msg = str(exc)
        check(
            "combined: row-count mismatch raises ValueError",
            raised,
        )
        check(
            "combined: error message names the row counts so "
            "users can diagnose",
            "4" in msg and "3" in msg and "v_006" in msg,
            detail=f"got {msg!r}",
        )

    # ==================================================================
    # 4. Backend AST — public API + 122ae-6 note
    # ==================================================================
    backend_src = (REPO_ROOT / "mufasa" / "utils"
                   / "csv_export.py").read_text()
    backend_tree = ast.parse(backend_src)
    top_fn_names = [
        n.name for n in backend_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for fn in ("export_features_csv", "export_labels_csv",
               "export_combined_csv"):
        check(
            f"csv_export module exports {fn}",
            fn in top_fn_names,
        )
    check(
        "csv_export imports load_features_for_video",
        "load_features_for_video" in backend_src,
    )
    check(
        "csv_export imports load_labels_for_video",
        "load_labels_for_video" in backend_src,
    )
    check(
        "csv_export records 122ae-6 in docstring/comments",
        "122ae-6" in backend_src,
    )

    # ==================================================================
    # 5. Form AST — contract + backend dispatch + Tools-page wire-up
    # ==================================================================
    form_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "data_export.py").read_text()
    form_tree = ast.parse(form_src)
    form_cls = next(
        (n for n in ast.walk(form_tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "ExportToCSVForm"),
        None,
    )
    check("ExportToCSVForm class defined", form_cls is not None)
    if form_cls is not None:
        method_names = {
            m.name for m in form_cls.body
            if isinstance(m, ast.FunctionDef)
        }
        for required in ("build", "collect_args", "target"):
            check(
                f"ExportToCSVForm implements {required}() "
                "(OperationForm contract)",
                required in method_names,
            )
    # Backend dispatch — form references all three exporters
    for backend in ("export_features_csv", "export_labels_csv",
                    "export_combined_csv"):
        check(
            f"ExportToCSVForm references {backend}",
            backend in form_src,
        )
    check(
        "ExportToCSVForm records 122ae-6 in code/comments",
        "122ae-6" in form_src,
    )

    # Tools-page wire-up
    tp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
              / "tools_page.py").read_text()
    check(
        "tools_page imports ExportToCSVForm",
        "from mufasa.ui_qt.forms.data_export import "
        "ExportToCSVForm" in tp_src,
    )
    check(
        "tools_page adds Export-to-CSV section",
        "Export to CSV" in tp_src
        and "ExportToCSVForm" in tp_src,
    )
    check(
        "tools_page records 122ae-6",
        "122ae-6" in tp_src,
    )

    print(
        f"smoke_122ae6_csv_export: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
