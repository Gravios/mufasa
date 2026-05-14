"""
tests/smoke_122an_dead_code_removal.py
=======================================

Patch 122an: csv-subtree audit Category B — dead-code removal.

Closes B1 + B2 from the csv_subtree_removal_audit:

* **B1** — :class:`FeatureSubsetsCalculator`'s
  ``append_to_features_extracted`` / ``append_to_targets_inserted``
  kwargs (and the matching UI fields in ``features.py`` +
  ``subset_feature_extractor_pop_up.py``) are removed. The
  pre-flight checks they gated never triggered an actual append
  write — the writer was retargeted to per-family parquet in
  122ae-3 and these flags became inert. Removing them clears
  ~25 lines of misleading API surface.

* **B2** — ``select_video_for_labelling_popup.py`` "continue
  mode" existence check no longer probes
  ``os.path.isfile(targets_inserted_file_path)`` (which returned
  False on v1-only projects, incorrectly blocking continue
  mode). Instead it attempts ``load_labels_for_video`` and
  raises only if FileNotFoundError surfaces — handles both v1
  projects and any legacy-fallback edge cases the helper
  supports.

Coverage — AST only (heavy Qt / Tk / cv2 / sklearn deps not
available in the sandbox).
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


def _signature_kwargs(src: str, func_name: str) -> set[str]:
    """Return the set of kwarg names declared on the first
    matching function/method."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != func_name:
                continue
            names = set()
            args = node.args
            for a in args.args + args.kwonlyargs:
                names.add(a.arg)
            return names
    return set()


def main() -> int:
    # ==================================================================
    # B1 — FeatureSubsetsCalculator kwargs removed
    # ==================================================================
    fs_path = (REPO_ROOT / "mufasa" / "feature_extractors"
               / "feature_subsets.py")
    fs_src = fs_path.read_text()

    init_kwargs = _signature_kwargs(fs_src, "__init__")
    check(
        "B1 (feature_subsets): append_to_features_extracted "
        "kwarg removed from FeatureSubsetsCalculator.__init__",
        "append_to_features_extracted" not in init_kwargs,
    )
    check(
        "B1 (feature_subsets): append_to_targets_inserted "
        "kwarg removed from FeatureSubsetsCalculator.__init__",
        "append_to_targets_inserted" not in init_kwargs,
    )
    check(
        "B1 (feature_subsets): no self.append_to_* attribute "
        "assignments left at code level",
        not any(
            ("self.append_to_features_extracted" in line
             or "self.append_to_targets_inserted" in line)
            and not line.lstrip().startswith("#")
            for line in fs_src.splitlines()
        ),
    )
    check(
        "B1 (feature_subsets): pre-flight 'append against "
        "features_dir / targets_folder' branches removed",
        not any(
            "check_same_files_exist_in_all_directories" in line
            and ("features_dir" in line or "targets_folder" in line)
            and not line.lstrip().startswith("#")
            for line in fs_src.splitlines()
        ),
    )
    check(
        "B1 (feature_subsets): docstring :param: lines for the "
        "two kwargs are dropped",
        not any(
            ":param bool append_to_features_extracted" in line
            for line in fs_src.splitlines()
        )
        and not any(
            ":param bool append_to_targets_inserted" in line
            for line in fs_src.splitlines()
        ),
    )
    check(
        "B1 (feature_subsets): >>> example no longer references "
        "the dead kwargs",
        ">>>" not in fs_src
        or all(
            "append_to_features_extracted" not in line
            and "append_to_targets_inserted" not in line
            for line in fs_src.splitlines()
            if line.lstrip().startswith(">>>")
        ),
    )
    check(
        "B1 (feature_subsets): records 122an patch number",
        "122an" in fs_src,
    )

    # ==================================================================
    # B1 — Qt features.py form fields removed
    # ==================================================================
    qt_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "features.py")
    qt_src = qt_path.read_text()
    check(
        "B1 (features.py): dest_append_features QRadioButton "
        "removed",
        not any(
            "self.dest_append_features = QRadioButton" in line
            and not line.lstrip().startswith("#")
            for line in qt_src.splitlines()
        ),
    )
    check(
        "B1 (features.py): dest_append_targets QRadioButton "
        "removed",
        not any(
            "self.dest_append_targets = QRadioButton" in line
            and not line.lstrip().startswith("#")
            for line in qt_src.splitlines()
        ),
    )
    check(
        "B1 (features.py): collect_args no longer returns "
        "append_features / append_targets keys",
        '"append_features"' not in qt_src
        and '"append_targets"' not in qt_src,
    )
    check(
        "B1 (features.py): FeatureSubsetsCalculator call sites "
        "no longer pass the dead kwargs",
        "append_to_features_extracted=" not in qt_src
        and "append_to_targets_inserted=" not in qt_src,
    )
    check(
        "B1 (features.py): target() signature no longer accepts "
        "append_features / append_targets",
        "append_features:" not in qt_src
        and "append_targets:" not in qt_src,
    )
    check(
        "B1 (features.py): records 122an patch number",
        "122an" in qt_src,
    )

    # ==================================================================
    # B1 — Tk popup checkboxes removed
    # ==================================================================
    tk_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "subset_feature_extractor_pop_up.py")
    tk_src = tk_path.read_text()
    check(
        "B1 (Tk popup): append_to_features_var checkbox removed",
        "append_to_features_var" not in tk_src,
    )
    check(
        "B1 (Tk popup): append_to_targets_var checkbox removed",
        "append_to_targets_var" not in tk_src,
    )
    check(
        "B1 (Tk popup): FeatureSubsetsCalculator call no longer "
        "passes append_to_* kwargs",
        "append_to_features_extracted=" not in tk_src
        and "append_to_targets_inserted=" not in tk_src,
    )
    check(
        "B1 (Tk popup): records 122an patch number",
        "122an" in tk_src,
    )

    # ==================================================================
    # B2 — select_video_for_labelling_popup existence check
    # ==================================================================
    sv_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "select_video_for_labelling_popup.py")
    sv_src = sv_path.read_text()
    check(
        "B2: continue-mode check no longer uses "
        "os.path.isfile(self.targets_inserted_file_path)",
        not any(
            "os.path.isfile(self.targets_inserted_file_path)"
            in line
            and not line.lstrip().startswith("#")
            for line in sv_src.splitlines()
        ),
    )
    check(
        "B2: probes v1 layout via load_labels_for_video",
        "load_labels_for_video" in sv_src,
    )
    check(
        "B2: catches FileNotFoundError from v1 probe",
        "FileNotFoundError" in sv_src,
    )
    check(
        "B2: no longer constructs targets_inserted_dir at code "
        "level (legacy path no longer used)",
        not any(
            "targets_inserted_dir = " in line
            and not line.lstrip().startswith("#")
            for line in sv_src.splitlines()
        ),
    )
    check(
        "B2: records 122an patch number",
        "122an" in sv_src,
    )

    # ==================================================================
    # Sanity — all touched files parse
    # ==================================================================
    for p in (fs_path, qt_path, tk_path, sv_path):
        try:
            ast.parse(p.read_text())
            ok = True
        except SyntaxError:
            ok = False
        check(f"AST parses: {p.name}", ok)

    print(
        f"smoke_122an_dead_code_removal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
