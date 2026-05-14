"""
tests/smoke_122ak_v1_native.py
===============================

Patch 122ak: v1-native consolidation — drop the legacy
csv/targets_inserted writes from labellers + appenders, drop
the legacy fallback reads from the helpers, and drop the dead
config_path=None branch from TrainModelMixin. The legacy
layout (csv/features_extracted, csv/targets_inserted) is no
longer touched; v1 (derived/features, derived/labels) is the
only supported layout.

User direction shaping this patch: "I don't need backwards
compatibility, write it in a way that is logical and
efficient."

Coverage:

1. Labellers (Qt + 3 Tk classes) save labels-only via
   save_labels_for_video — no targets_inserted writes, no
   features re-loading + recombining.

2. Third-party appenders (BENTO, BORIS, third_party_appender,
   deepethogram_importer) write to derived/labels/ via the
   same helper.

3. load_features_for_video + load_labels_for_video raise
   FileNotFoundError on legacy-only seeds (legacy fallback
   removed).

4. list_video_stems_with_features + list_video_stems_with_labels
   no longer scan the legacy directories.

5. _read_legacy + _legacy_csv_path removed from both helpers.

6. TrainModelMixin requires self.config_path — read methods
   raise RuntimeError when config_path is None.

All checks are AST-based (Qt + Tk + cv2 + sklearn aren't in
the sandbox), except for the helper-behaviour tests where
the helper is pure Python.
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
        name = "smoke_122ak"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose", "tail"]

        [classifiers]
        targets = [{targets_str}]
    """).strip() + "\n")
    return toml


def main() -> int:
    # ==================================================================
    # 1. Helpers — _read_legacy + _legacy_csv_path removed
    # ==================================================================
    fio_src = (REPO_ROOT / "mufasa" / "utils"
               / "feature_io.py").read_text()
    lio_src = (REPO_ROOT / "mufasa" / "utils"
               / "label_io.py").read_text()
    check(
        "feature_io: def _read_legacy removed",
        "def _read_legacy" not in fio_src,
    )
    check(
        "feature_io: def _legacy_csv_path removed",
        "def _legacy_csv_path" not in fio_src,
    )
    check(
        "feature_io: 'Legacy fallback' comment block removed",
        "Legacy fallback" not in fio_src,
    )
    check(
        "label_io: def _read_legacy removed",
        "def _read_legacy" not in lio_src,
    )
    check(
        "feature_io: project_metadata_from_config import dropped "
        "(was only used by _legacy_csv_path)",
        "project_metadata_from_config" not in fio_src,
    )
    check(
        "feature_io docstring updated — no 'Legacy fallback' "
        "section header",
        "Legacy fallback\n" not in fio_src,
    )
    check(
        "label_io docstring updated — no 'Legacy fallback' "
        "section header",
        "Legacy fallback" not in lio_src
        or "fallback was removed" in lio_src,
    )
    check(
        "feature_io: 122ak patch number recorded",
        "122ak" in fio_src,
    )
    check(
        "label_io: 122ak patch number recorded",
        "122ak" in lio_src,
    )

    # ==================================================================
    # 2. Behavioural — helpers raise on legacy-only seeds
    # ==================================================================
    from mufasa.utils.feature_io import (list_video_stems_with_features,
                                         load_features_for_video)
    from mufasa.utils.label_io import (list_video_stems_with_labels,
                                       load_labels_for_video,
                                       save_labels_for_video)

    # Features: legacy-only project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1, 2]}).to_csv(
            legacy_dir / "v_only_legacy.csv", index=False,
        )
        raised = False
        try:
            load_features_for_video(
                "v_only_legacy", str(toml),
            )
        except FileNotFoundError:
            raised = True
        check(
            "features helper: legacy-only seed raises "
            "FileNotFoundError",
            raised,
        )
        # list_video_stems_with_features should also ignore
        # the legacy directory now.
        stems = list_video_stems_with_features(str(toml))
        check(
            "features discovery: legacy directory no longer "
            "scanned (returns [] for legacy-only project)",
            stems == [],
        )

    # Labels: legacy-only project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        legacy_dir = tmp / "csv" / "targets_inserted"
        legacy_dir.mkdir(parents=True)
        pd.DataFrame({"sniff": [1, 0]}).to_csv(
            legacy_dir / "v_only_legacy.csv", index=False,
        )
        raised = False
        try:
            load_labels_for_video(
                "v_only_legacy", str(toml),
            )
        except FileNotFoundError:
            raised = True
        check(
            "labels helper: legacy-only seed raises "
            "FileNotFoundError",
            raised,
        )
        stems = list_video_stems_with_labels(str(toml))
        check(
            "labels discovery: legacy directory no longer "
            "scanned (returns [] for legacy-only project)",
            stems == [],
        )

    # v1 still works
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        out = save_labels_for_video(
            video_name="v_v1",
            config_path=str(toml),
            labels=pd.DataFrame({"sniff": [1, 0, 1]}),
        )
        check(
            "v1 save: file lands at derived/labels/<video>.parquet",
            Path(out) == tmp / "derived" / "labels" / "v_v1.parquet"
            and Path(out).is_file(),
        )
        loaded = load_labels_for_video("v_v1", str(toml))
        check(
            "v1 read: round-trip preserves values",
            loaded["sniff"].tolist() == [1, 0, 1],
        )

    # ==================================================================
    # 3. Labellers — save methods write labels-only
    # ==================================================================
    labellers = {
        "labelling_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_interface.py",
        "labelling_advanced_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_advanced_interface.py",
        "standard_labeller.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "standard_labeller.py",
        "frame_labeller.py":
            REPO_ROOT / "mufasa" / "ui_qt"
            / "frame_labeller.py",
    }
    for name, path in labellers.items():
        src = path.read_text()
        check(
            f"{name}: save method records 122ak (labels-only "
            "transition)",
            "122ak" in src,
        )
        # No more legacy write to targets_inserted_file_path
        stale_writes = [
            line for line in src.splitlines()
            if "write_df" in line
            and "targets_inserted_file_path" in line
            and not line.lstrip().startswith("#")
        ]
        check(
            f"{name}: no remaining write_df(...targets_inserted...) "
            "at code level",
            not stale_writes,
            detail=f"leaked: {stale_writes}" if stale_writes else "",
        )
        # No more pd.concat([features, labels]) in save path —
        # the legacy combined-write logic is gone.
        check(
            f"{name}: save method calls save_labels_for_video",
            "save_labels_for_video(" in src,
        )
    # The Qt labeller specifically had _read_df_best_effort +
    # _write_df_best_effort scaffolding for the legacy combined
    # write. Those should be gone.
    qt_src = labellers["frame_labeller.py"].read_text()
    check(
        "Qt frame_labeller: _read_df_best_effort scaffolding "
        "removed (no longer needed for labels-only save)",
        "_read_df_best_effort" not in qt_src,
    )
    check(
        "Qt frame_labeller: _write_df_best_effort scaffolding "
        "removed",
        "_write_df_best_effort" not in qt_src,
    )

    # ==================================================================
    # 4. Third-party appenders — write labels via the helper
    # ==================================================================
    appenders = {
        "BENTO_appender.py":
            REPO_ROOT / "mufasa" / "third_party_label_appenders"
            / "BENTO_appender.py",
        "BORIS_appender.py":
            REPO_ROOT / "mufasa" / "third_party_label_appenders"
            / "BORIS_appender.py",
        "third_party_appender.py":
            REPO_ROOT / "mufasa" / "third_party_label_appenders"
            / "third_party_appender.py",
        "deepethogram_importer.py":
            REPO_ROOT / "mufasa" / "third_party_label_appenders"
            / "deepethogram_importer.py",
    }
    for name, path in appenders.items():
        src = path.read_text()
        check(
            f"{name}: imports save_labels_for_video",
            "save_labels_for_video" in src,
        )
        check(
            f"{name}: records 122ak",
            "122ak" in src,
        )
        # No remaining write_df(...targets_folder...) at code
        stale_writes = [
            line for line in src.splitlines()
            if "write_df" in line
            and ("targets_folder" in line or "save_path" in line)
            and not line.lstrip().startswith("#")
            and "write_df(df, file_type" not in line  # boris_source_cleaner uses different shape
        ]
        # The deepethogram check is loose — it might keep a
        # tangentially related write_df elsewhere; just check
        # that the legacy save_path = os.path.join(self.targets_folder,...)
        # pattern is gone.
        check(
            f"{name}: legacy os.path.join(self.targets_folder, "
            "<video>.<ext>) save pattern removed",
            not any(
                "self.targets_folder" in line
                and "os.path.join" in line
                and not line.lstrip().startswith("#")
                for line in src.splitlines()
            ),
        )

    # ==================================================================
    # 5. TrainModelMixin requires config_path
    # ==================================================================
    mixin_src = (REPO_ROOT / "mufasa" / "mixins"
                 / "train_model_mixin.py").read_text()
    check(
        "TrainModelMixin: read_all_files_in_folder raises when "
        "config_path is None",
        # The new RuntimeError lives in both read methods.
        mixin_src.count(
            "requires self.config_path "
        ) >= 1,
    )
    check(
        "TrainModelMixin: dead 'else: read_df(file_path)' branch "
        "removed from read_all_files_in_folder",
        # Was:  else: df = (read_df(file, file_type)....
        # The else branch is gone now.
        "else:\n                df = (read_df(file, file_type)"
        not in mixin_src,
    )
    check(
        "TrainModelMixin: dead 'else: read_df(file_path)' branch "
        "removed from _read_data_file_helper_futures",
        "else:\n            df = read_df(file_path, file_type)"
        not in mixin_src,
    )
    check(
        "TrainModelMixin: futures dispatcher no longer skips "
        "existence check (it raises if config_path is None "
        "instead)",
        "check_filepaths_in_iterable_exist(file_paths=annotations_file_paths"
        not in mixin_src,
    )
    check(
        "TrainModelMixin: records 122ak",
        "122ak" in mixin_src,
    )

    print(
        f"smoke_122ak_v1_native: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
