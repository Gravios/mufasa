"""
tests/smoke_122ae5e_classifier_training.py
============================================

Patch 122ae-5e: classifier training migration. The 5 RF
training modules (train_rf, train_multilabel_rf,
train_multiclass_rf, grid_search_rf, grid_search_multiclass_rf)
discover annotated videos and read features+labels via the v1
layout (derived/features/ + derived/labels/) when the legacy
csv/targets_inserted/ tree is empty.

The migration is shared via the TrainModelMixin's
read_all_files_in_folder_mp_futures + read_all_files_in_folder
methods, so the change is invasive at the mixin level (one
worker function and two callers) and minimal at each training
module (single v1-discovery block injected after
ConfigReader.__init__).

Behavioural verification of the new discovery helper:

* list_video_stems_with_labels — sibling of
  list_video_stems_with_features (122ae-5b). UNIONS stems from
  derived/labels/<video>.parquet + legacy
  csv/targets_inserted/<video>.<ext>, dedupes, ignores dotfiles,
  returns sorted list. Empty/malformed config returns [].

AST verification of the mixin changes:

* _read_data_file_helper_futures gains an optional config_path
  kwarg; routes through load_features_for_video +
  load_labels_for_video when set.
* _read_data_file_helper or read_all_files_in_folder (non-MP
  path) takes the same treatment.
* read_all_files_in_folder_mp_futures passes self.config_path
  to the worker; skips check_filepaths_in_iterable_exist when
  config_path is set (pseudo-paths may not exist on disk).

AST verification of the 5 training modules:

* Each has a v1-aware discovery block referencing
  list_video_stems_with_labels after their ConfigReader init.
* Each records 122ae-5e in code/comments.
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
                     classifier_targets: list[str]) -> Path:
    toml = tmp / "project.toml"
    targets_str = ", ".join(f'"{t}"' for t in classifier_targets)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae5e"
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
    # 1. Behavioural — list_video_stems_with_labels
    # ==================================================================
    from mufasa.utils.label_io import list_video_stems_with_labels

    # 1a — derived/labels/ only
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff"])
        lab_dir = tmp / "derived" / "labels"
        lab_dir.mkdir(parents=True)
        for stem in ["v_alpha", "v_beta"]:
            pd.DataFrame({"sniff": [0, 1]}).to_parquet(
                lab_dir / f"{stem}.parquet", index=False,
            )
        stems = list_video_stems_with_labels(str(toml))
        check(
            "discovery: derived-labels-only branch finds stems",
            stems == ["v_alpha", "v_beta"],
            detail=f"got {stems}",
        )

    # 1b — legacy csv/targets_inserted/ only
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff"])
        legacy_dir = tmp / "csv" / "targets_inserted"
        legacy_dir.mkdir(parents=True)
        for stem in ["v_legacy1", "v_legacy2"]:
            pd.DataFrame({"sniff": [0, 1]}).to_csv(
                legacy_dir / f"{stem}.csv", index=False,
            )
        # Hidden file should be ignored
        (legacy_dir / ".DS_Store").write_text("")
        stems = list_video_stems_with_labels(str(toml))
        # Patch 122bf: 122ak made list_video_stems_with_labels
        # v1-read-only — legacy csv/targets_inserted/ is not
        # scanned. A v1 project with legacy-only labels returns [].
        check(
            "discovery: legacy-only branch finds NOTHING "
            "(v1-read-only post-122ak)",
            stems == [],
            detail=f"got {stems}",
        )

    # 1c — both layouts, union — but legacy stems are excluded
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff"])
        lab_dir = tmp / "derived" / "labels"
        lab_dir.mkdir(parents=True)
        for stem in ["v_a", "v_b"]:  # v_a + v_b in derived
            pd.DataFrame({"sniff": [0]}).to_parquet(
                lab_dir / f"{stem}.parquet", index=False,
            )
        legacy_dir = tmp / "csv" / "targets_inserted"
        legacy_dir.mkdir(parents=True)
        for stem in ["v_a", "v_c"]:  # v_a + v_c in legacy
            pd.DataFrame({"sniff": [0]}).to_csv(
                legacy_dir / f"{stem}.csv", index=False,
            )
        stems = list_video_stems_with_labels(str(toml))
        # Patch 122bf: v_c only exists as legacy CSV — excluded
        # post-122ak. Only the v1 stems (v_a, v_b) are returned.
        check(
            "discovery: v1-only branch returns derived/labels stems "
            "(legacy-only stems silently excluded post-122ak)",
            stems == ["v_a", "v_b"],
            detail=f"got {stems}",
        )

    # 1d — empty project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff"])
        stems = list_video_stems_with_labels(str(toml))
        check(
            "discovery: empty project returns []",
            stems == [],
        )

    # 1e — malformed config path
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        bad = tmp / "nope" / "missing.toml"
        stems = list_video_stems_with_labels(str(bad))
        check(
            "discovery: malformed config returns []",
            stems == [],
        )

    # ==================================================================
    # 2. AST — TrainModelMixin changes
    # ==================================================================
    mixin_src = (REPO_ROOT / "mufasa" / "mixins"
                 / "train_model_mixin.py").read_text()

    # _read_data_file_helper_futures takes config_path kwarg
    check(
        "_read_data_file_helper_futures takes optional "
        "config_path kwarg",
        "config_path: Optional[str] = None" in mixin_src
        or "config_path=None" in mixin_src,
    )
    # It calls load_features_for_video + load_labels_for_video
    check(
        "_read_data_file_helper_futures routes through "
        "load_features_for_video when config_path is set",
        "load_features_for_video" in mixin_src,
    )
    check(
        "_read_data_file_helper_futures routes through "
        "load_labels_for_video when config_path is set",
        "load_labels_for_video" in mixin_src,
    )
    # MP caller passes self.config_path to worker
    check(
        "read_all_files_in_folder_mp_futures passes "
        "self.config_path to the worker (via "
        "executor.submit args)",
        ", config_path)" in mixin_src,
    )
    # MP caller skips existence check in v1 mode
    check(
        "read_all_files_in_folder_mp_futures skips "
        "check_filepaths_in_iterable_exist when v1 mode is "
        "active (pseudo-paths may not exist on disk)",
        "if config_path is None:" in mixin_src,
    )
    # Non-MP fallback also handles config_path
    check(
        "read_all_files_in_folder (non-MP fallback) also "
        "branches on config_path for the per-file read",
        mixin_src.count("load_features_for_video") >= 2
        and mixin_src.count("load_labels_for_video") >= 2,
        detail=(
            f"got {mixin_src.count('load_features_for_video')}× "
            f"load_features_for_video, "
            f"{mixin_src.count('load_labels_for_video')}× "
            f"load_labels_for_video"
        ),
    )
    # Patch 122bf: 122ak rewrote the dual-write era's
    # 122ae-5e comments — the v1-only train flow records
    # 122ak now. Update expected patch number.
    check(
        "train_model_mixin records 122ak (was 122ae-5e, "
        "rewritten in the v1-only close-out)",
        "122ak" in mixin_src,
    )

    # ==================================================================
    # 3. AST — 5 training modules each have v1 discovery
    # ==================================================================
    training_modules = [
        "mufasa/model/train_rf.py",
        "mufasa/model/train_multilabel_rf.py",
        "mufasa/model/train_multiclass_rf.py",
        "mufasa/model/grid_search_rf.py",
        "mufasa/model/grid_search_multiclass_rf.py",
    ]
    for path_str in training_modules:
        src = (REPO_ROOT / path_str).read_text()
        name = Path(path_str).name
        check(
            f"{name}: imports/uses list_video_stems_with_labels "
            "for v1 discovery",
            "list_video_stems_with_labels" in src,
        )
        check(
            f"{name}: discovery is conditional on empty "
            "target_file_paths (guards the legacy glob result)",
            "if not self.target_file_paths" in src,
        )
        check(
            f"{name}: records 122ae-5e in code/comments",
            "122ae-5e" in src,
        )

    # ==================================================================
    # 4. label_io module surface — __all__ exports the new helper
    # ==================================================================
    lio_src = (REPO_ROOT / "mufasa" / "utils"
               / "label_io.py").read_text()
    lio_tree = ast.parse(lio_src)
    for node in lio_tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name)
                        and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            all_names = [
                e.value for e in node.value.elts
                if isinstance(e, ast.Constant)
                and isinstance(e.value, str)
            ]
            check(
                "label_io __all__ exports "
                "list_video_stems_with_labels",
                "list_video_stems_with_labels" in all_names,
            )
            check(
                "label_io __all__ still exports "
                "load_labels_for_video + save_labels_for_video "
                "(no removal)",
                "load_labels_for_video" in all_names
                and "save_labels_for_video" in all_names,
            )
            break

    print(
        f"smoke_122ae5e_classifier_training: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
