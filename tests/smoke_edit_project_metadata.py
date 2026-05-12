"""
tests/smoke_edit_project_metadata.py
====================================

Patch 122n: regression guard for the edit-metadata surface on
the Projects page.

Coverage:

1. **AST** — ``EditProjectMetadataDialog`` exists with the
   expected fields (file type, animal count, animal IDs,
   body parts, classifier targets), an Auto-detect button,
   a Save / Cancel button row, a ``metadata_updated`` Signal,
   and a guard that refuses legacy INI projects.

2. **AST** — :class:`ProjectInfoForm` has an Edit button next
   to Refresh, opens ``EditProjectMetadataDialog``, and refreshes
   its display on save.

3. **AST** — ``"none configured"`` no longer appears in the
   project_info.py source; ``"none"`` is the new wording.

4. **Behavioural** — round-trip a v1 project.toml: read it via
   :func:`project_metadata_from_config`, apply the same write
   the dialog performs, re-read, and assert the new values are
   reflected. This exercises the read-modify-write path that
   the dialog's _on_save delegates to; the Qt event loop side
   isn't reachable in this sandbox.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import tomllib
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


def main() -> int:
    # ------------------------------------------------------------------
    # 1. EditProjectMetadataDialog shape
    # ------------------------------------------------------------------
    dlg_path = (
        REPO_ROOT
        / "mufasa" / "ui_qt" / "dialogs"
        / "edit_project_metadata_dialog.py"
    )
    check(
        "edit_project_metadata_dialog.py exists",
        dlg_path.is_file(),
    )
    if not dlg_path.is_file():
        print(
            f"smoke_edit_project_metadata: "
            f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
        )
        return 1

    dlg_src = dlg_path.read_text()
    dlg_tree = ast.parse(dlg_src)

    dlg_cls = None
    for node in ast.walk(dlg_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "EditProjectMetadataDialog"):
            dlg_cls = node
            break
    check(
        "EditProjectMetadataDialog class defined",
        dlg_cls is not None,
    )

    if dlg_cls is not None:
        cls_src = ast.unparse(dlg_cls)
        # Signals
        check(
            "metadata_updated Signal(str) declared",
            "metadata_updated" in cls_src and "Signal(str)" in cls_src,
        )
        # Field widgets
        for attr in (
            "self._file_type_combo",
            "self._animal_count",
            "self._animal_ids_edit",
            "self._body_parts_edit",
            "self._classifiers_edit",
        ):
            check(
                f"EditProjectMetadataDialog creates {attr}",
                attr in cls_src,
            )
        # Auto-detect button + handler
        check(
            "Auto-detect from pose file button + handler present",
            "Auto-detect from pose file" in cls_src
            and "_autodetect_body_parts" in cls_src,
        )
        check(
            "Auto-detect strategy chain exists",
            "_extract_bodyparts_from_pose_file" in cls_src,
        )
        # Save + Cancel via QDialogButtonBox
        check(
            "QDialogButtonBox.Cancel + Save AcceptRole present",
            "QDialogButtonBox.Cancel" in cls_src
            and "QDialogButtonBox.AcceptRole" in cls_src,
        )
        check(
            "_on_save reads and writes project.toml",
            "read_project_toml" in cls_src
            and "write_project_toml" in cls_src,
        )
        check(
            "_on_save emits metadata_updated",
            "self.metadata_updated.emit(" in cls_src,
        )
        # Legacy guard
        check(
            "Legacy INI projects refused (not silently edited)",
            "_is_v1" in cls_src
            and "Legacy project" in cls_src,
        )

    # ------------------------------------------------------------------
    # 2. ProjectInfoForm Edit button wiring
    # ------------------------------------------------------------------
    info_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "project_info.py"
    )
    info_src = info_path.read_text()

    check(
        "ProjectInfoForm has an Edit button",
        "Edit…" in info_src or "Edit\\u2026" in info_src,
    )
    check(
        "ProjectInfoForm._open_edit_dialog method defined",
        "_open_edit_dialog" in info_src,
    )
    check(
        "_open_edit_dialog imports EditProjectMetadataDialog",
        "EditProjectMetadataDialog" in info_src,
    )
    check(
        "Edit dialog's metadata_updated wired to _populate refresh",
        "metadata_updated.connect" in info_src
        and "_populate" in info_src,
    )

    # ------------------------------------------------------------------
    # 3. Wording: "none configured" gone
    # ------------------------------------------------------------------
    check(
        "ProjectInfoForm no longer says 'none configured'",
        "none configured" not in info_src,
    )
    # Positive: the new wording is in use
    check(
        "ProjectInfoForm uses '<i>none</i>' for missing body parts / "
        "classifiers",
        "<i>none</i>" in info_src,
    )

    # ------------------------------------------------------------------
    # 4. Behavioural — round-trip via project_layout helpers
    # ------------------------------------------------------------------
    from mufasa.utils.config_creator import ProjectConfigCreator
    from mufasa.project_layout import (
        project_metadata_from_config,
        read_project_toml,
        write_project_toml,
    )

    with tempfile.TemporaryDirectory() as tmp:
        creator = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="exp_122n",
            target_list=["OldClf"],
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,
            animal_cnt=1,
            file_type="csv",
        )
        cfg = creator.config_path

        # Read initial state
        before = project_metadata_from_config(cfg)
        check(
            "round-trip pre-state: file_type = csv",
            before["file_type"] == "csv",
        )
        check(
            "round-trip pre-state: 1 animal",
            before["animal_count"] == 1,
        )
        check(
            "round-trip pre-state: 7 body parts (from preset)",
            len(before["body_parts"]) == 7,
        )
        check(
            "round-trip pre-state: 1 classifier",
            before["classifier_targets"] == ["OldClf"],
        )

        # Apply the same edits the dialog's _on_save performs
        data = read_project_toml(cfg)
        pose = dict(data.get("pose", {}))
        pose["file_type"] = "parquet"
        pose["animal_count"] = 2
        pose["animal_ids"] = ["Mouse_A", "Mouse_B"]
        pose["body_parts"] = ["Nose", "Ear_left", "Ear_right",
                              "Center", "Tail_base"]
        data["pose"] = pose
        classifiers = dict(data.get("classifiers", {}))
        classifiers["targets"] = ["Attack", "Sniff", "Approach"]
        data["classifiers"] = classifiers
        write_project_toml(Path(cfg), data)

        # Re-read and verify
        after = project_metadata_from_config(cfg)
        check(
            "round-trip post-state: file_type = parquet",
            after["file_type"] == "parquet",
        )
        check(
            "round-trip post-state: 2 animals",
            after["animal_count"] == 2,
        )
        check(
            "round-trip post-state: animal_ids changed",
            after["animal_ids"] == ["Mouse_A", "Mouse_B"],
        )
        check(
            "round-trip post-state: body_parts changed",
            after["body_parts"] == ["Nose", "Ear_left", "Ear_right",
                                    "Center", "Tail_base"],
        )
        check(
            "round-trip post-state: classifier_targets changed",
            after["classifier_targets"]
            == ["Attack", "Sniff", "Approach"],
        )

        # Verify the on-disk TOML reflects the edit
        with open(cfg, "rb") as f:
            raw = tomllib.load(f)
        check(
            "on-disk project.toml [pose].file_type = parquet",
            raw["pose"]["file_type"] == "parquet",
        )
        check(
            "on-disk project.toml [classifiers].targets correct",
            raw["classifiers"]["targets"]
            == ["Attack", "Sniff", "Approach"],
        )

    print(
        f"smoke_edit_project_metadata: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
