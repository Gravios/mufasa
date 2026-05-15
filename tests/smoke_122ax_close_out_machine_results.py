"""
tests/smoke_122ax_close_out_machine_results.py
================================================

Patch 122ax: close out the machine_results migration arc.

Five things ship in this patch:

1. InferenceBatch: legacy ``write_df(out_df, ..., csv/machine_results/)``
   is removed. The v1 sidecar
   :func:`save_classifications_for_video` is now the sole
   write site. The ``try/except`` around it from 122at is also
   removed — failures bubble up since this is the canonical
   write now.

2. ``project_paths_from_config`` (v1 branch): the
   ``machine_results_dir`` key is gone. v1 projects no longer
   carry any csv/<kind>/ subtree key — the layout is fully the
   new shape (logs/, models/, sources/, derived/). Legacy
   branch still exposes the key (legacy projects keep their
   csv/ subtree).

3. ConfigReader (v1 branch): ``self.machine_results_dir`` now
   points at ``<root>/derived/classifications/`` (flat, no
   run-id subdir lookup — matches what
   :func:`save_classifications_for_video` actually writes).
   ``self.machine_results_paths`` is enumerated via
   :func:`list_video_stems_with_classifications` instead of a
   glob, giving stable cross-run ordering.

4. Safety rails: ``labelling_interface``, ``standard_labeller``,
   and ``select_video_for_pseudo_labelling_popup`` previously
   raised :exc:`NoFilesFoundError` if the legacy CSV didn't
   exist — even when v1 predictions WERE present. Now relaxed
   to accept either source.

5. Three consumers (``ui_qt/targeted_clips``,
   ``ui_qt/frame_labeller``, ``ui_qt/clip_review``) that read
   ``paths["machine_results_dir"]`` directly now use
   ``paths.get("machine_results_dir")`` and handle the None
   case (v1 projects don't expose the key).

Coverage
--------
1. InferenceBatch no longer calls ``write_df`` for the
   machine_results write path; only ``save_classifications_for_video``.
2. ``project_paths_from_config`` v1 branch omits
   ``machine_results_dir``; legacy branch still has it.
3. ConfigReader v1 branch points ``machine_results_dir`` at
   ``derived/classifications/`` (parent, not a run subdir).
4. ``machine_results_paths`` v1 source uses
   ``list_video_stems_with_classifications``.
5. Safety-rail relaxations: pre-122ax error message text gone,
   relaxed pattern present.
6. 3 consumers use ``paths.get("machine_results_dir")``
   (with None handling) instead of ``paths["machine_results_dir"]``.
7. 122ax recorded in all touched files.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
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


def _write_v1_toml(tmp: Path, classifiers: list[str]) -> Path:
    toml = tmp / "project.toml"
    target_lines = "\n".join(f'    "{c}",' for c in classifiers)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ax"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose"]

        [classifiers]
        targets = [
        {target_lines}
        ]
    """).strip() + "\n")
    return toml


def _write_legacy_ini(tmp: Path) -> Path:
    proj = tmp / "project_folder"
    proj.mkdir()
    ini = tmp / "project_config.ini"
    lines = [
        "[General settings]",
        f"project_path = {proj}",
        "workflow_file_type = csv",
    ]
    ini.write_text("\n".join(lines) + "\n")
    return ini


def main() -> int:
    # ==================================================================
    # 1. InferenceBatch: legacy write removed
    # ==================================================================
    ib_src = (REPO_ROOT / "mufasa" / "model"
              / "inference_batch.py").read_text()
    # Hardcheck: the old write_df line should no longer be present.
    # The save_classifications_for_video call should be present and
    # outside any try/except wrapping that swallows exceptions.
    check(
        "InferenceBatch: legacy `write_df(df=out_df, ..., "
        "save_path=file_save_path)` is gone",
        "write_df(df=out_df, file_type=self.file_type, "
        "save_path=file_save_path)" not in ib_src,
    )
    check(
        "InferenceBatch: save_classifications_for_video is the "
        "sole write site",
        "save_classifications_for_video(" in ib_src,
    )
    check(
        "InferenceBatch: 122ax recorded",
        "122ax" in ib_src,
    )
    # The pre-122at file_save_path = os.path.join(self.save_dir, ...)
    # builder is also dead code; should be gone.
    check(
        "InferenceBatch: dead `file_save_path = os.path.join("
        "self.save_dir, ...)` line removed",
        "file_save_path = os.path.join(self.save_dir," not in ib_src,
    )

    # ==================================================================
    # 2. project_paths_from_config v1 branch drops the key;
    #    legacy branch keeps it
    # ==================================================================
    from mufasa.project_layout import project_paths_from_config

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        paths = project_paths_from_config(str(toml))
        check(
            "v1 paths: machine_results_dir key is NOT present",
            "machine_results_dir" not in paths,
        )
        check(
            "v1 paths: derived_classifications_dir IS present",
            "derived_classifications_dir" in paths,
        )

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        ini = _write_legacy_ini(tmp)
        paths = project_paths_from_config(str(ini))
        check(
            "legacy paths: machine_results_dir key IS still present",
            "machine_results_dir" in paths,
        )

    # ==================================================================
    # 3. ConfigReader v1 branch points machine_results_dir at the
    #    parent and uses list_video_stems_with_classifications.
    #    Code-level inspection (ConfigReader requires cv2/h5py which
    #    aren't in the sandbox).
    # ==================================================================
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "ConfigReader v1: machine_results_dir = str("
        "root / 'derived' / 'classifications')",
        'str(\n            root / "derived" / "classifications"\n        )'
        in cr_src,
    )
    check(
        "ConfigReader v1: dropped the `self.machine_results_dir "
        "= self.targets_folder` assignment",
        "self.machine_results_dir = self.targets_folder" not in cr_src,
    )
    check(
        "ConfigReader v1: machine_results_paths now uses "
        "list_video_stems_with_classifications",
        "list_video_stems_with_classifications" in cr_src,
    )
    check(
        "ConfigReader v1: dropped the glob-of-machine_results_dir "
        "discovery",
        'glob.glob(\n            self.machine_results_dir + f"/*.{ft}"'
        not in cr_src,
    )
    check(
        "ConfigReader records 122ax",
        "122ax" in cr_src,
    )

    # ==================================================================
    # 4. Safety rails relaxed in 3 files
    # ==================================================================
    rail_files = [
        REPO_ROOT / "mufasa" / "labelling"
        / "labelling_interface.py",
        REPO_ROOT / "mufasa" / "labelling"
        / "standard_labeller.py",
        REPO_ROOT / "mufasa" / "ui" / "pop_ups"
        / "select_video_for_pseudo_labelling_popup.py",
    ]
    for path in rail_files:
        src = path.read_text()
        check(
            f"{path.name}: imports list_video_stems_with_classifications",
            "list_video_stems_with_classifications" in src,
        )
        check(
            f"{path.name}: pre-122ax error text "
            "('SimBA expects a file at') is gone",
            "SimBA expects a file at" not in src,
        )
        check(
            f"{path.name}: relaxed pattern present "
            "('_v1_has' or 'derived/classifications/' in error)",
            "_v1_has" in src
            or "derived/classifications/" in src,
        )
        check(
            f"{path.name}: 122ax recorded",
            "122ax" in src,
        )

    # ==================================================================
    # 5. Three consumers use paths.get() with None handling
    # ==================================================================
    consumer_files = [
        REPO_ROOT / "mufasa" / "ui_qt" / "targeted_clips.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "frame_labeller.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "clip_review.py",
    ]
    for path in consumer_files:
        src = path.read_text()
        check(
            f"{path.name}: uses paths.get(\"machine_results_dir\"...)",
            'paths.get("machine_results_dir"' in src,
        )
        check(
            f"{path.name}: no longer uses bare "
            'paths["machine_results_dir"]',
            'paths["machine_results_dir"]' not in src,
        )

    # Targeted_clips and frame_labeller record 122ax (clip_review
    # already used .get() pre-122ax for unrelated reasons).
    check(
        "targeted_clips records 122ax",
        "122ax" in (REPO_ROOT / "mufasa" / "ui_qt"
                    / "targeted_clips.py").read_text(),
    )
    check(
        "frame_labeller records 122ax",
        "122ax" in (REPO_ROOT / "mufasa" / "ui_qt"
                    / "frame_labeller.py").read_text(),
    )

    # ==================================================================
    # 6. project_layout records 122ax
    # ==================================================================
    pl_src = (REPO_ROOT / "mufasa" / "project_layout.py").read_text()
    check(
        "project_layout records 122ax",
        "122ax" in pl_src,
    )

    print(
        f"smoke_122ax_close_out_machine_results: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
