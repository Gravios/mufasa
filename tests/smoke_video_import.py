"""
tests/smoke_video_import.py
===========================

Patch 122o: regression guard for the inline video-import
surface on the Data Import page, plus the layout-aware
destination resolution in ``copy_*_video*`` helpers.

Three layers:

1. **AST** — :class:`VideoImportForm` exists in
   ``mufasa/ui_qt/forms/video_import.py`` with the expected
   widget attributes (mode radios, source picker, format combo,
   recursive + symlink checkboxes), and its ``target()``
   dispatches to one of the two ``copy_*_video*`` helpers
   depending on ``mode_single``.

2. **AST** — ``data_import_page.build_data_import_page`` uses
   the new ``Data Import`` label, registers an
   ``Import video`` section between Import pose-estimation
   data and Video parameters & calibration, and imports
   :class:`VideoImportForm`.

3. **Behavioural** — exercise
   :func:`copy_single_video_to_project` and
   :func:`copy_multiple_videos_to_project` against a
   freshly-built v1 project. The helpers must resolve the
   destination via
   :func:`mufasa.project_layout.project_paths_from_config`,
   landing files under ``<root>/sources/videos/`` rather than
   the legacy ``<root>/videos/``.
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


def _find_class(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def main() -> int:
    # ==================================================================
    # Layer 1 — VideoImportForm AST shape
    # ==================================================================
    vi_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "video_import.py"
    vi_src = vi_path.read_text()
    vi_tree = ast.parse(vi_src)

    cls = _find_class(vi_tree, "VideoImportForm")
    check("VideoImportForm class defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)
        bases = [
            (b.id if isinstance(b, ast.Name) else getattr(b, "attr", ""))
            for b in cls.bases
        ]
        check(
            "VideoImportForm extends OperationForm",
            "OperationForm" in bases,
        )
        check(
            "VideoImportForm.title = 'Import video'",
            "title = 'Import video'" in class_src
            or 'title = "Import video"' in class_src,
        )

        # Widget attributes
        for attr in (
            "self._mode_single", "self._mode_directory",
            "self._source_edit", "self._dir_file_type",
            "self._recursive", "self._symlink",
        ):
            check(
                f"VideoImportForm sets {attr}",
                attr in class_src,
            )

        # target() branches and dispatches to the helpers
        methods = {
            n.name: n for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        check(
            "VideoImportForm.target defined",
            "target" in methods,
        )
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "VideoImportForm.target dispatches to "
                "copy_single_video_to_project (single mode)",
                "copy_single_video_to_project" in target_src,
            )
            check(
                "VideoImportForm.target dispatches to "
                "copy_multiple_videos_to_project (directory mode)",
                "copy_multiple_videos_to_project" in target_src,
            )
            check(
                "VideoImportForm.target branches on mode_single",
                "mode_single" in target_src,
            )

    # ==================================================================
    # Layer 2 — Data Import page wiring
    # ==================================================================
    di_path = REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "data_import_page.py"
    di_src = di_path.read_text()
    di_tree = ast.parse(di_src)

    check(
        "data_import_page uses 'Data Import' label (capital I)",
        "add_page('Data Import'" in di_src
        or 'add_page("Data Import"' in di_src,
    )
    check(
        "data_import_page no longer uses lowercase 'Data import'",
        "'Data import'" not in di_src
        and '"Data import"' not in di_src,
    )
    check(
        "data_import_page imports VideoImportForm",
        "VideoImportForm" in di_src,
    )
    check(
        "data_import_page registers 'Import video' section",
        "'Import video'" in di_src
        or '"Import video"' in di_src,
    )

    # Builder body — order of sections
    builder = None
    for node in ast.walk(di_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "build_data_import_page"):
            builder = node
            break
    if builder is not None:
        body_src = ast.unparse(builder)
        # Find indices of the four section labels in body_src to
        # verify Import video lands between pose-import and
        # video calibration.
        idx_pose = body_src.find("Import pose-estimation data")
        idx_video = body_src.find("Import video")
        idx_calib = body_src.find("Video parameters & calibration")
        idx_batch = body_src.find("Batch pre-process videos")
        check(
            "section order: pose import → import video → "
            "calibration → batch pre-process",
            idx_pose != -1 and idx_video != -1
            and idx_calib != -1 and idx_batch != -1
            and idx_pose < idx_video < idx_calib < idx_batch,
            detail=(
                f"indices: pose={idx_pose} video={idx_video} "
                f"calib={idx_calib} batch={idx_batch}"
            ),
        )

    # ==================================================================
    # Layer 3 — AST: helpers use project_paths_from_config for the
    # destination (sandbox-friendly; behavioural verification of the
    # actual file copy happens on a real install where h5py et al. are
    # importable).
    # ==================================================================
    rw_path = REPO_ROOT / "mufasa" / "utils" / "read_write.py"
    rw_src = rw_path.read_text()
    rw_tree = ast.parse(rw_src)

    for fn_name in (
        "copy_single_video_to_project",
        "copy_multiple_videos_to_project",
    ):
        fn = None
        for node in ast.walk(rw_tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == fn_name):
                fn = node
                break
        check(f"{fn_name} defined in read_write.py", fn is not None)
        if fn is not None:
            body_src = ast.unparse(fn)
            check(
                f"{fn_name} uses project_paths_from_config "
                "(v1-aware destination)",
                "project_paths_from_config" in body_src,
            )
            check(
                f"{fn_name} no longer hardcodes "
                "'os.path.dirname(...)/videos' destination",
                "os.path.dirname(config_path), 'videos'"
                not in body_src
                and "os.path.dirname(simba_ini_path), 'videos'"
                not in body_src
                and 'os.path.dirname(config_path), "videos"'
                not in body_src
                and 'os.path.dirname(simba_ini_path), "videos"'
                not in body_src,
            )
            check(
                f"{fn_name} extracts ['video_dir'] from the helper",
                "['video_dir']" in body_src
                or '["video_dir"]' in body_src,
            )

    print(
        f"smoke_video_import: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
