"""
tests/smoke_frame_ops_cluster.py
================================

Patch 122s: regression guard for the three frame-ops forms in
``mufasa/ui_qt/forms/video_frames.py`` plus the legacy-popup
removals in ``mufasa/ui/pop_ups/video_processing_pop_up.py``
and SimBA.py wiring scrubs.

AST-only — the video forms transitively import cv2 / ffmpeg
helpers from the video_processors backend, neither available
in the sandbox.

Coverage:

1. **ExtractFramesForm — completeness** beyond the prior stub
   state. Form must wire SEQ extraction, image format,
   greyscale, clahe, and include_fn through to the backend;
   directory mode must dispatch per-video.
2. **MergeFramesToVideoForm shape** — directory picker, FPS /
   quality / format / GPU controls, target() calls
   ``frames_to_movie``.
3. **ImportFrameDirectoryForm shape** — src + dst pickers,
   symlink toggle, target() calls shutil.copytree or
   os.symlink, collect_args() validates source-has-images
   and destination-doesn't-exist.
4. **Legacy class removal** — the four classes are gone from
   ``video_processing_pop_up.py``; deletion-record markers
   remain.
5. **SimBA.py wiring scrubbed** — no remaining references to
   the four classes as code (only deletion-record comments).
6. **Video Processing page** — new sections registered.
"""
from __future__ import annotations

import ast
import re
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
    # video_frames.py — three forms
    # ==================================================================
    vf_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "video_frames.py")
    vf_src = vf_path.read_text()
    vf_tree = ast.parse(vf_src)

    # --- ExtractFramesForm (rewritten) ---
    cls = _find_class(vf_tree, "ExtractFramesForm")
    check("ExtractFramesForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.scope", "self.save_dir_edit", "self.start_frame",
            "self.end_frame", "self.fmt_cb", "self.greyscale",
            "self.clahe", "self.include_fn",
        ):
            check(
                f"ExtractFramesForm sets {attr}",
                attr in class_src,
            )
        methods = {
            n.name: n for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "ExtractFramesForm.target wires SEQ extractor "
                "(no more NotImplementedError)",
                "extract_seq_frames" in target_src
                and "NotImplementedError" not in target_src,
            )
            check(
                "ExtractFramesForm.target passes img_format to "
                "extract_frame_range",
                "img_format=" in target_src,
            )
            check(
                "ExtractFramesForm.target passes greyscale + "
                "clahe + include_fn",
                "greyscale=" in target_src
                and "clahe=" in target_src
                and "include_fn=" in target_src,
            )
            check(
                "ExtractFramesForm.target passes save_dir to "
                "extract_frame_range",
                "save_dir=" in target_src,
            )
            check(
                "ExtractFramesForm.target handles directory mode "
                "(loops over Path(path).iterdir())",
                "iterdir()" in target_src,
            )

    # --- MergeFramesToVideoForm (new) ---
    cls = _find_class(vf_tree, "MergeFramesToVideoForm")
    check("MergeFramesToVideoForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.dir_edit", "self.fps", "self.quality",
            "self.fmt_cb", "self.gpu",
        ):
            check(
                f"MergeFramesToVideoForm sets {attr}",
                attr in class_src,
            )
        check(
            "MergeFramesToVideoForm.target calls frames_to_movie",
            "frames_to_movie(" in class_src,
        )
        check(
            "MergeFramesToVideoForm format combo includes mp4/avi/webm",
            "'mp4'" in class_src and "'avi'" in class_src
            and "'webm'" in class_src,
        )

    # --- ImportFrameDirectoryForm (new) ---
    cls = _find_class(vf_tree, "ImportFrameDirectoryForm")
    check("ImportFrameDirectoryForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.src_edit", "self.dst_edit", "self.symlink",
        ):
            check(
                f"ImportFrameDirectoryForm sets {attr}",
                attr in class_src,
            )
        check(
            "ImportFrameDirectoryForm.target uses shutil.copytree "
            "for copy mode",
            "shutil.copytree" in class_src,
        )
        check(
            "ImportFrameDirectoryForm.target uses os.symlink "
            "for symlink mode",
            "os.symlink" in class_src,
        )
        # validation: must have images
        if "collect_args" in {
            n.name for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }:
            for n in cls.body:
                if (isinstance(n, ast.FunctionDef)
                        and n.name == "collect_args"):
                    ca_src = ast.unparse(n)
                    check(
                        "ImportFrameDirectoryForm.collect_args "
                        "validates the source has image files",
                        "_IMG_EXTS" in ca_src
                        or "No image files found" in ca_src,
                    )
                    check(
                        "ImportFrameDirectoryForm.collect_args "
                        "refuses an already-existing destination",
                        "exists()" in ca_src
                        and "already exists" in ca_src,
                    )

    # ==================================================================
    # Legacy class removal from video_processing_pop_up.py
    # ==================================================================
    vp_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "video_processing_pop_up.py")
    vp_src = vp_path.read_text()
    vp_tree = ast.parse(vp_src)
    defined = {
        n.name for n in vp_tree.body
        if isinstance(n, ast.ClassDef)
    }

    DOOMED = {
        "ImportFrameDirectoryPopUp", "MergeFrames2VideoPopUp",
        "ExtractSpecificFramesPopUp", "ExtractSEQFramesPopUp",
    }
    for d in sorted(DOOMED):
        check(
            f"legacy {d} class no longer defined",
            d not in defined,
        )
        marker_re = re.compile(
            rf"^# {re.escape(d)}: removed in patch 122s",
            re.MULTILINE,
        )
        check(
            f"deletion-record marker for {d} present",
            bool(marker_re.search(vp_src)),
        )

    # ==================================================================
    # SimBA.py — no remaining code references to the 4 classes
    # ==================================================================
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    for d in sorted(DOOMED):
        # The string may appear in a deletion-record comment; require
        # that no NON-comment line mentions it.
        leaked = any(
            d in line and not line.lstrip().startswith("#")
            for line in simba_src.splitlines()
        )
        check(
            f"SimBA.py has no code-level reference to {d}",
            not leaked,
        )

    # ==================================================================
    # video_processing_page.py — new sections registered
    # ==================================================================
    vpp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "video_processing_page.py").read_text()
    check(
        "video_processing_page imports MergeFramesToVideoForm",
        "MergeFramesToVideoForm" in vpp_src,
    )
    check(
        "video_processing_page imports ImportFrameDirectoryForm",
        "ImportFrameDirectoryForm" in vpp_src,
    )
    check(
        "video_processing_page registers a 'Merge frames → video' "
        "or similarly-labelled section",
        "Merge frames" in vpp_src,
    )
    check(
        "video_processing_page registers an 'Import frame directory' "
        "section",
        "Import frame directory" in vpp_src,
    )

    print(
        f"smoke_frame_ops_cluster: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
