"""
tests/smoke_122cb_blur_brightness_backends.py
===============================================

Patch 122cb: blur + brightness/contrast FFmpeg-based backend
functions added to `mufasa/video_processors/video_processing.py`.
Wired into `VideoFiltersForm`'s `blur` and `brightness_contrast`
branches.

The new backends:

* `video_blur(video_path, kernel_size, method='gaussian',
              save_dir, gpu)` — FFmpeg's gblur / boxblur filter.
* `video_brightness_contrast(video_path, brightness, contrast,
              save_dir, gpu)` — FFmpeg's eq filter.

Both follow the `video_to_bw` template: file-or-dir input via
`find_all_videos_in_directory`; output alongside source (or to
`save_dir` if provided); per-file FFmpeg subprocess invocation
with GPU codec swap.

Coverage
--------
1. `video_blur` function defined at module scope in
   `mufasa/video_processors/video_processing.py`.
2. `video_brightness_contrast` function defined at module scope.
3. `video_blur` signature has the expected kwargs.
4. `video_brightness_contrast` signature has the expected kwargs.
5. `video_blur` body references FFmpeg's gblur filter.
6. `video_blur` body references FFmpeg's boxblur filter (for
   the method='box' branch).
7. `video_brightness_contrast` body references FFmpeg's eq filter.
8. `VideoFiltersForm.target()` blur branch calls `video_blur`
   (not NotImplementedError).
9. `VideoFiltersForm.target()` brightness_contrast branch calls
   `video_brightness_contrast` (not NotImplementedError).
10. Form's panel kwargs (kernel_size, brightness, contrast) are
    forwarded to the backends.
11. `qt_form_runtime_gaps.md` §2b marks blur and brightness as
    FIXED in 122cb.
12. `qt_form_runtime_gaps.md` §2b reflects the reduced failing-op
    count for VideoFiltersForm (1 of 5, was 3 of 5 after 122ca).
13. `backend_audit.md` §4c marks both fixes as DONE in 122cb.
14. All mufasa/**/*.py files parse cleanly.
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


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    vp_path = pkg / "video_processors" / "video_processing.py"
    src = vp_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # 1-2. New functions defined at module scope
    # ==================================================================
    blur_fn = None
    bc_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "video_blur":
                blur_fn = node
            elif node.name == "video_brightness_contrast":
                bc_fn = node
    check("video_blur defined at module scope", blur_fn is not None)
    check("video_brightness_contrast defined at module scope", bc_fn is not None)

    # ==================================================================
    # 3-4. Signatures
    # ==================================================================
    if blur_fn is not None:
        arg_names = [a.arg for a in blur_fn.args.args]
        for kw in ("video_path", "kernel_size", "method",
                   "save_dir", "gpu"):
            check(
                f"video_blur accepts kwarg '{kw}'",
                kw in arg_names,
            )

    if bc_fn is not None:
        arg_names = [a.arg for a in bc_fn.args.args]
        for kw in ("video_path", "brightness", "contrast",
                   "save_dir", "gpu"):
            check(
                f"video_brightness_contrast accepts kwarg '{kw}'",
                kw in arg_names,
            )

    # ==================================================================
    # 5-7. FFmpeg filter references in bodies
    # ==================================================================
    if blur_fn is not None:
        body_src = ast.unparse(blur_fn)
        check(
            "video_blur uses FFmpeg's gblur filter (gaussian method)",
            "gblur" in body_src,
        )
        check(
            "video_blur uses FFmpeg's boxblur filter (box method)",
            "boxblur" in body_src,
        )

    if bc_fn is not None:
        body_src = ast.unparse(bc_fn)
        check(
            "video_brightness_contrast uses FFmpeg's eq filter",
            "eq=brightness=" in body_src or "eq " in body_src,
        )

    # ==================================================================
    # 8-10. Form wiring
    # ==================================================================
    form_path = pkg / "ui_qt" / "forms" / "video_filters.py"
    form_src = form_path.read_text()
    form_tree = ast.parse(form_src)
    target_src = ""
    for node in ast.walk(form_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "VideoFiltersForm"):
            for stmt in node.body:
                if (isinstance(stmt, ast.FunctionDef)
                        and stmt.name == "target"):
                    target_src = ast.unparse(stmt)
                    break

    # blur branch
    blur_branch_start = target_src.find("'blur'")
    blur_branch_end = target_src.find("'brightness_contrast'")
    blur_branch = (target_src[blur_branch_start:blur_branch_end]
                   if blur_branch_start >= 0 else "")
    check(
        "blur branch calls _vp.video_blur (not raises)",
        "video_blur" in blur_branch
        and "NotImplementedError" not in blur_branch,
    )
    check(
        "blur branch forwards kernel_size",
        "kernel_size" in blur_branch,
    )

    # brightness_contrast branch
    bc_branch_start = target_src.find("'brightness_contrast'")
    bc_branch = (target_src[bc_branch_start:]
                 if bc_branch_start >= 0 else "")
    check(
        "brightness_contrast branch calls "
        "_vp.video_brightness_contrast (not raises)",
        "video_brightness_contrast" in bc_branch
        and "NotImplementedError" not in bc_branch,
    )
    check(
        "brightness_contrast branch forwards brightness + contrast",
        "brightness" in bc_branch and "contrast" in bc_branch,
    )

    # ==================================================================
    # 11-13. Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2b marks blur as FIXED in 122cb",
        "FIXED in patch 122cb" in gaps and "blur" in gaps.lower(),
    )
    check(
        "qt_form_runtime_gaps.md §2b marks brightness as FIXED in 122cb",
        "FIXED in patch 122cb" in gaps
        and ("brightness" in gaps.lower() or "Brightness" in gaps),
    )
    check(
        "qt_form_runtime_gaps.md §2b reflects reduced VideoFiltersForm "
        "failing count (no longer 3 or 4 of 5; CLAHE preview later "
        "closed in 122ci → 0)",
        # Don't pin a specific count — later patches reduce it
        # further. Just check the original 3-failing / 4-failing
        # status is gone.
        "3 OPERATIONS FAIL" not in gaps
        and "4 OPERATIONS FAIL" not in gaps,
    )

    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4c marks blur fix as DONE in 122cb",
        "DONE in patch 122cb" in audit,
    )
    check(
        "backend_audit.md §4c references video_blur backend by name",
        "video_blur" in audit,
    )
    check(
        "backend_audit.md §4c references "
        "video_brightness_contrast backend by name",
        "video_brightness_contrast" in audit,
    )

    # ==================================================================
    # 14. All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122cb_blur_brightness_backends: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
