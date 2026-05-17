"""
tests/smoke_122cf_crop_multi_rewire.py
========================================

Patch 122cf: CropVideosForm multi-crop branch wired. Closes the
last of the 7 originally-counted runtime gaps from the 122by audit.

Pre-122cf the `multi=True, is_dir=False` branch of CropVideosForm
raised NotImplementedError. The 122bz audit found `MultiCropper`
but its mental model is incompatible (folder-mode, many videos →
N crops each, while the form promises single-video multi-region).

Post-122cf the form:

* Adds a `crop_count` QSpinBox (2–20, default 2; suffix
  " regions").
* Adds `_refresh_multi_state` that gates `multicrop` enablement
  on rectangle-shape AND single-file scope. Auto-unchecks
  multicrop when shape changes to circle/polygon OR when scope
  switches to directory. Auto-enables/disables `crop_count`
  with `multicrop`.
* `collect_args` returns `crop_count` only when multicrop is set.
* `target` multi-crop branch loops `ROISelector(path).run()`
  `crop_count` times, then calls `crop_video(...)` with the
  selected region. Output files use `_crop1`, `_crop2`, …,
  `_cropN` suffixes; if files already exist, a timestamp is
  appended to avoid the backend's clobber-guard.
* Does NOT use `MultiCropper` — the mental-model mismatch
  remains; this patch goes with "match form UX via loop"
  instead of reworking the form to fold into folder-mode.

Coverage
--------
1. CropVideosForm has new `crop_count` QSpinBox.
2. CropVideosForm has new `_refresh_multi_state` method.
3. Shape combo box change is wired to `_refresh_multi_state`.
4. multicrop toggle is wired to `_refresh_multi_state`.
5. collect_args returns `crop_count` when multi is set.
6. target() multi-crop-single branch no longer raises
   NotImplementedError.
7. target() multi-crop branch imports ROISelector.
8. target() multi-crop branch imports crop_video.
9. target() multi-crop branch does NOT use MultiCropper.
10. target() multi-crop branch loops `range(crop_count)`.
11. target() multi-crop branch builds suffixed output paths
    (`_crop1`, `_crop2`, …).
12. target() multi-crop branch handles existing-file collisions
    by appending a timestamp.
13. CropVideosForm docstring updated: "Known gap" → "Resolved
    in 122cf".
14. qt_form_runtime_gaps.md §2c marks CropVideos FIXED in 122cf.
15. qt_form_runtime_gaps.md §1 summary marks all 7 originally
    counted gaps closed.
16. backend_audit.md §4b item 6 marks the rewire DONE in 122cf.
17. All mufasa/**/*.py files parse cleanly.
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
    src = (pkg / "ui_qt" / "forms" / "video_editing.py").read_text()
    tree = ast.parse(src)

    form_cls = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "CropVideosForm"):
            form_cls = node
            break
    check("CropVideosForm class found", form_cls is not None)
    if form_cls is None:
        return 1
    cls_src = ast.unparse(form_cls)

    # ==================================================================
    # New widget + slot
    # ==================================================================
    check(
        "CropVideosForm defines `crop_count` QSpinBox",
        "self.crop_count = QSpinBox" in cls_src,
    )
    check(
        "CropVideosForm defines `_refresh_multi_state` method",
        "def _refresh_multi_state" in cls_src,
    )
    check(
        "shape combo box wired to `_refresh_multi_state`",
        "self.shape_cb.currentIndexChanged.connect("
        "self._refresh_multi_state)" in cls_src,
    )
    check(
        "multicrop toggle wired to `_refresh_multi_state`",
        "self.multicrop.toggled.connect(self._refresh_multi_state)"
        in cls_src,
    )

    # ==================================================================
    # collect_args + target
    # ==================================================================
    collect_args_src = ""
    target_src = ""
    for stmt in form_cls.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "collect_args":
                collect_args_src = ast.unparse(stmt)
            elif stmt.name == "target":
                target_src = ast.unparse(stmt)

    check(
        "collect_args returns `crop_count` when multi is set",
        "'crop_count'" in collect_args_src
        and ".crop_count" in collect_args_src,
    )

    # target() — find the multi-crop block via the `if shape == ...`
    # rectangle branch.
    check(
        "target() multi-crop-single branch no longer raises "
        "NotImplementedError",
        # The 'NotImplementedError' literal should be gone from the
        # rectangle branch entirely. CLAHE in a different form
        # still raises but this is the video_editing.py file, so
        # any remaining NotImplementedError would be a regression.
        "NotImplementedError" not in target_src,
    )
    check(
        "target() multi-crop branch imports ROISelector",
        "from mufasa.video_processors.roi_selector import" in target_src
        and "ROISelector" in target_src,
    )
    check(
        "target() multi-crop branch calls crop_video",
        "_vp.crop_video" in target_src
        or "crop_video(" in target_src,
    )
    check(
        "target() multi-crop branch does NOT use MultiCropper",
        "MultiCropper" not in target_src,
    )
    check(
        "target() multi-crop branch loops `range(crop_count)`",
        "range(crop_count)" in target_src,
    )
    check(
        "target() multi-crop branch builds suffixed output paths "
        "(_crop1, _crop2, ...)",
        "_crop" in target_src
        and ("f'{file_name}_crop{i + 1}" in target_src
             or 'f"{file_name}_crop{i + 1}' in target_src),
    )
    check(
        "target() handles existing-file collisions with timestamp",
        "isfile" in target_src
        and ("strftime" in target_src or "stamp" in target_src),
    )

    # ==================================================================
    # Docstring update
    # ==================================================================
    docstring = ast.get_docstring(form_cls) or ""
    check(
        "CropVideosForm docstring no longer has 'Known gap' note",
        "Known gap" not in docstring,
    )
    check(
        "CropVideosForm docstring references 122cf resolution",
        "122cf" in docstring,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2c marks CropVideos FIXED in 122cf",
        "FIXED in patch 122cf" in gaps,
    )
    check(
        "qt_form_runtime_gaps.md §1 summary marks "
        "all 7 originally-counted gaps closed",
        "all 7 originally-counted runtime gaps closed" in gaps,
    )
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4b item 6 marks rewire DONE in 122cf",
        "DONE in patch 122cf" in audit,
    )

    # ==================================================================
    # All files parse cleanly
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
        f"smoke_122cf_crop_multi_rewire: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
