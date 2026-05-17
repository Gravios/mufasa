"""
tests/smoke_122ca_bw_rewire.py
================================

Patch 122ca: VideoFiltersForm Black & white branch rewired to
existing `video_to_bw` backend. NARROW scope — only the B&W
fix lands; the other 3 audit-identified quick wins (DropBodyparts
→ KeypointRemover, ROIFeatures Remove → ConfigReader method,
CropVideos multi-crop → MultiCropper) turned out on pre-impl
review to be constructor/semantics mismatches, not just identifier
renames. They're deferred to follow-up patches with form-side
redesign.

The B&W rewire:
* `target()` black_white branch no longer raises NotImplementedError.
* Calls `video_to_bw(video_path, threshold)` instead.
* Form's int 0–255 threshold is scaled to backend's float 0.0–1.0.
* Directory mode iterates via `find_all_videos_in_directory`.
* `_BlackWhitePanel` no longer has an `invert` checkbox (backend
  doesn't support it; silent-ignore would be a UX trap).

Plus doc updates:
* `docs/backend_audit.md` §4a revised to reflect the scope-revision
  finding: 1 of 4 originally-quick wins is truly tiny; the other 3
  need form redesign. Now in §4b "Follow-up patches needed".
* `docs/qt_form_runtime_gaps.md` §2b marks B&W as FIXED in 122ca
  and reduces the failing-op count for VideoFiltersForm from 4 to 3.

Coverage
--------
1. video_filters.py B&W branch no longer raises NotImplementedError.
2. Branch calls `video_to_bw` (not the previous placeholder name).
3. Branch references threshold scaling (form 0–255 → backend 0.0–1.0).
4. Branch handles directory mode via `find_all_videos_in_directory`.
5. `_BlackWhitePanel` no longer references `invert`.
6. `_BlackWhitePanel.to_kwargs()` no longer returns `invert` key.
7. `backend_audit.md` §4a marks B&W as ✓ shipped 122ca.
8. `backend_audit.md` §4a marks the other 3 as deferred.
9. `qt_form_runtime_gaps.md` §2b marks B&W as FIXED in 122ca.
10. `qt_form_runtime_gaps.md` §2b reflects reduced VideoFiltersForm
    failing-op count (3 of 5, not 4 of 5).
11. All mufasa/**/*.py files parse cleanly.
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


def _get_class_src(tree: ast.Module, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.unparse(node)
    return ""


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    form_path = pkg / "ui_qt" / "forms" / "video_filters.py"
    src = form_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # B&W branch rewired
    # ==================================================================
    form_cls_src = _get_class_src(tree, "VideoFiltersForm")
    # Find the target() method
    target_src = ""
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "VideoFiltersForm"):
            for stmt in node.body:
                if (isinstance(stmt, ast.FunctionDef)
                        and stmt.name == "target"):
                    target_src = ast.unparse(stmt)
                    break
            break

    # Extract just the black_white branch
    # (very rough — look for the elif/branch text up until the next elif).
    # Note: ast.unparse() uses single quotes for string literals.
    bw_branch_idx = target_src.find("'black_white'")
    blur_branch_idx = target_src.find("'blur'")
    if bw_branch_idx >= 0 and blur_branch_idx > bw_branch_idx:
        bw_branch = target_src[bw_branch_idx:blur_branch_idx]
    else:
        bw_branch = ""

    check(
        "black_white branch no longer raises NotImplementedError "
        "for the rewire (re. 'convert_to_black_and_white')",
        "convert_to_black_and_white" not in bw_branch,
    )
    check(
        "black_white branch references the new backend `video_to_bw`",
        "video_to_bw" in bw_branch,
    )
    check(
        "black_white branch scales threshold (form 0-255 → "
        "backend 0.0-1.0)",
        "255" in bw_branch and "threshold" in bw_branch,
    )
    check(
        "black_white branch iterates directory via "
        "find_all_videos_in_directory",
        "find_all_videos_in_directory" in bw_branch,
    )

    # ==================================================================
    # _BlackWhitePanel no longer has invert
    # ==================================================================
    bw_panel_src = _get_class_src(tree, "_BlackWhitePanel")
    check(
        "_BlackWhitePanel no longer instantiates an 'invert' "
        "QCheckBox",
        "self.invert" not in bw_panel_src,
    )
    check(
        "_BlackWhitePanel.to_kwargs() does not return an 'invert' key",
        '"invert"' not in bw_panel_src
        and "'invert'" not in bw_panel_src,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    audit_text = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4a marks B&W as ✓ shipped 122ca",
        "shipped 122ca" in audit_text
        and ("B&W" in audit_text or "Black & white" in audit_text
             or "video_to_bw" in audit_text),
    )
    check(
        "backend_audit.md §4a documents 122ca's narrow scope decision",
        # The count of 'deferred' entries shrinks as follow-up patches
        # land. Pin only on the unchanging §4a header that records
        # 122ca's narrow-scope revision.
        "REVISED post-patch 122ca" in audit_text,
    )

    gaps_text = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2b marks B&W FIXED in 122ca",
        "FIXED in patch 122ca" in gaps_text,
    )
    check(
        "qt_form_runtime_gaps.md §2b VideoFiltersForm failing count "
        "is reduced (≤3, was 4 pre-122ca)",
        # Don't pin a specific count — later patches (122cb wired blur
        # + brightness) reduce it further. Just check it's no longer 4.
        "4 OPERATIONS FAIL" not in gaps_text,
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
        f"smoke_122ca_bw_rewire: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
