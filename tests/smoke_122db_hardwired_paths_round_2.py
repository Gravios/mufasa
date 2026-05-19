"""
tests/smoke_122db_hardwired_paths_round_2.py
==============================================

Patch 122db: continue the hardwired-paths audit follow-up.

After 122da re-triage, this is the second tranche of fixes. The
genuine Qt-reachable bugs (after correctly accounting for
`self.project_path` resolution + defensive fallbacks + legacy-
gated branches) reduced from "12" to **3**:

* `video_processors/video_processing.py:2207` — frames extraction
  using broken `read_config_entry` + legacy-only subpath.
* `ui_qt/forms/pose_cleanup.py:1318` — smoothed_v2 default
  unconditionally set even for v1 projects.
* `ui_qt/forms/visualizations.py:1233` — per-route src_dir
  hardcodes legacy csv path. DEFERRED — needs a route-metadata
  refactor, scope beyond 122db.

This patch fixes the first two. The third is tracked in the
audit doc with "Deferred — needs route-metadata refactor".

Coverage
--------
1.  video_processing.py extract_frames_from_all_videos_in_directory
    no longer uses `read_config_entry(..., "General settings",
    "project_path", ...)` (which fails on v1 .toml).
2.  Same function now uses project_paths_from_config to get
    project_root.
3.  Same function branches by layout for frames root:
    `derived/frames/extracted/` for v1; `frames/input/` for legacy.
4.  pose_cleanup.py L1318 (smoothed-v2 default) is gated by
    `if not config_path.endswith(".toml")` so v1 projects leave
    the field blank.
5.  Audit doc records the re-triage with the false-positive
    breakdown (3 categories: self.project_path-based,
    defensive fallback, branch-gated).
6.  Audit doc adds the "triage rule" process lesson.
7.  Parse-clean.
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


def _code_only(src: str) -> str:
    """Drop comment-only lines so checks don't fire on
    archaeological breadcrumb comments quoting pre-fix code.
    Established pattern from 122d9 / 122da."""
    return "\n".join(
        line for line in src.split("\n")
        if not line.lstrip().startswith("#")
    )


def _function_body_unparsed(src: str, name: str) -> str:
    """Return ast.unparse() of a top-level function body — drops
    comments naturally."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return ""


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # --- video_processing.py ---
    vp_path = pkg / "video_processors" / "video_processing.py"
    vp_src = vp_path.read_text()
    vp_body = _function_body_unparsed(
        vp_src, "extract_frames_from_all_videos_in_directory",
    )
    check(
        "extract_frames_from_all_videos_in_directory exists",
        bool(vp_body),
    )

    # 1. No more read_config_entry on "General settings", "project_path"
    bad_call = re.search(
        r"read_config_entry\([^)]*['\"]General settings['\"][^)]*"
        r"project_path",
        vp_body, re.DOTALL,
    )
    check(
        "No `read_config_entry(..., \"General settings\", "
        "\"project_path\", ...)` in extract_frames_…",
        bad_call is None,
    )

    # 2. Uses project_paths_from_config + project_root
    check(
        "extract_frames_… uses project_paths_from_config + "
        "project_root",
        "project_paths_from_config" in vp_body
        and ("'project_root'" in vp_body
             or '"project_root"' in vp_body),
    )

    # 3. Branches by layout — v1 → derived/frames/extracted/,
    # legacy → frames/input/
    check(
        "extract_frames_… constructs the v1 frames path "
        "(derived/frames/extracted/)",
        "'derived'" in vp_body and "'extracted'" in vp_body
        or '"derived"' in vp_body and '"extracted"' in vp_body,
    )
    check(
        "extract_frames_… preserves the legacy frames path "
        "(frames/input/)",
        ("'input'" in vp_body or '"input"' in vp_body)
        and ("'frames'" in vp_body or '"frames"' in vp_body),
    )
    check(
        "extract_frames_… branches on `.endswith(\".toml\")` "
        "(layout detection)",
        ".endswith('.toml')" in vp_body
        or '.endswith(".toml")' in vp_body,
    )

    # 4. pose_cleanup.py L1318 — smoothed-v2 default now legacy-gated
    pc_path = pkg / "ui_qt" / "forms" / "pose_cleanup.py"
    pc_src = pc_path.read_text()
    # Find the smoothed_v2 block — it's near L1318.
    # Verify there's a `endswith(".toml")` check within 12 lines
    # above any `csv/smoothed_v2` join.
    pc_lines = pc_src.split("\n")
    smoothed_lines = [
        i for i, L in enumerate(pc_lines)
        if "smoothed_v2" in L and "csv" in L
        and not L.lstrip().startswith("#")
    ]
    smoothed_gated = []
    for ln in smoothed_lines:
        for lookback in range(ln, max(0, ln - 15), -1):
            if "endswith" in pc_lines[lookback]:
                smoothed_gated.append(ln)
                break
    check(
        f"pose_cleanup.py smoothed_v2 references are gated by "
        f"`.endswith(\".toml\")` v1 check (found "
        f"{len(smoothed_gated)} / {len(smoothed_lines)} gated)",
        len(smoothed_gated) == len(smoothed_lines)
        and len(smoothed_lines) > 0,
    )

    # 5. Audit doc records re-triage
    audit_path = REPO_ROOT / "docs" / "hardwired_paths_audit.md"
    audit_src = audit_path.read_text()
    check(
        "Audit doc records re-triage with 'NOT A BUG' "
        "categorisations",
        "NOT A BUG (re-triage)" in audit_src,
    )
    check(
        "Audit doc names the 3 false-positive categories",
        "defensive" in audit_src.lower()
        and ("branch-gated" in audit_src.lower()
             or "legacy gated" in audit_src.lower()
             or "branch_gated" in audit_src.lower())
        and "self.project_path" in audit_src,
    )

    # 6. Audit doc adds the triage-rule process lesson (#5)
    check(
        "Audit doc adds the 'Triage rule' process lesson",
        "Triage rule" in audit_src or "triage rule" in audit_src,
    )

    # 7. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122db_hardwired_paths_round_2: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
