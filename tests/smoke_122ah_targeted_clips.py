"""
tests/smoke_122ah_targeted_clips.py
====================================

Patch 122ah: closes the last v1-awareness regression found in
the post-122ag audit sweep.

Background — what the audit looked for:
The user asked "any other hardwired paths in the project?"
after 122ag closed roi_define_panel + VideoInfoForm +
clip_review. A systematic sweep of mufasa/ui_qt/ and
mufasa/cli/ for the bug shape that crashed those modules
(configparser reading [General settings].project_path
directly) found one more real bug and several
intentionally-legacy branches.

Real bug fixed:
  mufasa/ui_qt/targeted_clips.py — ClipRangeAnnotator.
  _load_project_metadata read project_path + workflow_file_type
  directly via configparser, then hardcoded
  '<project>/frames/input/advanced_clip_annotator/<video>' and
  '<project>/csv/machine_results/' subtree joins.
  v1 users couldn't open the targeted clip annotator at all.

Intentional legacy branches (NOT bugs):
  - input_source_picker.py — '# Legacy second' branch of
    dual-layout discovery; gated by v1-first ordering.
  - forms/addons.py — gated by '.toml' extension check that
    returns early for v1 (cue-light ROI metadata not in v1
    schema yet).
  - forms/pose_cleanup.py — two '# Legacy INI' branches in
    dual-layout helpers.
  - forms/classifier.py — '# Legacy: rewrite [SML settings]'
    branch.
  - video_info.py — defensive 'except Exception' fallback
    blocks I wrote in 122ag.

Known deferred item (still open, not in scope here):
  - mufasa/utils/project_reconfigure.py — INI-only,
    documented in the session journal.

Coverage for the fix:

1. targeted_clips.py no longer imports configparser.
2. _load_project_metadata calls project_paths_from_config +
   project_metadata_from_config.
3. self.target_dir derived from paths['project_root'] +
   conventional frames/input/advanced_clip_annotator subtree.
4. self.machine_results_dir reads paths['machine_results_dir']
   directly from the helper.
5. self.file_type reads from project metadata, not INI.
6. No direct reads of '[General settings]' anywhere in
   targeted_clips.py.
7. The 122ah patch note appears in the docstring.

Cross-file invariant: no Qt-surface module (mufasa/ui_qt/)
reads '[General settings]' directly outside of explicitly-
marked legacy branches in dual-layout helpers.
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
    tc_path = REPO_ROOT / "mufasa" / "ui_qt" / "targeted_clips.py"
    tc_src = tc_path.read_text()
    tc_tree = ast.parse(tc_src)

    # ----- 1. configparser import gone -----
    check(
        "targeted_clips no longer imports configparser",
        not any(
            isinstance(n, ast.Import)
            and any(a.name == "configparser" for a in n.names)
            for n in tc_tree.body
        ),
    )

    # ----- 2. layout helpers in use -----
    check(
        "targeted_clips imports / uses project_paths_from_config",
        "project_paths_from_config" in tc_src,
    )
    check(
        "targeted_clips imports / uses project_metadata_from_config "
        "(for v1 file_type)",
        "project_metadata_from_config" in tc_src,
    )

    # ----- 3 & 4. derived paths use helper keys -----
    check(
        "targeted_clips reads paths['project_root'] for target_dir",
        'paths["project_root"]' in tc_src
        or "paths['project_root']" in tc_src,
    )
    check(
        "targeted_clips reads paths['machine_results_dir'] directly",
        'paths["machine_results_dir"]' in tc_src
        or "paths['machine_results_dir']" in tc_src,
    )

    # ----- 5. file_type from metadata -----
    check(
        "targeted_clips reads file_type from project metadata",
        'meta.get("file_type"' in tc_src
        or "meta.get('file_type'" in tc_src,
    )

    # ----- 6. no direct INI reads (in non-comment lines) -----
    leaked_general_settings = any(
        ("'General settings'" in line or '"General settings"' in line)
        and not line.lstrip().startswith("#")
        for line in tc_src.splitlines()
    )
    check(
        "targeted_clips has no code-level reads of "
        "'[General settings]' (comments OK)",
        not leaked_general_settings,
    )

    # ----- 7. 122ah note in docstring -----
    check(
        "targeted_clips records the 122ah fix",
        "122ah" in tc_src,
    )

    # ==================================================================
    # Cross-file invariant: no Qt-surface module reads [General settings]
    # at code level outside intentionally-legacy branches.
    # ==================================================================
    # Sites whitelisted as 'gated legacy branch'. Each entry is
    # (relative_path, reason_for_whitelist). If you add more
    # gated branches in future patches, add them here with the
    # rationale.
    LEGACY_BRANCH_WHITELIST = {
        "mufasa/ui_qt/input_source_picker.py":
            "v1-first dual-layout discovery; legacy branch gated "
            "by `if config_path` after v1 sources scanned first",
        "mufasa/ui_qt/forms/addons.py":
            "cue-light ROI metadata not in v1 schema yet; gated "
            "by `.toml` extension early-return",
        "mufasa/ui_qt/forms/pose_cleanup.py":
            "dual-layout helpers; legacy branches commented as "
            "such (# Legacy INI)",
        "mufasa/ui_qt/forms/classifier.py":
            "dual-layout helper; legacy branch commented as such",
        # Patch 122ai extensions: directories outside ui_qt/ that
        # the user-reported roi_logic.py bug forced us to audit too.
        "mufasa/mixins/config_reader.py":
            "v1-aware ConfigReader adapter — wraps TOML→INI "
            "translation internally; the configparser usage here "
            "is post-translation and is the canonical way to "
            "consume the synthetic INI",
        "mufasa/mixins/train_model_mixin.py":
            "takes an already-parsed config object as a method "
            "argument; doesn't read from disk itself",
        "mufasa/utils/toml_to_configparser.py":
            "synthesises a ConfigParser FROM TOML data — writes, "
            "not reads",
        "mufasa/utils/project_reconfigure.py":
            "known deferred item — INI-only project "
            "reconfiguration; flagged in session journal",
    }

    # Patch 122ai: scope expanded from mufasa/ui_qt/ only (122ah)
    # to ALL of mufasa/ (excluding the legacy Tk popups, which are
    # only reachable via SimBA.py and are out of scope for v1).
    # The user-reported roi_logic.py bug was outside the 122ah
    # scope and so wasn't caught; this widening prevents the
    # next instance from slipping through.
    AUDIT_ROOTS = [
        REPO_ROOT / "mufasa" / "ui_qt",
        REPO_ROOT / "mufasa" / "roi_tools",
        REPO_ROOT / "mufasa" / "utils",
        REPO_ROOT / "mufasa" / "mixins",
        REPO_ROOT / "mufasa" / "feature_extractors",
        REPO_ROOT / "mufasa" / "cli",
    ]

    leaks: list[tuple[str, int, str]] = []
    # Patch 122ai: heuristic refined. Bare '"General settings"'
    # appears legitimately in:
    #   * docstring usage examples (mufasa/utils/read_write.py)
    #   * the GENERAL_SETTINGS = "General settings" constant
    #     (mufasa/utils/enums.py)
    # Catching those as bugs would produce false positives. The
    # real bug shape is a configparser .get() / .has_section()
    # call that references the section by name — match that
    # specifically.
    import re
    BUG_PATTERN = re.compile(
        r"""\.(get|has_section|getint|getboolean|getfloat|"""
        r"""items)\s*\(\s*["']General\s+settings["']""",
    )
    for root in AUDIT_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            rel = str(p.relative_to(REPO_ROOT))
            if rel in LEGACY_BRANCH_WHITELIST:
                continue
            src = p.read_text()
            for i, line in enumerate(src.splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                if BUG_PATTERN.search(line):
                    leaks.append((rel, i, line.strip()))

    check(
        "No active backend module reads '[General settings]' "
        "outside the gated-legacy whitelist (scope: ui_qt, "
        "roi_tools, utils, mixins, feature_extractors, cli)",
        not leaks,
        detail=f"unexpected leaks: {leaks}" if leaks else "",
    )

    print(
        f"smoke_122ah_targeted_clips: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
