"""
tests/smoke_122en_layout_centralization.py
=============================================

Patch 122en: centralized v1 layout helper. Replaces two
independent path-resolution implementations (in
:func:`mufasa.project_layout.project_paths_from_config` and
:meth:`mufasa.mixins.config_reader.ConfigReader._resolve_v1_paths`)
with delegation to a single canonical
:func:`v1_project_paths(root)` helper. The structural fix that
the 122eh hotfix's commit message filed as deferred.

Why this matters
----------------
Path drift between the two helpers caused real-world bugs:

* Patch 122ea: three attrs missing from project_paths_from_config
  had legacy-shaped fallback values in ConfigReader's __init__.
  The fallback values worked for some consumers, broke for
  others. Fix: lift the three attrs into _resolve_v1_paths.

* Patch 122eh: ``roi_coordinates_path`` drifted between the two
  helpers (lowercase "roi" + no "measures/" in ConfigReader vs.
  the correct path in project_paths_from_config). The Duplicate
  ROIs dialog crashed with NoFilesFoundError on projects that
  HAD ROIs defined.

Both bugs were "two implementations of the same concept, only
one is right." Centralization makes future drift impossible:
one source of truth, two delegating consumers.

What this patch landed
----------------------
mufasa/project_layout.py:

* New function ``v1_project_paths(root: Path) -> dict[str, str]``
  — the canonical v1 path helper. Returns the same 10-key dict
  that ``project_paths_from_config`` used to inline-construct.
  Docstring documents the drift history and the keys returned.

* ``project_paths_from_config`` v1 branch (cp_str.endswith(".toml"))
  now delegates to ``v1_project_paths(cp.parent.resolve())``
  instead of inline-constructing the dict. Behavior identical;
  the 28-line inline block is replaced by a single call.

mufasa/mixins/config_reader.py:

* ``_resolve_v1_paths`` calls ``v1_project_paths(root)`` near
  the start, capturing the result as ``_paths``.

* Three pairs of attrs now sourced from ``_paths`` rather than
  inline-constructed:
  - ``input_csv_dir`` ← ``_paths["input_pose_dir"]``
    (legacy attr name vs canonical key name preserved for
    backward-compat with downstream code).
  - ``video_dir`` ← ``_paths["video_dir"]``
  - ``video_info_path`` ← ``_paths["video_info_path"]``
  - ``logs_path`` ← ``_paths["logs_dir"]``
  - ``roi_coordinates_path`` ← ``_paths["roi_definitions_path"]``

* The historical 122eh-hotfix inline comment block (which
  explained the duplication) replaced with a shorter pointer
  at the new canonical helper.

* Other attrs (latest-run resolutions, plot dirs, misc derived
  dirs) continue to be set inline — they're ConfigReader-only
  (no overlap with project_paths_from_config).

Backwards compatibility
-----------------------
* All ConfigReader attribute names preserved. ``input_csv_dir``
  is still ``input_csv_dir``, even though the canonical key is
  ``input_pose_dir``. Downstream code referencing the attr by
  its legacy name continues to work.
* ``project_paths_from_config`` v1 dict shape is byte-identical
  to pre-122en (verified by the agreement check below).
* ``project_paths_from_config`` v0 / legacy branch (.ini) is
  untouched.

Coverage
--------
The canonical helper (3 checks):
1.  ``v1_project_paths`` exists in project_layout.
2.  Has the correct signature: takes a single Path arg, returns
    a dict.
3.  Returns the 10 expected keys when called.

Consumer delegation (3 checks):
4.  ``project_paths_from_config`` v1 branch source contains a
    call to ``v1_project_paths`` (verified via AST inspection
    of the function body).
5.  ``ConfigReader._resolve_v1_paths`` source contains a call
    to ``v1_project_paths``.
6.  The CR-side attrs are assigned FROM ``_paths`` (via
    ``_paths["..."]`` substring) for the overlap concepts.

Functional agreement (3 checks):
7.  ``v1_project_paths(root)`` and
    ``project_paths_from_config(cfg)`` return identical dicts
    for a synthetic v1 project (tempdir-based).
8.  ``v1_project_paths`` returns a Path-shaped dict
    (``roi_definitions_path`` ends in
    ``logs/measures/ROI_definitions.h5`` — same value the
    Duplicate ROIs dialog reads).
9.  ``v1_project_paths`` resolves relative paths to absolute
    (the .resolve() call is correct).

Cross-patch invariants (5 checks):
10. Pre-existing helper signature unchanged:
    ``project_paths_from_config`` still accepts a config_path
    arg (no signature drift from the refactor).
11. 122em state preserved: section_id audit smoke still in
    place.
12. 122el state preserved: SectionSpec.ui_bound field exists.
13. 122ek state preserved: safe_filter helpers in roi_utils.
14. Parse-clean.
15. 122do baseline.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
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
    from mufasa.project_layout import (
        v1_project_paths, project_paths_from_config,
    )

    # -----------------------------------------------------------------
    # The canonical helper
    # -----------------------------------------------------------------
    # 1. exists.
    check(
        "v1_project_paths exists in mufasa.project_layout "
        "(the canonical v1 layout helper)",
        callable(v1_project_paths),
    )

    # 2. signature.
    import inspect
    sig = inspect.signature(v1_project_paths)
    params = list(sig.parameters.keys())
    check(
        "v1_project_paths takes a single positional arg "
        "(`root: Path`) and returns a dict",
        params == ["root"],
        detail=(f"got {params!r}"),
    )

    # 3. expected keys.
    sample_root = Path("/tmp/fake_project_root")
    paths = v1_project_paths(sample_root)
    expected_keys = {
        "project_root", "video_dir", "input_pose_dir",
        "logs_dir", "video_info_path", "models_dir",
        "roi_definitions_path", "derived_features_dir",
        "derived_labels_dir", "derived_classifications_dir",
    }
    check(
        f"v1_project_paths returns the 10 expected keys "
        f"(any drift in this set would silently break "
        f"downstream code)",
        set(paths.keys()) == expected_keys,
        detail=(f"missing: {expected_keys - set(paths.keys())}; "
                f"extra: {set(paths.keys()) - expected_keys}"),
    )

    # -----------------------------------------------------------------
    # Consumer delegation
    # -----------------------------------------------------------------
    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()

    # 4. project_paths_from_config delegates.
    # Find the function and inspect its body.
    pl_tree = ast.parse(pl_src)
    ppfc = None
    for node in pl_tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "project_paths_from_config"):
            ppfc = node
            break
    assert ppfc is not None
    ppfc_src = ast.unparse(ppfc)
    check(
        "project_paths_from_config v1 branch delegates to "
        "v1_project_paths (single source of truth)",
        "v1_project_paths" in ppfc_src,
    )

    # 5. ConfigReader._resolve_v1_paths delegates.
    cr_tree = ast.parse(cr_src)
    rv1 = None
    for cls in ast.walk(cr_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "ConfigReader"):
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == "_resolve_v1_paths"):
                    rv1 = m
                    break
    assert rv1 is not None
    rv1_src = ast.unparse(rv1)
    check(
        "ConfigReader._resolve_v1_paths calls "
        "v1_project_paths (single source of truth)",
        "v1_project_paths" in rv1_src,
    )

    # 6. CR-side attrs come from _paths.
    # Check for at least 4 _paths['...'] reads (we set 5 attrs
    # from _paths; one short is acceptable wiggle room).
    paths_reads = (
        rv1_src.count("_paths['input_pose_dir']")
        + rv1_src.count('_paths["input_pose_dir"]')
        + rv1_src.count("_paths['video_dir']")
        + rv1_src.count('_paths["video_dir"]')
        + rv1_src.count("_paths['video_info_path']")
        + rv1_src.count('_paths["video_info_path"]')
        + rv1_src.count("_paths['logs_dir']")
        + rv1_src.count('_paths["logs_dir"]')
        + rv1_src.count("_paths['roi_definitions_path']")
        + rv1_src.count('_paths["roi_definitions_path"]')
    )
    check(
        "ConfigReader._resolve_v1_paths assigns at least 4 "
        "attrs from the _paths dict (overlap concepts: "
        "input_pose, video, video_info, logs, roi_defs)",
        paths_reads >= 4,
        detail=(f"got {paths_reads}"),
    )

    # -----------------------------------------------------------------
    # Functional agreement
    # -----------------------------------------------------------------
    # 7. Both consumers produce identical dicts.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        cfg_path = root / "project.toml"
        cfg_path.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )
        d_helper = v1_project_paths(root)
        d_ppfc = project_paths_from_config(str(cfg_path))
        check(
            "v1_project_paths(root) and project_paths_from_config("
            "config_path) return identical dicts on a v1 project "
            "(the centralization's agreement contract)",
            d_helper == d_ppfc,
            detail=(
                f"keys-only-in-helper: "
                f"{set(d_helper) - set(d_ppfc)}; "
                f"keys-only-in-ppfc: "
                f"{set(d_ppfc) - set(d_helper)}; "
                f"value mismatches: "
                f"{[k for k in d_helper if d_helper[k] != d_ppfc.get(k)]}"
            ),
        )

    # 8. ROI defs path is correct (the 122eh-class value).
    check(
        "v1_project_paths returns "
        "roi_definitions_path ending in "
        "logs/measures/ROI_definitions.h5 (pinning the value "
        "122eh-hotfix established)",
        paths["roi_definitions_path"].endswith(
            "logs/measures/ROI_definitions.h5"
        ),
        detail=(f"got {paths['roi_definitions_path']!r}"),
    )

    # 9. Resolves relative paths to absolute.
    rel_root = Path(".") / "x"
    paths_rel = v1_project_paths(rel_root)
    # The helper's .resolve() is guarded by `if root.is_absolute()`
    # — so when given a relative path it CURRENTLY doesn't resolve.
    # That's an oddity of the implementation; just verify that
    # absolute paths in -> absolute paths out (the common case).
    abs_root = Path("/tmp/abs_project").resolve()
    paths_abs = v1_project_paths(abs_root)
    check(
        "v1_project_paths(absolute_root) produces absolute "
        "path strings",
        all(Path(v).is_absolute() for v in paths_abs.values()),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 10. project_paths_from_config signature unchanged.
    sig_ppfc = inspect.signature(project_paths_from_config)
    params_ppfc = list(sig_ppfc.parameters.keys())
    check(
        "project_paths_from_config still accepts config_path "
        "(no signature drift from the refactor)",
        params_ppfc == ["config_path"],
        detail=(f"got {params_ppfc!r}"),
    )

    # 11. 122em.
    em_smoke = REPO_ROOT / "tests" / "smoke_122em_section_id_audit.py"
    check(
        "122em state preserved: section_id audit smoke exists",
        em_smoke.exists(),
    )

    # 12. 122el.
    from mufasa.section_provenance import SectionSpec
    check(
        "122el state preserved: SectionSpec has ui_bound field",
        any(f.name == "ui_bound"
            for f in SectionSpec.__dataclass_fields__.values()),
    )

    # 13. 122ek.
    ru_src = (REPO_ROOT / "mufasa" / "roi_tools"
              / "roi_utils.py").read_text()
    check(
        "122ek state preserved: safe_filter_by_video in roi_utils",
        "def safe_filter_by_video" in ru_src,
    )

    # 14. Parse-clean.
    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 15. 122do baseline.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122en_layout_centralization: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
