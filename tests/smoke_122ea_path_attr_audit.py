"""
tests/smoke_122ea_path_attr_audit.py
======================================

Patch 122ea: complete the per-attribute audit deferred by 122dy.

Context
-------
122dy ripped ConfigReader's ``_is_v1`` legacy branch, which folded
all path resolution into ``_resolve_v1_paths``. Three path
attributes weren't covered by that method and were left as
legacy-shaped fallbacks at the end of ``__init__``:

* ``annotated_frm_dir`` — "Extract labelled frames" feature.
* ``single_validation_video_save_dir`` — "Validate single model"
  workbench form's video outputs.
* ``data_table_path`` — live-data-table video renders (consumer
  is orphaned post-tk-removal but the attribute is kept).

The 122dy commit explicitly flagged this as a deferred per-
attribute audit. 122ea executes it.

What this patch landed
----------------------
``mufasa/mixins/config_reader.py``:
* Added three lines to ``_resolve_v1_paths``:
    self.annotated_frm_dir                = derived/annotated/
    self.single_validation_video_save_dir = derived/validation/
    self.data_table_path                  = derived/data_tables/
* Removed the three legacy-shaped fallback ``os.path.join``
  assignments from ``__init__``.
* Updated ``_resolve_v1_paths`` docstring to add 122ea to the
  history bullets and document the new attrs.

The consumers pick up the new locations transparently because
they all use ``os.makedirs(exist_ok=True)`` against the path
attribute they read — no per-consumer edits needed.

Out of scope (deferred to future patches):
* Docstring ``.ini`` example paths — ~364 occurrences across
  the codebase. Cosmetic; mechanical sweep needs its own
  focused patch.
* ``load_features_for_video`` / ``load_labels_for_video``
  internal legacy-CSV fallback branches. Bigger surgery than
  122dz's kwarg sweep; deserves its own patch.

Coverage
--------
ConfigReader changes:
1.  ``_resolve_v1_paths`` sets ``self.annotated_frm_dir``.
2.  ``_resolve_v1_paths`` sets
    ``self.single_validation_video_save_dir``.
3.  ``_resolve_v1_paths`` sets ``self.data_table_path``.
4.  The three attrs point at ``derived/`` paths (not legacy
    ``frames/output/`` paths) — substring check.
5.  ``ConfigReader.__init__`` no longer contains the legacy
    ``os.path.join(self.project_path, Paths.ANNOTATED_FRAMES_DIR.value)``
    pattern.
6.  Same for ``Paths.SINGLE_CLF_VALIDATION``.
7.  Same for ``Paths.DATA_TABLE``.
8.  ``_resolve_v1_paths`` docstring mentions 122ea.

Cross-patch invariants:
9.  122dz state preserved: ``load_machine_results_for_video``
    has no ``legacy_fallback`` parameter.
10. 122dy state preserved: ConfigReader rejects non-.toml.
11. 122dx state preserved: ``ui_qt/app.py`` still gone.
12. 122dw state preserved: ``cli/migrate_project.py`` still gone.
13. 122dv state preserved: no SkipOutlierCorrectionForm.
14. 122ds state preserved: SECTIONS DAG still validates.
15. Parse-clean across mufasa/**/*.py.
16. 122do baseline: no ``Optional[`` in non-docstring positions
    across mufasa/ui_qt/.
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


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _self_attrs_assigned(method: ast.FunctionDef) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    out.add(tgt.attr)
                if isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if (isinstance(elt, ast.Attribute)
                                and isinstance(elt.value, ast.Name)
                                and elt.value.id == "self"):
                            out.add(elt.attr)
    return out


def main() -> int:
    cr_path = REPO_ROOT / "mufasa" / "mixins" / "config_reader.py"
    cr_src = cr_path.read_text()
    cr_tree = ast.parse(cr_src)
    cls = _ast_find_class(cr_tree, "ConfigReader")
    assert cls is not None
    init_method = _ast_method(cls, "__init__")
    resolve_method = _ast_method(cls, "_resolve_v1_paths")
    assert init_method is not None
    assert resolve_method is not None

    resolve_attrs = _self_attrs_assigned(resolve_method)
    init_attrs = _self_attrs_assigned(init_method)
    init_src = ast.unparse(init_method)
    resolve_src = ast.unparse(resolve_method)

    # 1-3. Three attrs assigned in _resolve_v1_paths.
    check(
        "_resolve_v1_paths assigns self.annotated_frm_dir",
        "annotated_frm_dir" in resolve_attrs,
    )
    check(
        "_resolve_v1_paths assigns "
        "self.single_validation_video_save_dir",
        "single_validation_video_save_dir" in resolve_attrs,
    )
    check(
        "_resolve_v1_paths assigns self.data_table_path",
        "data_table_path" in resolve_attrs,
    )

    # 4. Values point at derived/ paths, not legacy frames/output/.
    # We pattern-check the source rather than try to evaluate the
    # method (which needs cv2 etc).
    has_v1_annotated = bool(re.search(
        r'self\.annotated_frm_dir\s*=\s*str\s*\(\s*derived\s*/',
        resolve_src,
    ))
    has_v1_validation = bool(re.search(
        r'self\.single_validation_video_save_dir\s*=\s*str\s*\(\s*\n?\s*derived\s*/',
        resolve_src,
    ))
    has_v1_data_table = bool(re.search(
        r'self\.data_table_path\s*=\s*str\s*\(\s*derived\s*/',
        resolve_src,
    ))
    check(
        "annotated_frm_dir / single_validation_video_save_dir / "
        "data_table_path all resolve to ``derived/<...>`` (v1 "
        "locations) in _resolve_v1_paths source",
        has_v1_annotated and has_v1_validation and has_v1_data_table,
        detail=(
            f"annotated={has_v1_annotated} "
            f"validation={has_v1_validation} "
            f"data_table={has_v1_data_table}"
        ),
    )

    # 5-7. The three legacy ``os.path.join(..., Paths.XXX.value)``
    # patterns are gone from __init__.
    has_legacy_annotated = (
        "Paths.ANNOTATED_FRAMES_DIR" in init_src
    )
    has_legacy_validation = (
        "Paths.SINGLE_CLF_VALIDATION" in init_src
    )
    has_legacy_data_table = (
        "Paths.DATA_TABLE.value" in init_src
        or "Paths.DATA_TABLE)" in init_src
    )
    check(
        "ConfigReader.__init__ no longer assigns "
        "annotated_frm_dir from Paths.ANNOTATED_FRAMES_DIR "
        "(legacy fallback gone)",
        not has_legacy_annotated,
    )
    check(
        "ConfigReader.__init__ no longer assigns "
        "single_validation_video_save_dir from "
        "Paths.SINGLE_CLF_VALIDATION (legacy fallback gone)",
        not has_legacy_validation,
    )
    check(
        "ConfigReader.__init__ no longer assigns data_table_path "
        "from Paths.DATA_TABLE (legacy fallback gone)",
        not has_legacy_data_table,
    )

    # 8. _resolve_v1_paths docstring mentions 122ea.
    resolve_doc = ast.get_docstring(resolve_method) or ""
    check(
        "_resolve_v1_paths docstring mentions 122ea (audit "
        "breadcrumb)",
        "122ea" in resolve_doc,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 9. 122dz state preserved.
    cio_src = (REPO_ROOT / "mufasa" / "utils"
               / "classification_io.py").read_text()
    cio_tree = ast.parse(cio_src)
    helper = None
    for node in ast.walk(cio_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "load_machine_results_for_video"):
            helper = node
            break
    assert helper is not None
    helper_params = (
        [a.arg for a in helper.args.args]
        + [a.arg for a in helper.args.kwonlyargs]
    )
    check(
        "122dz state preserved: load_machine_results_for_video "
        "still has no legacy_fallback parameter",
        "legacy_fallback" not in helper_params,
    )

    # 10. 122dy state preserved.
    check(
        "122dy state preserved: ConfigReader.__init__ still "
        "raises on a non-.toml config_path",
        "InvalidInputError" in cr_src
        and ".toml" in cr_src
        and "122dy" in cr_src,
    )

    # 11. 122dx state preserved.
    check(
        "122dx state preserved: mufasa/ui_qt/app.py still gone",
        not (REPO_ROOT / "mufasa" / "ui_qt" / "app.py").exists(),
    )

    # 12. 122dw state preserved.
    check(
        "122dw state preserved: mufasa/cli/migrate_project.py "
        "still gone",
        not (REPO_ROOT / "mufasa" / "cli"
             / "migrate_project.py").exists(),
    )

    # 13. 122dv state preserved.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_cleanup.py").read_text()
    check(
        "122dv state preserved: no SkipOutlierCorrectionForm",
        "class SkipOutlierCorrectionForm" not in pc_src,
    )

    # 14. SECTIONS DAG.
    try:
        from mufasa.section_provenance import SECTIONS
        sections_ok = len(SECTIONS) > 0
    except Exception:
        sections_ok = False
    check(
        "122ds state preserved: SECTIONS still imports + DAG "
        "validates",
        sections_ok,
    )

    # 15. Parse-clean.
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

    # 16. 122do baseline.
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
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122ea_path_attr_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
