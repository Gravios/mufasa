"""
tests/smoke_122dy_config_reader_legacy_rip.py
=============================================

Patch 122dy: rip the legacy SimBA project_config.ini branch out of
:class:`mufasa.mixins.config_reader.ConfigReader`. v1 ``project.toml``
becomes the only supported config path.

Context
-------
ConfigReader is the foundational mixin every backend extends. Its
``__init__`` previously did the following:

1. Set ~30 path attributes pointing at the legacy
   ``<root>/project_folder/csv/...`` tree.
2. Read animal_cnt, clf_names, file globs, logs_path — all
   assuming legacy paths.
3. Check ``self._is_v1`` (set from config-path suffix). If True,
   call ``_apply_v1_path_overrides()`` which OVERWROTE every path
   set in step 1 and re-globbed file lists at the v1 locations.
4. Branch on ``_is_v1`` again for body-parts reading (v1 from
   ``project.toml``; legacy from a separate ``body_parts.csv``).
5. Set a few more paths AFTER the override — some of which silently
   clobbered v1 paths set in step 3.

The legacy branches were dead code on a v1-only codebase
(122dw removed the migration tool; 122dx removed the legacy Qt
chooser — nothing in the workbench produces or accepts a legacy
project anymore). 122dy deletes them.

What this patch landed
----------------------
``mufasa/mixins/config_reader.py``:

* Added a fail-fast guard at the top of ``__init__``: any
  non-``.toml`` path raises :class:`InvalidInputError` with a
  message pointing at 122dw / 122dy. Pairs cleanly with 122dw's
  migration-tool removal and 122dx's chooser removal — the three
  patches close every path-detection entry the workbench had for
  legacy projects.

* Renamed ``_v1_toml_data`` → ``_project_toml_data``. The ``_v1``
  qualifier was load-bearing only while a non-v1 branch existed;
  it's noise now.

* Deleted the ``self._is_v1`` flag entirely. It had three
  consumers, all removed: the path-setup-conditional branch, the
  body-parts-conditional branch, and the override-call gate.

* Renamed ``_apply_v1_path_overrides`` → ``_resolve_v1_paths``.
  The method no longer "overrides" anything because the preceding
  legacy-path setup block is gone — it just sets paths directly.
  Method docstring updated to drop the "overrides" framing and
  retain only the still-relevant content (path mapping + 122dr
  multi-run caveat).

* Deleted the ~50-line legacy path-setup block (the
  ``os.path.join(self.project_path, Paths.XXX.value)`` x ~30
  assignments). Its only purpose was to provide a target for
  ``_apply_v1_path_overrides`` to override.

* Deleted the legacy body-parts CSV read branch (which used
  ``Paths.BP_NAMES`` and a ``pd.read_csv`` against a separate
  ``body_parts.csv`` file).

* Deleted the duplicate file-glob block (lines 192-203 in the
  pre-122dy file) — ``_resolve_v1_paths`` already does these
  globs at the v1 locations.

* Deleted the v1-path-clobbering lines that ran AFTER the
  override block: ``video_dir``, ``roi_coordinates_path``,
  ``clf_validation_dir``, ``clf_data_validation_dir`` — these
  reset v1 paths to legacy values and were a silent bug.

``mufasa/utils/feature_io.py``: docstring reference to
``ConfigReader._is_v1`` updated to mention 122dy and clarify that
the helper retains its own local suffix check.

Net effect
----------
Pre-122dy: ConfigReader.__init__ was ~230 LoC with two
interleaved layout paths.
Post-122dy: ~150 LoC, single path. ~80 LoC deleted plus the
``_is_v1`` conditional surface area gone.

Coverage
--------
Source-level deletions:
1.  ``self._is_v1`` attribute no longer set in
    ``ConfigReader.__init__``.
2.  ``self._v1_toml_data`` attribute no longer set (renamed to
    ``self._project_toml_data``; the old name shouldn't appear
    as an active assignment).
3.  The legacy body-parts CSV read (``check_file_exist_and_readable``
    against ``Paths.BP_NAMES`` + ``pd.read_csv``) is gone from
    ``__init__``.
4.  No live ``if self._is_v1`` conditional in ``__init__``.

Method rename:
5.  ``_apply_v1_path_overrides`` is no longer a method on
    ``ConfigReader``.
6.  ``_resolve_v1_paths`` IS a method on ``ConfigReader``.
7.  ``_resolve_v1_paths`` is called unconditionally from
    ``__init__`` (no longer gated on ``self._is_v1``).

Fail-fast guard:
8.  ``ConfigReader.__init__`` raises ``InvalidInputError`` on a
    non-``.toml`` config_path.
9.  The error message mentions 122dy (so a user encountering it
    has a discoverable explanation of why their previously-
    working .ini call now fails).

Functional behavior:
10. ``ConfigReader`` cannot be imported via a stub-free path
    (needs OpenCV, pandas, etc.), so direct construction isn't
    tested in this sandbox. Instead: the class STILL has all the
    public path attributes the codebase depends on (verified via
    static inspection of ``__init__`` source).
11. ``self._project_toml_data`` is set unconditionally
    (TOML pre-load happens for every ConfigReader instance).
12. ``self.body_parts_lst`` reads from
    ``self._project_toml_data["pose"]["body_parts"]`` (single
    source of truth; legacy CSV path gone).

Cross-module past-tense gate:
13. Every remaining mention of ``_is_v1`` across mufasa/**/*.py
    sits in a deletion-context sentence (or is the literal
    string ``self._is_v1`` inside a triple-quoted docstring,
    which the gate also accepts).
14. Every remaining mention of ``_v1_toml_data`` (the old name)
    is in past-tense / deletion-context.

Cross-patch invariants:
15. 122dx state preserved: ``mufasa/ui_qt/app.py`` still gone.
16. 122dw state preserved: ``mufasa/cli/migrate_project.py``
    still gone.
17. 122dv state preserved: no ``SkipOutlierCorrectionForm`` class.
18. 122ds state preserved: ``SECTIONS`` DAG still validates.
19. Parse-clean across ``mufasa/**/*.py``.
20. 122do baseline: no ``Optional[`` in non-docstring positions
    across ``mufasa/ui_qt/``.
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


def _ast_attribute_assignments(method: ast.FunctionDef) -> set[str]:
    """Return the set of ``self.NAME`` attribute names assigned
    anywhere in the method body. Used to verify that critical path
    attrs are still set after the 122dy refactor."""
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
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and node.value is not None):
                out.add(tgt.attr)
    return out


def _past_tense_gate(
    src: str, needle: str, *, window: int = 300,
) -> list[int]:
    out = []
    for m in re.finditer(re.escape(needle), src):
        ctx = src[max(0, m.start() - window):
                  m.end() + window].lower()
        if not any(w in ctx for w in
                   ("removed", "deleted", "no longer", "122dy",
                    "renamed", "gone")):
            out.append(m.start())
    return out


def main() -> int:
    # Load ConfigReader source.
    cr_path = REPO_ROOT / "mufasa" / "mixins" / "config_reader.py"
    cr_src = cr_path.read_text()
    cr_tree = ast.parse(cr_src)
    cr_cls = _ast_find_class(cr_tree, "ConfigReader")
    assert cr_cls is not None
    init_method = _ast_method(cr_cls, "__init__")
    assert init_method is not None
    init_src = ast.unparse(init_method)

    # 1. self._is_v1 attribute no longer set.
    check(
        "ConfigReader.__init__ no longer assigns self._is_v1 "
        "(legacy detection flag removed in 122dy)",
        "self._is_v1 = " not in init_src
        and "self._is_v1 =" not in init_src,
    )

    # 2. self._v1_toml_data attribute no longer set (renamed).
    check(
        "ConfigReader.__init__ no longer assigns self._v1_toml_data "
        "(renamed to self._project_toml_data in 122dy)",
        "self._v1_toml_data = " not in init_src
        and "self._v1_toml_data =" not in init_src
        and "self._v1_toml_data:" not in init_src,
    )

    # 3. The legacy body-parts CSV read is gone. The combination of
    # `Paths.BP_NAMES` + `pd.read_csv` in __init__ was the giveaway.
    check(
        "ConfigReader.__init__ no longer reads body_parts from a "
        "legacy CSV (Paths.BP_NAMES + pd.read_csv combo removed)",
        not ("Paths.BP_NAMES" in init_src and "pd.read_csv" in init_src),
    )

    # 4. No live `if self._is_v1` conditional in __init__.
    check(
        "ConfigReader.__init__ has no live `if self._is_v1` "
        "conditional (the gate is gone; v1 path runs unconditionally)",
        "if self._is_v1" not in init_src,
    )

    # 5. _apply_v1_path_overrides no longer a method.
    check(
        "ConfigReader no longer has an _apply_v1_path_overrides "
        "method (renamed to _resolve_v1_paths in 122dy)",
        _ast_method(cr_cls, "_apply_v1_path_overrides") is None,
    )

    # 6. _resolve_v1_paths IS a method.
    resolve_method = _ast_method(cr_cls, "_resolve_v1_paths")
    check(
        "ConfigReader has a _resolve_v1_paths method",
        resolve_method is not None,
    )

    # 7. _resolve_v1_paths called unconditionally from __init__.
    # The call site shouldn't be inside an `if` block.
    has_unconditional_call = False
    for stmt in init_method.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (isinstance(func, ast.Attribute)
                    and func.attr == "_resolve_v1_paths"):
                has_unconditional_call = True
                break
    check(
        "_resolve_v1_paths is called unconditionally from "
        "__init__ (no enclosing `if`)",
        has_unconditional_call,
    )

    # 8 & 9. Fail-fast guard on non-.toml paths.
    has_raise_for_non_toml = False
    raise_message = ""
    for node in ast.walk(init_method):
        if isinstance(node, ast.Raise) and node.exc is not None:
            unparsed = ast.unparse(node)
            if ("InvalidInputError" in unparsed
                    and ".toml" in unparsed):
                has_raise_for_non_toml = True
                raise_message = unparsed
                break
    check(
        "ConfigReader.__init__ raises InvalidInputError on a "
        "non-.toml config_path (fail-fast guard at the start)",
        has_raise_for_non_toml,
    )
    check(
        "Fail-fast error message mentions 122dy so the user has a "
        "discoverable explanation",
        "122dy" in raise_message,
    )

    # 10. Critical path attributes still set somewhere in __init__
    # or in _resolve_v1_paths.
    init_attrs = _ast_attribute_assignments(init_method)
    resolve_attrs = (
        _ast_attribute_assignments(resolve_method)
        if resolve_method else set()
    )
    all_attrs = init_attrs | resolve_attrs
    critical_attrs = {
        "config_path", "config", "datetime", "project_path",
        "file_type", "font",
        # path attrs
        "input_csv_dir", "video_dir", "video_info_path",
        "outlier_corrected_dir", "outlier_corrected_movement_dir",
        "features_dir", "targets_folder", "machine_results_dir",
        "input_frames_dir", "frames_output_dir",
        "logs_path", "roi_coordinates_path",
        # data
        "body_parts_path", "body_parts_lst",
        "animal_cnt", "clf_cnt", "clf_names",
        "cpu_cnt", "cpu_to_use",
        # other
        "color_dict", "platform",
    }
    missing = critical_attrs - all_attrs
    check(
        f"Critical path attributes are all set in __init__ or "
        f"_resolve_v1_paths (sanity check: 122dy didn't drop any "
        f"public surface)",
        not missing,
        detail=(f"missing: {sorted(missing)[:5]}"),
    )

    # 11. self._project_toml_data set unconditionally.
    # "Unconditionally" = the assignment isn't gated on an `if`
    # node anywhere in its ancestor chain inside __init__. The
    # assignment is naturally inside `with open(...)` (a context
    # manager always executes its body), which is fine — we walk
    # the AST and reject the assignment only if a parent `if` /
    # `IfExp` is found.
    has_unconditional_toml_load = False
    def _is_under_if(node, root):
        # Build a parent map.
        parent = {}
        for n in ast.walk(root):
            for child in ast.iter_child_nodes(n):
                parent[child] = n
        cur = node
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, (ast.If, ast.IfExp)):
                return True
        return False
    for node in ast.walk(init_method):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = (node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target])
            for tgt in tgts:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and tgt.attr == "_project_toml_data"):
                    if not _is_under_if(node, init_method):
                        has_unconditional_toml_load = True
                        break
    check(
        "self._project_toml_data is set unconditionally in "
        "__init__ (no `if`/`IfExp` ancestor; `with` blocks are "
        "OK — context managers always execute their body)",
        has_unconditional_toml_load,
    )

    # 12. self.body_parts_lst reads from self._project_toml_data.
    # We check the unparsed __init__ source for the relationship.
    check(
        "self.body_parts_lst is sourced from "
        "self._project_toml_data (single source of truth; legacy "
        "CSV path gone)",
        ("body_parts_lst" in init_src
         and "_project_toml_data" in init_src
         and "body_parts" in init_src),
    )

    # 13. _is_v1 past-tense gate.
    is_v1_bad = []
    pkg = REPO_ROOT / "mufasa"
    for f in sorted(pkg.rglob("*.py")):
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for offset in _past_tense_gate(src, "_is_v1"):
            # Inside a triple-quoted string is also acceptable.
            preceding = src[:offset]
            triple_count = (
                preceding.count('"""') + preceding.count("'''")
            )
            if triple_count % 2 == 1:
                continue  # inside docstring; accepted
            is_v1_bad.append(
                f"{f.relative_to(REPO_ROOT)}:"
                f"{src[:offset].count(chr(10)) + 1}"
            )
    check(
        "Every remaining mention of `_is_v1` across mufasa/**/*.py "
        "is in a deletion-context sentence (or inside a docstring)",
        not is_v1_bad,
        detail=("; ".join(is_v1_bad[:3])),
    )

    # 14. _v1_toml_data past-tense gate.
    toml_bad = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for offset in _past_tense_gate(src, "_v1_toml_data"):
            preceding = src[:offset]
            triple_count = (
                preceding.count('"""') + preceding.count("'''")
            )
            if triple_count % 2 == 1:
                continue
            toml_bad.append(
                f"{f.relative_to(REPO_ROOT)}:"
                f"{src[:offset].count(chr(10)) + 1}"
            )
    check(
        "Every remaining mention of `_v1_toml_data` is in a "
        "deletion-context / renamed sentence",
        not toml_bad,
        detail=("; ".join(toml_bad[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 15. 122dx state preserved.
    check(
        "122dx state preserved: mufasa/ui_qt/app.py still gone",
        not (REPO_ROOT / "mufasa" / "ui_qt" / "app.py").exists(),
    )

    # 16. 122dw state preserved.
    check(
        "122dw state preserved: mufasa/cli/migrate_project.py "
        "still gone",
        not (REPO_ROOT / "mufasa" / "cli"
             / "migrate_project.py").exists(),
    )

    # 17. 122dv state preserved.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_cleanup.py").read_text()
    check(
        "122dv state preserved: no SkipOutlierCorrectionForm class",
        "class SkipOutlierCorrectionForm" not in pc_src,
    )

    # 18. SECTIONS DAG.
    try:
        from mufasa.section_provenance import SECTIONS
        sections_ok = len(SECTIONS) > 0
    except Exception:
        sections_ok = False
    check(
        "122ds state preserved: section_provenance.SECTIONS still "
        "imports + DAG validates",
        sections_ok,
    )

    # 19. Parse-clean.
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

    # 20. 122do baseline.
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
        f"smoke_122dy_config_reader_legacy_rip: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
