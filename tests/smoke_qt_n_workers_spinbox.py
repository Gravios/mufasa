"""Tests for the n_workers spinbox in FeatureSubsetExtractorForm.

Verifies that the Qt form exposes n_workers as a QSpinBox, threads
it through collect_args, and passes it to the backend. PySide6
imports are heavy and not always available in the sandbox, so we
do AST inspection rather than instantiating the widgets.

    PYTHONPATH=. python tests/smoke_qt_n_workers_spinbox.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path("mufasa/ui_qt/forms/features.py").read_text()
    tree = ast.parse(src)

    # ------------------------------------------------------------------ #
    # Case 1: QSpinBox imported
    # ------------------------------------------------------------------ #
    qspinbox_imported = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "PySide6.QtWidgets":
                for alias in node.names:
                    if alias.name == "QSpinBox":
                        qspinbox_imported = True
    assert qspinbox_imported, (
        "QSpinBox must be imported from PySide6.QtWidgets"
    )

    # ------------------------------------------------------------------ #
    # Case 2: build() creates a QSpinBox and assigns to self.n_workers
    # ------------------------------------------------------------------ #
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureSubsetExtractorForm":
            cls = node
            break
    assert cls is not None, "FeatureSubsetExtractorForm class missing"

    build = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build":
            build = node
            break
    assert build is not None, "build() method missing"

    build_src = ast.unparse(build)
    assert "QSpinBox" in build_src, (
        "build() must instantiate a QSpinBox"
    )
    assert "self.n_workers" in build_src, (
        "build() must assign the spinbox to self.n_workers"
    )
    assert ".setMinimum(1)" in build_src, (
        "n_workers spinbox should have minimum=1 (1 = sequential)"
    )
    assert ".setValue(1)" in build_src, (
        "n_workers spinbox should default to 1 (preserves "
        "byte-equivalent sequential behavior)"
    )

    # ------------------------------------------------------------------ #
    # Case 3: maximum is dynamic (uses os.cpu_count) so it adapts to
    # the workstation
    # ------------------------------------------------------------------ #
    assert "cpu_count" in build_src, (
        "Maximum worker count should be tied to os.cpu_count() so "
        "the spinbox cap matches the actual CPU"
    )

    # ------------------------------------------------------------------ #
    # Case 4: collect_args includes n_workers in the returned dict
    # ------------------------------------------------------------------ #
    collect = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "collect_args":
            collect = node
            break
    assert collect is not None
    collect_src = ast.unparse(collect)
    assert "'n_workers'" in collect_src or '"n_workers"' in collect_src, (
        "collect_args must include 'n_workers' key in returned dict"
    )
    assert "self.n_workers.value()" in collect_src, (
        "collect_args must read self.n_workers.value()"
    )

    # ------------------------------------------------------------------ #
    # Case 5: target() takes n_workers and passes it to the backend
    # ------------------------------------------------------------------ #
    target = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "target":
            target = node
            break
    assert target is not None
    # All target args are keyword-only; check they're declared
    target_arg_names = (
        [a.arg for a in target.args.args]
        + [a.arg for a in target.args.kwonlyargs]
    )
    assert "n_workers" in target_arg_names, (
        "target() must accept n_workers parameter"
    )

    target_src = ast.unparse(target)
    assert "n_workers=n_workers" in target_src, (
        "target() must pass n_workers through to FeatureSubsetsCalculator"
    )

    # ------------------------------------------------------------------ #
    # Case 6: there's a tooltip that mentions verification
    # ------------------------------------------------------------------ #
    assert "setToolTip" in build_src, (
        "n_workers spinbox should have a setToolTip call"
    )
    assert "verify" in build_src.lower(), (
        "tooltip should mention verifying parallel output"
    )

    # ------------------------------------------------------------------ #
    # Case 7: there's a visible hint label below the spinbox (not
    # just the tooltip — verification matters enough to be visible
    # in the form, not buried)
    # ------------------------------------------------------------------ #
    assert "smoke_feature_parallel_verify" in build_src, (
        "form should reference the verification script so users "
        "know where to look"
    )

    print("smoke_qt_n_workers_spinbox: 7/7 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
