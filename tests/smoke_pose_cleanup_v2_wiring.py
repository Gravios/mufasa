"""Smoke test for patch-121 wiring of the Kalman v2 smoother
into the Pose Cleanup workbench page.

PySide6 isn't available in the sandbox, so we verify the
wiring at the AST/source level: the form class is defined in
the right module, listed in __all__, and the page builder
references it.

    PYTHONPATH=. python tests/smoke_pose_cleanup_v2_wiring.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _check_form_module() -> None:
    """The KalmanV2SmoothingForm class must exist in
    mufasa/ui_qt/forms/pose_cleanup.py and be exported.
    """
    src_path = Path("mufasa/ui_qt/forms/pose_cleanup.py")
    src = src_path.read_text()
    tree = ast.parse(src)

    # Find class def
    classes = {
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    assert "KalmanV2SmoothingForm" in classes, (
        f"KalmanV2SmoothingForm not defined in {src_path}; "
        f"found {sorted(classes)}"
    )

    # Find __all__ assignment
    all_targets = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            if isinstance(node.value, ast.List):
                all_targets = [
                    elt.value for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                ]
    assert "KalmanV2SmoothingForm" in all_targets, (
        f"KalmanV2SmoothingForm not in __all__; "
        f"got {all_targets}"
    )

    # The class should subclass OperationForm
    target_class = None
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.ClassDef)
            and n.name == "KalmanV2SmoothingForm"
        ):
            target_class = n
            break
    assert target_class is not None
    base_names = [
        b.id for b in target_class.bases
        if isinstance(b, ast.Name)
    ]
    assert "OperationForm" in base_names, (
        f"KalmanV2SmoothingForm should subclass OperationForm; "
        f"got {base_names}"
    )

    # Class must define build, collect_args, target methods
    method_names = {
        n.name for n in target_class.body
        if isinstance(n, ast.FunctionDef)
    }
    for required in ("build", "collect_args", "target"):
        assert required in method_names, (
            f"KalmanV2SmoothingForm missing required method "
            f"{required!r}; defined: {sorted(method_names)}"
        )

    # The target() method must call smooth_pose_v2
    target_method = next(
        n for n in target_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "target"
    )
    target_src = ast.unparse(target_method)
    assert "smooth_pose_v2" in target_src, (
        "target() should invoke smooth_pose_v2 from the v2 "
        "smoother module"
    )
    # And reference all three feature flags so we know the
    # patch-121 extensions are actually plumbed
    for flag in (
        "with_drift",
        "orientation_drift_segments",
        "const_accel_segments",
    ):
        assert flag in target_src, (
            f"target() should reference layout flag {flag!r} "
            f"so the patch-121 extensions reach the smoother"
        )

    # Patch 121g: target() must dispatch on a 'mode' arg with
    # both 'train' and 'load' branches, and call smooth_pose_v2
    # with save_model in train mode and load_model in load mode.
    assert "mode" in target_method.args.kwonlyargs[0].arg or any(
        a.arg == "mode" for a in target_method.args.kwonlyargs
    ), (
        "target() should accept a keyword 'mode' arg (train/load)"
    )
    assert "save_model" in target_src, (
        "target() should pass save_model in train mode"
    )
    assert "load_model" in target_src, (
        "target() should pass load_model in load mode"
    )

    # collect_args must validate both modes (mode_train + mode_load
    # widgets, training subset, model file path)
    collect_method = next(
        n for n in target_class.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "collect_args"
    )
    collect_src = ast.unparse(collect_method)
    for required in (
        "mode_train", "load_model_path",
        "training_files", "em_max_iter",
        "em_tol", "em_damping", "em_aggregation",
    ):
        assert required in collect_src, (
            f"collect_args should reference {required!r} "
            f"to surface the new train/load UI fields"
        )

    # build() must construct the new mode/training/load widgets
    build_method = next(
        n for n in target_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "build"
    )
    build_src = ast.unparse(build_method)
    for required in (
        "mode_train", "mode_load",
        "train_group", "load_group",
        "load_model_path", "save_model_path",
        "train_file_list", "em_tol", "em_damping",
        "em_aggregation", "warm_start_sigma",
        "use_perspective", "use_validation",
    ):
        assert required in build_src, (
            f"build() should construct widget {required!r} "
            f"for the train/load UI"
        )

    # Patch 121h: a module-level helper resolves the default
    # model dir (~/.config/mufasa/models/), and the form's
    # save/load paths flow through it.
    helpers = {
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "_default_model_dir" in helpers, (
        "module should define _default_model_dir() helper "
        "for the standard ~/.config/mufasa/models/ location"
    )
    helper = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_default_model_dir"
    )
    helper_src = ast.unparse(helper)
    assert ".config" in helper_src and "mufasa" in helper_src, (
        "_default_model_dir should resolve to "
        "~/.config/mufasa/models/"
    )
    assert "models" in helper_src
    # The form's save/load callbacks should use the helper
    full_class_src = ast.unparse(target_class)
    assert (
        full_class_src.count("_default_model_dir") >= 2
    ), (
        "form should call _default_model_dir() in both "
        "save and load workflows"
    )

    # Patch 122b: dual-save wiring. The form imports the three
    # project_layout helpers and the target() method calls
    # mirror + import on both train and load paths.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "project_layout" in node.module:
                for alias in node.names:
                    imported.add(alias.name)
    for name in (
        "import_model_into_project",
        "mirror_model_to_global_cache",
        "resolve_v1_project_root",
    ):
        assert name in imported, (
            f"form should import {name!r} from mufasa.project_layout "
            f"for the patch-122b dual-save flow; got {sorted(imported)}"
        )
    # The target() body should mention each helper at least once.
    for name in (
        "mirror_model_to_global_cache",
        "import_model_into_project",
        "resolve_v1_project_root",
    ):
        assert name in target_src, (
            f"target() should call {name!r} so dual-save runs "
            f"on every save/load path"
        )


def _check_page_registration() -> None:
    """The pose_cleanup_page builder must add a section that
    instantiates KalmanV2SmoothingForm.
    """
    src_path = Path("mufasa/ui_qt/pages/pose_cleanup_page.py")
    src = src_path.read_text()
    assert "KalmanV2SmoothingForm" in src, (
        f"{src_path} doesn't reference KalmanV2SmoothingForm; "
        f"the new section won't be visible in the workbench"
    )
    # Also import line
    tree = ast.parse(src)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    assert "KalmanV2SmoothingForm" in imported_names, (
        f"{src_path} doesn't import KalmanV2SmoothingForm"
    )


def main() -> int:
    _check_form_module()
    _check_page_registration()
    print(
        "smoke_pose_cleanup_v2_wiring: 2/2 checks passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
