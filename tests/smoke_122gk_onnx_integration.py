"""
tests/smoke_122gk_onnx_integration.py
=====================================

Patch 122gk — ONNX support, part 2: wire OnnxClassifier into the inference
path and record model format in project.toml.

* TrainModelMixin.read_clf dispatches by extension — ``.onnx`` ->
  OnnxClassifier (lazy import), else read_pickle. Because OnnxClassifier
  duck-types as an sklearn forest, clf_predict_proba is unchanged.
* inference_batch and inference_validation load via read_clf (not
  read_pickle), so both batch and validation inference accept ONNX models.
* project.toml records ``model_format`` per classifier in
  [classifier_inference.<name>] (explicit value wins, else derived from the
  model_path extension via model_format_for_path).
* The Run inference browse dialog lists ``.onnx`` alongside ``.sav``.
"""

import ast
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MIXIN = REPO_ROOT / "mufasa" / "mixins" / "train_model_mixin.py"
BATCH = REPO_ROOT / "mufasa" / "model" / "inference_batch.py"
VALID = REPO_ROOT / "mufasa" / "model" / "inference_validation.py"
FORM = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "run_inference.py"
LAYOUT = REPO_ROOT / "mufasa" / "project_layout.py"

CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label, cond, *, detail=""):
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _fn_src(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.unparse(n)
    return ""


def main():
    try:
        mixin_tree = ast.parse(MIXIN.read_text(encoding="utf-8"))
        ast.parse(LAYOUT.read_text(encoding="utf-8"))
        parsed = True
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122gk_onnx_integration: 0/6 checks passed")
        return 1
    check("train_model_mixin.py + project_layout.py parse cleanly", parsed)

    read_clf = _fn_src(mixin_tree, "read_clf")
    check(
        "read_clf dispatches .onnx -> OnnxClassifier (lazy) else read_pickle",
        ".onnx" in read_clf and "OnnxClassifier" in read_clf
        and "import" in read_clf and "self.read_pickle" in read_clf,
    )

    batch = BATCH.read_text(encoding="utf-8")
    valid = VALID.read_text(encoding="utf-8")
    check(
        "inference_batch + inference_validation load clf via read_clf",
        "self.read_clf(file_path=" in batch and "self.read_clf(file_path=" in valid,
    )

    try:
        from mufasa.project_layout import (
            PROJECT_LAYOUT_VERSION,
            read_classifier_inference_settings,
            write_classifier_inference_settings,
        )
        d = Path(tempfile.mkdtemp())
        cfg = d / "project.toml"
        cfg.write_text(f"project_layout_version = {PROJECT_LAYOUT_VERSION}\n")
        write_classifier_inference_settings(
            cfg,
            {"Rear": {"model_path": "/x/Rear.onnx", "threshold": 0.5},
             "Sniff": {"model_path": "/x/Sniff.sav"}},
        )
        got = read_classifier_inference_settings(cfg)
        derived_ok = (got["Rear"].get("model_format") == "onnx"
                      and got["Sniff"].get("model_format") == "sklearn")
        write_classifier_inference_settings(
            cfg, {"Rear": {"model_path": "/x/Rear.onnx", "model_format": "sklearn"}})
        override_ok = (read_classifier_inference_settings(cfg)["Rear"].get(
            "model_format") == "sklearn")
    except Exception as e:  # noqa: BLE001
        derived_ok = override_ok = False
        print(f"  (functional error: {e})")

    check("model_format round-trips: .onnx->onnx, .sav->sklearn", derived_ok)
    check("explicit model_format overrides extension-derived value", override_ok)

    form = FORM.read_text(encoding="utf-8")
    check("Run inference browse filter offers *.onnx", "*.onnx" in form)

    print(f"smoke_122gk_onnx_integration: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed")
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
