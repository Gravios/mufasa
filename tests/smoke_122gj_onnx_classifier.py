"""
tests/smoke_122gj_onnx_classifier.py
====================================

Patch 122gj — ONNX support, part 1: the OnnxClassifier shim + sklearn->ONNX
export/convert helpers + the ``onnx`` optional-dependency extra.

Provides a version-stable inference path that decouples behaviour
classifiers from the scikit-learn pickle format (which is not portable
across sklearn versions — the root cause of the .sav load failures).

Sandbox note: onnxruntime/skl2onnx cannot be imported here, so the runtime
numerical-equivalence check happens on the user's machine. These checks are
AST/structure-based plus a lazy-import import of the module itself (numpy
only), confirming the heavy deps are deferred.

CHECKS (7)
* onnx_classifier.py parses cleanly
* the module imports with numpy alone (onnxruntime/skl2onnx NOT imported at
  module top level — they are lazy)
* OnnxClassifier defines predict_proba and sets the sklearn-compatible
  attrs (n_features_in_, n_classes_, classes_) used by clf_predict_proba
* export_rf_to_onnx and convert_sav_to_onnx are defined
* export uses zipmap=False (clean probability tensor) and records a
  max-proba-delta validation
* _require raises a helpful 'pip install mufasa[onnx]' error for a missing
  dep
* pyproject.toml declares an [onnx] extra (onnx + onnxruntime + skl2onnx)
"""

import ast
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MOD = REPO_ROOT / "mufasa" / "model" / "onnx_classifier.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"

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
    src = MOD.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: onnx_classifier.py parses — {e}")
        print("smoke_122gj_onnx_classifier: 0/7 checks passed")
        return 1
    check("onnx_classifier.py parses cleanly", True)

    # module imports with numpy alone; heavy deps must be lazy
    top_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    heavy = {"onnxruntime", "skl2onnx", "onnx"}
    ok_import = False
    try:
        spec = importlib.util.spec_from_file_location("ocls_smoke", MOD)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        ok_import = True
    except Exception as e:  # noqa: BLE001
        print(f"  (import error: {e})")
    check(
        "imports with numpy alone; onnxruntime/skl2onnx/onnx not top-level",
        ok_import and not (heavy & top_imports),
        detail=f"heavy top-level imports: {heavy & top_imports}",
    )

    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    oc = classes.get("OnnxClassifier")
    if oc is not None:
        body = ast.unparse(oc)
        check(
            "OnnxClassifier defines predict_proba + sklearn-compat attrs",
            "def predict_proba" in body
            and "self.n_features_in_" in body
            and "self.n_classes_" in body
            and "self.classes_" in body,
        )
    else:
        check("OnnxClassifier defines predict_proba + sklearn-compat attrs", False)

    check(
        "export_rf_to_onnx and convert_sav_to_onnx defined",
        "export_rf_to_onnx" in fns and "convert_sav_to_onnx" in fns,
    )

    export_src = ""
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "export_rf_to_onnx":
            export_src = ast.unparse(n)
    check(
        "export uses zipmap=False and records max-proba-delta validation",
        "'zipmap': False" in export_src and "max_delta" in export_src,
    )

    check(
        "_require raises helpful 'pip install mufasa[onnx]' error",
        ok_import and _require_errors(m),
    )

    pyproject = PYPROJECT.read_text(encoding="utf-8")
    check(
        "pyproject declares [onnx] extra (onnx + onnxruntime + skl2onnx)",
        "onnx = [" in pyproject
        and "onnxruntime" in pyproject
        and "skl2onnx" in pyproject,
    )

    print(f"smoke_122gj_onnx_classifier: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed")
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


def _require_errors(module) -> bool:
    try:
        module._require("definitely_not_installed_xyz_pkg")
        return False
    except ImportError as e:
        return "pip install mufasa[onnx]" in str(e)
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
