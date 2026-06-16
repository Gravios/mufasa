"""
tests/smoke_122gl_feature_dispatch.py
=====================================

Patch 122gl — feature dispatch: apply a model to exactly the features it was
trained on.

A model trained on an older / smaller feature set (e.g. the 758-feature
SimBA set) fails on data carrying the current 2380-feature superset with a
FeatureNumberMismatchError. This patch lets the *model* carry an ordered
feature manifest and has inference select those columns from the computed
feature frame.

* OnnxClassifier.feature_names_in_ is read from ONNX metadata
  (mufasa.feature_names, JSON). None when absent.
* export_rf_to_onnx accepts/embeds feature_names (defaults to
  clf.feature_names_in_); attach_feature_names() adds the manifest to an
  already-converted .onnx in place (pure onnx, no legacy env).
* TrainModelMixin.select_model_features(clf, x_df) selects/reorders x_df to
  clf.feature_names_in_ when present (sklearn>=1.0 models have it natively),
  else passes through; missing expected columns raise MissingColumnsError
  listing them (surfaces schema divergence instead of silent misalignment).
* inference_batch (default path) and inference_validation call it before
  clf_predict_proba.

CHECKS (7)
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OCLS = REPO / "mufasa" / "model" / "onnx_classifier.py"
MIXIN = REPO / "mufasa" / "mixins" / "train_model_mixin.py"
BATCH = REPO / "mufasa" / "model" / "inference_batch.py"
VALID = REPO / "mufasa" / "model" / "inference_validation.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    spec = importlib.util.spec_from_file_location("ocls", OCLS)
    oc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oc)

    class S:
        pass

    dec = oc.OnnxClassifier._decode_feature_names
    check(
        "OnnxClassifier decodes feature_names_in_ from metadata (None if absent)",
        list(dec(S(), {oc.META_FEATURE_NAMES: json.dumps(["f1", "f2"])})) == ["f1", "f2"]
        and dec(S(), {}) is None,
    )

    check("attach_feature_names() helper exists", hasattr(oc, "attach_feature_names"))

    src = OCLS.read_text(encoding="utf-8")
    check(
        "OnnxClassifier sets self.feature_names_in_ and export embeds it",
        "self.feature_names_in_" in src and "META_FEATURE_NAMES: json.dumps" in src,
    )

    # exec the real select_model_features body in isolation
    tree = ast.parse(MIXIN.read_text(encoding="utf-8"))
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "select_model_features"),
        None,
    )
    if fn is None:
        check("select_model_features defined", False)
        check("select_model_features select/reorder/passthrough", False)
        check("select_model_features raises on missing", False)
    else:
        check("select_model_features defined", True)

        class MissingColumnsError(Exception):
            def __init__(self, msg, source=None):
                super().__init__(msg)

        ns = {"pd": pd, "MissingColumnsError": MissingColumnsError}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<s>", "exec"), ns)
        sel = ns["select_model_features"]

        class Self:
            __class__ = type("X", (), {"__name__": "X"})

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        class Clf:
            feature_names_in_ = np.array(["c", "a"])

        class ClfNone:
            feature_names_in_ = None

        sel_ok = list(sel(Self(), Clf(), df).columns) == ["c", "a"]
        pass_ok = sel(Self(), ClfNone(), df).shape[1] == 3
        check("select_model_features selects+reorders / passes through", sel_ok and pass_ok)

        class ClfMiss:
            feature_names_in_ = np.array(["a", "ZZZ"])

        raised = False
        try:
            sel(Self(), ClfMiss(), df)
        except MissingColumnsError as e:
            raised = "ZZZ" in str(e)
        check("select_model_features raises MissingColumnsError on missing", raised)

    batch = BATCH.read_text(encoding="utf-8")
    valid = VALID.read_text(encoding="utf-8")
    check(
        "inference_batch + inference_validation apply select_model_features",
        "self.select_model_features(clf=clf" in batch
        and "self.select_model_features(clf=clf" in valid,
    )

    print(f"smoke_122gl_feature_dispatch: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
