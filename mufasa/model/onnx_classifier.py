"""
mufasa/model/onnx_classifier.py
===============================

Version-stable inference for Mufasa behaviour classifiers via ONNX.

WHY
---
A ``.sav`` classifier is a pickled scikit-learn ``RandomForestClassifier``.
scikit-learn does not support loading a pickle across versions, so a model
trained under (say) sklearn 0.22.2 cannot be loaded under 1.8.0 — exactly
the failure Mufasa surfaces as a ``CorruptedFileError``. ONNX sidesteps
this: the model is frozen as a computation graph that ``onnxruntime`` runs
independently of scikit-learn, so a model exported today keeps producing
identical predictions for years regardless of the installed sklearn.

WHAT THIS MODULE PROVIDES
-------------------------
* :class:`OnnxClassifier` — a duck-typed stand-in for a fitted sklearn
  RandomForest. It exposes the attributes Mufasa's inference path relies
  on (``n_features_in_``, ``n_classes_``, ``classes_``) and a
  ``predict_proba`` returning an ``(n_samples, n_classes)`` array, so it is
  interchangeable with an sklearn classifier inside
  :meth:`mufasa.mixins.train_model_mixin.TrainModelMixin.clf_predict_proba`.
* :func:`export_rf_to_onnx` — convert an in-memory fitted RandomForest to
  ONNX, embedding the metadata needed to reload it, and (by default)
  validating that the ONNX probabilities match sklearn within a tolerance.
* :func:`convert_sav_to_onnx` — convenience wrapper that loads a ``.sav``
  loadable in the *current* environment and exports it. (Models from an
  incompatible sklearn must first be loaded in an environment matching
  their original version — see docs/ONNX.md.)

PRECISION NOTE
--------------
The ONNX ``TreeEnsembleClassifier`` operator computes in single precision,
so converted probabilities can differ slightly from sklearn's float64
output. For behaviour scoring a difference near a decision threshold can
flip a frame, so :func:`export_rf_to_onnx` validates the maximum
probability delta on a sample and raises (or warns) if it exceeds ``atol``.

``onnxruntime`` and ``skl2onnx`` are optional (``pip install mufasa[onnx]``)
and imported lazily, so importing this module never requires them.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.ensemble import RandomForestClassifier

# Custom ONNX metadata keys (stored in the model's metadata_props) so an
# exported model is self-describing and can be reloaded without the source
# estimator.
META_N_FEATURES = "mufasa.n_features_in"
META_CLASSES = "mufasa.classes"
META_SOURCE_SKLEARN = "mufasa.source_sklearn_version"
META_CONVERTED_UTC = "mufasa.converted_utc"
META_MAX_PROBA_DELTA = "mufasa.max_proba_delta"
# Ordered training feature names (JSON list). Lets inference select exactly
# the columns the model was trained on (feature dispatch) instead of feeding
# the full computed feature set. sklearn 0.22-era models carry no feature
# names, so this is supplied at export/attach time.
META_FEATURE_NAMES = "mufasa.feature_names"

# Default tolerance for the sklearn-vs-ONNX probability check.
DEFAULT_ATOL = 1e-4


def _require(module: str, extra: str = "onnx"):
    """Import an optional dependency, raising a helpful error if absent."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise ImportError(
            f"'{module}' is required for ONNX support. Install the optional "
            f"dependencies with: pip install mufasa[{extra}]"
        ) from exc


class OnnxClassifier:
    """Drop-in inference shim around an ONNX RandomForest.

    Quacks like a fitted sklearn ``RandomForestClassifier`` for the subset
    of the API Mufasa's inference path uses.

    :param model_path: Path to a ``.onnx`` model exported by
        :func:`export_rf_to_onnx` (or any skl2onnx classifier export with
        ``zipmap=False``).
    """

    def __init__(self, model_path: str | os.PathLike) -> None:
        ort = _require("onnxruntime")
        model_path = os.fspath(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._proba_output = self._resolve_proba_output()

        meta = self._session.get_modelmeta().custom_metadata_map or {}
        self.classes_ = self._decode_classes(meta)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = self._resolve_n_features(meta)
        # Provenance, surfaced for logging / model cards.
        self.source_sklearn_version = meta.get(META_SOURCE_SKLEARN, "")
        self.converted_utc = meta.get(META_CONVERTED_UTC, "")
        # Ordered training feature names, if embedded. None -> inference uses
        # the full feature set (legacy behaviour); a list -> the inference
        # loop selects exactly these columns in this order (feature dispatch).
        self.feature_names_in_ = self._decode_feature_names(meta)

    # -- introspection helpers ------------------------------------------
    def _resolve_proba_output(self) -> str:
        outputs = self._session.get_outputs()
        for o in outputs:
            if o.name in ("probabilities", "output_probability"):
                return o.name
        # skl2onnx classifier outputs are [label, probabilities]; fall back
        # to the second output, or the only one if there is just one.
        return outputs[1].name if len(outputs) > 1 else outputs[0].name

    def _decode_feature_names(self, meta: dict[str, str]) -> np.ndarray | None:
        raw = meta.get(META_FEATURE_NAMES, "")
        if not raw:
            return None
        try:
            names = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if isinstance(names, list) and names:
            return np.array([str(n) for n in names])
        return None

    def _decode_classes(self, meta: dict[str, str]) -> np.ndarray:
        raw = meta.get(META_CLASSES, "")
        if raw:
            parts = [p for p in raw.split(",") if p != ""]
            try:
                return np.array([int(p) for p in parts])
            except ValueError:
                return np.array(parts)
        # Unknown classes: infer width from the proba output's last dim.
        shape = self._session.get_outputs()
        for o in shape:
            if o.name == self._proba_output and o.shape and o.shape[-1]:
                try:
                    return np.arange(int(o.shape[-1]))
                except (TypeError, ValueError):
                    break
        return np.array([0, 1])

    def _resolve_n_features(self, meta: dict[str, str]) -> int:
        raw = meta.get(META_N_FEATURES, "")
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        inp = self._session.get_inputs()[0]
        if inp.shape and inp.shape[-1] not in (None, "None"):
            try:
                return int(inp.shape[-1])
            except (TypeError, ValueError):
                pass
        raise ValueError(
            f"Could not determine n_features for ONNX model {self.model_path}"
        )

    # -- the sklearn-compatible surface ---------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape ``(n_samples, n_classes)``.

        Input is cast to ``float32`` (the dtype the ONNX tree operator
        expects). The returned array matches sklearn's ``predict_proba``
        column order (``classes_``), so callers can index the positive
        class as ``[:, 1]`` exactly as with an sklearn forest.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = self._session.run([self._proba_output], {self._input_name: X})[0]
        return self._as_proba_array(out)

    def _as_proba_array(self, out: Any) -> np.ndarray:
        # zipmap=False path: already an (n, n_classes) float tensor.
        if isinstance(out, np.ndarray) and out.ndim == 2:
            return out.astype(np.float64, copy=False)
        # zipmap path (defensive): a list of {class: prob} dicts.
        if isinstance(out, list) and out and isinstance(out[0], dict):
            cols = list(self.classes_)
            return np.array(
                [[float(row.get(c, row.get(int(c), 0.0))) for c in cols] for row in out],
                dtype=np.float64,
            )
        return np.asarray(out, dtype=np.float64)


def _set_onnx_metadata(onx, props: dict[str, str]) -> None:
    """Attach string metadata to an ONNX ModelProto in place."""
    for key, value in props.items():
        entry = onx.metadata_props.add()
        entry.key = key
        entry.value = str(value)


def export_rf_to_onnx(
    clf: RandomForestClassifier,
    save_path: str | os.PathLike,
    *,
    sample: np.ndarray | None = None,
    validate: bool = True,
    atol: float = DEFAULT_ATOL,
    n_validate: int = 256,
    error_on_mismatch: bool = False,
    source_sklearn_version: str | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a fitted sklearn RandomForest to ONNX at ``save_path``.

    :param clf: A fitted ``RandomForestClassifier``.
    :param save_path: Destination ``.onnx`` path.
    :param sample: Optional representative feature array
        ``(k, n_features)`` for the probability-equivalence check. A real
        feature sample gives the most meaningful delta; when omitted a
        random sample is used.
    :param validate: If True, compare ONNX vs sklearn probabilities.
    :param atol: Maximum allowed absolute probability difference.
    :param error_on_mismatch: If True, raise when the delta exceeds
        ``atol`` (otherwise warn). Default warns, so a tiny float32 drift
        does not block export.
    :param feature_names: Ordered training feature names to embed, enabling
        feature dispatch at inference (the loop selects exactly these
        columns). Defaults to ``clf.feature_names_in_`` when present
        (sklearn >= 1.0 fitted on a DataFrame); 0.22-era models carry none,
        so pass the names from the original training data.
    :returns: A metadata dict (save_path, n_features, classes,
        max_proba_delta, source_sklearn_version, converted_utc,
        feature_names).
    """
    skl2onnx = _require("skl2onnx")
    from skl2onnx.common.data_types import FloatTensorType

    if feature_names is None:
        fni = getattr(clf, "feature_names_in_", None)
        feature_names = [str(n) for n in fni] if fni is not None else None
    if feature_names is not None and len(feature_names) != int(
        getattr(clf, "n_features_in_", None) or getattr(clf, "n_features_", 0) or 0
    ):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but the model "
            f"expects {getattr(clf, 'n_features_in_', getattr(clf, 'n_features_', '?'))} features."
        )

    n_features = int(
        getattr(clf, "n_features_in_", None)
        or getattr(clf, "n_features_", None)
        or 0
    )
    if not n_features:
        raise ValueError("Classifier exposes no n_features_in_/n_features_.")

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onx = skl2onnx.convert_sklearn(
        clf,
        initial_types=initial_types,
        options={type(clf): {"zipmap": False}},
    )

    if source_sklearn_version is None:
        try:
            import sklearn
            source_sklearn_version = sklearn.__version__
        except Exception:
            source_sklearn_version = ""

    classes = getattr(clf, "classes_", np.array([0, 1]))
    converted_utc = _dt.datetime.now(_dt.UTC).isoformat()

    # -- numerical validation (sklearn float64 vs ONNX float32) ---------
    max_delta: float | None = None
    if validate:
        if sample is None:
            rng = np.random.RandomState(0)
            sample = rng.standard_normal((n_validate, n_features))
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape[1] != n_features:
            raise ValueError(
                f"sample has {sample.shape[1]} features, model expects {n_features}."
            )
        ref = clf.predict_proba(sample)
        ort = _require("onnxruntime")
        sess = ort.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        proba_out = None
        for o in sess.get_outputs():
            if o.name in ("probabilities", "output_probability"):
                proba_out = o.name
        if proba_out is None:
            outs = sess.get_outputs()
            proba_out = outs[1].name if len(outs) > 1 else outs[0].name
        got = sess.run(
            [proba_out],
            {sess.get_inputs()[0].name: sample.astype(np.float32)},
        )[0]
        max_delta = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        if max_delta > atol:
            msg = (
                f"ONNX probabilities diverge from sklearn by up to "
                f"{max_delta:.2e} (> atol={atol:.0e}). This is the float32 "
                f"tree-operator drift; validate against your data before use."
            )
            if error_on_mismatch:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    _set_onnx_metadata(
        onx,
        {
            META_N_FEATURES: n_features,
            META_CLASSES: ",".join(str(int(c)) if isinstance(c, (int, np.integer)) else str(c) for c in classes),
            META_SOURCE_SKLEARN: source_sklearn_version or "",
            META_CONVERTED_UTC: converted_utc,
            META_MAX_PROBA_DELTA: "" if max_delta is None else f"{max_delta:.6e}",
            **(
                {META_FEATURE_NAMES: json.dumps([str(n) for n in feature_names])}
                if feature_names is not None
                else {}
            ),
        },
    )

    save_path = os.fspath(save_path)
    with open(save_path, "wb") as f:
        f.write(onx.SerializeToString())

    return {
        "save_path": save_path,
        "n_features": n_features,
        "classes": list(classes),
        "max_proba_delta": max_delta,
        "source_sklearn_version": source_sklearn_version or "",
        "converted_utc": converted_utc,
        "feature_names": list(feature_names) if feature_names is not None else None,
    }


def convert_sav_to_onnx(
    sav_path: str | os.PathLike,
    onnx_path: str | os.PathLike | None = None,
    **export_kwargs: Any,
) -> dict[str, Any]:
    """Load a ``.sav`` (must be loadable in the *current* sklearn) and
    export it to ONNX. ``onnx_path`` defaults to ``sav_path`` with a
    ``.onnx`` suffix. Extra kwargs pass through to
    :func:`export_rf_to_onnx`.

    Models pickled by an incompatible sklearn cannot be loaded here; load
    them in an environment matching their original version first (see
    docs/ONNX.md).
    """
    import pickle

    sav_path = os.fspath(sav_path)
    with open(sav_path, "rb") as f:
        clf = pickle.load(f)
    if onnx_path is None:
        onnx_path = os.path.splitext(sav_path)[0] + ".onnx"
    return export_rf_to_onnx(clf, onnx_path, **export_kwargs)


def attach_feature_names(
    onnx_path: str | os.PathLike,
    feature_names: list[str],
) -> int:
    """Attach (or replace) the ordered training feature-name manifest on an
    existing ONNX model in place, enabling feature dispatch at inference.

    Pure ONNX metadata editing — needs only ``onnx`` (no sklearn, no legacy
    environment), so it runs in the normal Mufasa env on a model that was
    already converted without names. The number of names must equal the
    model's input feature count.

    :param onnx_path: Path to the ``.onnx`` model (overwritten in place).
    :param feature_names: Ordered feature names, length == model n_features.
    :returns: The number of feature names written.
    """
    onnx = _require("onnx")
    feature_names = [str(n) for n in feature_names]
    model_path = os.fspath(onnx_path)
    model = onnx.load(model_path)

    # Sanity-check length against the model's declared input width.
    expected = None
    try:
        dims = model.graph.input[0].type.tensor_type.shape.dim
        expected = dims[-1].dim_value or None
    except (IndexError, AttributeError):
        expected = None
    if expected and expected != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but the model "
            f"input declares {expected} features."
        )

    # Drop any existing manifest, then write the new one.
    keep = [p for p in model.metadata_props if p.key != META_FEATURE_NAMES]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    entry = model.metadata_props.add()
    entry.key = META_FEATURE_NAMES
    entry.value = json.dumps(feature_names)

    onnx.save(model, model_path)
    return len(feature_names)
