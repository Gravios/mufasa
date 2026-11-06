"""
mufasa.utils.feature_names
==========================

Display names for features in figures (patch 122jk).

Column names such as ``"Animal_1 head velocity forward along body-head axis
(mm/s)_FEATURE_SUBSET"`` are exact but long; for figures the user can give
shorter names ("forward head velocity"). They are kept per project in
``logs/measures/feature_names.toml``::

    [names]
    "Animal_1 head velocity forward along body-head axis (mm/s)_FEATURE_SUBSET" = "forward head velocity"

and used by mufasa-landscape, mufasa-landscape-animate and the Feature
Visualization dialogs. Only names that differ from the default are stored.

Patch 122jl: a feature can also be marked for a log10 scale; the
landscape tools then analyse and plot log10 of it (values ≤ 0 become
missing) and label it "log10 <name>"::

    [scales]
    "Animal_1 movement a (mm)_FEATURE_SUBSET" = "log10"

Only log10 entries are stored (linear is the default).

Patch 122jn: features that are close to discrete (counts, quantised
values) can be marked for jitter. Figures then add a small uniform noise to
them — at most ±0.4 of the typical step between the values they take — so
stacked points spread out; analyses (dip test, t-SNE) use the real values::

    [jitter]
    "Animal_1 number of rears_FEATURE_SUBSET" = true
"""
from __future__ import annotations

import tomllib
from pathlib import Path

FILENAME = "feature_names.toml"
SCALES = ("linear", "log10")
SUFFIX = "_FEATURE_SUBSET"


def default_name(col: str) -> str:
    """The column without the legacy ``_FEATURE_SUBSET`` suffix."""
    return col[: -len(SUFFIX)] if col.endswith(SUFFIX) else col


def names_path(config_path) -> Path:
    from mufasa.project_layout import project_paths_from_config
    roi = project_paths_from_config(str(config_path)).get("roi_definitions_path")
    base = Path(roi).parent if roi else Path(config_path).parent / "logs" / "measures"
    return base / FILENAME


def _read(path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_names(path) -> dict[str, str]:
    names = _read(path).get("names", {})
    return {str(k): str(v) for k, v in names.items() if str(v).strip()}


def read_scales(path) -> dict[str, str]:
    """{column: "log10"} for the features marked log10."""
    sc = _read(path).get("scales", {})
    return {str(k): "log10" for k, v in sc.items() if str(v).strip().lower() == "log10"}


def read_jitter(path) -> set[str]:
    """Columns marked for jitter in figures."""
    return {str(k) for k, v in _read(path).get("jitter", {}).items() if v is True}


def load_jitter(config_path) -> set[str]:
    return read_jitter(names_path(config_path))


def value_step(v) -> float:
    """Typical spacing between the distinct values of ``v``: the median gap
    between neighbouring distinct values (0 when fewer than two)."""
    import numpy as np
    u = np.unique(np.asarray(v, np.float64)[np.isfinite(v)])
    if u.size < 2:
        return 0.0
    if u.size > 20000:   # continuous anyway; keep it cheap
        u = u[:: u.size // 20000]
    return float(np.median(np.diff(u)))


def jitter_values(v, seed: int = 0, frac: float = 0.4, step: float | None = None):
    """``v`` plus uniform noise in ±``frac`` × the value step (NaN stays
    NaN). For figures only."""
    import numpy as np
    v = np.asarray(v, np.float64)
    st = value_step(v) if step is None else step
    if not st > 0:
        return v.copy()
    rng = np.random.default_rng(seed)
    return v + rng.uniform(-frac * st, frac * st, v.shape)


def load_scales(config_path) -> dict[str, str]:
    return read_scales(names_path(config_path))


def scale_of(col: str, scales: dict[str, str] | None) -> str:
    return "log10" if (scales or {}).get(col) == "log10" else "linear"


def load_names(config_path) -> dict[str, str]:
    return read_names(names_path(config_path))


def save_names(config_path, updates: dict[str, str],
               scales: dict[str, str] | None = None,
               jitter: dict[str, bool] | None = None) -> Path:
    """Merge ``updates`` ({column: name}) into the project's names. A name
    that is empty or equal to the default removes the entry. ``scales``
    ({column: "linear" | "log10"}) is merged the same way (linear removes)."""
    from mufasa.project_layout import write_project_toml
    p = names_path(config_path)
    names, sc, jit = read_names(p), read_scales(p), read_jitter(p)
    for col, on in (jitter or {}).items():
        (jit.add if on else jit.discard)(col)
    for col, name in updates.items():
        name = (name or "").strip()
        if not name or name == default_name(col):
            names.pop(col, None)
        else:
            names[col] = name
    for col, s in (scales or {}).items():
        if s not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, not {s!r}")
        if s == "log10":
            sc[col] = "log10"
        else:
            sc.pop(col, None)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"names": dict(sorted(names.items()))}
    if sc:
        data["scales"] = dict(sorted(sc.items()))
    if jit:
        data["jitter"] = {c: True for c in sorted(jit)}
    write_project_toml(p, data)
    return p


def display(col: str, names: dict[str, str] | None,
            scales: dict[str, str] | None = None) -> str:
    """Name to show for ``col``: the user's, else the default; prefixed
    "log10 " when ``scales`` marks it log10."""
    name = (names or {}).get(col) or default_name(col)
    return f"log10 {name}" if scale_of(col, scales) == "log10" else name


def apply_scales(df, scales: dict[str, str] | None, log=print):
    """log10 of the columns marked log10 (values ≤ 0 → NaN), in a copy;
    returns (frame, [columns transformed])."""
    import numpy as np
    cols = [c for c in df.columns if scale_of(c, scales) == "log10"]
    if not cols:
        return df, []
    out = df.copy()
    for c in cols:
        v = out[c].to_numpy(np.float64)
        bad = ~(v > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(bad, np.nan, np.log10(np.where(bad, 1.0, v)))
        nb = int((bad & np.isfinite(v)).sum())
        log(f"[scales] log10 {default_name(c)}" + (f" ({nb:,} values ≤ 0 → missing)"
                                                   if nb else ""))
    return out, cols


__all__ = ["FILENAME", "SCALES", "apply_scales", "jitter_values", "load_jitter",
           "read_jitter", "value_step", "default_name", "display", "load_names",
           "load_scales", "names_path", "read_names", "read_scales", "save_names",
           "scale_of"]
