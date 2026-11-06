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
"""
from __future__ import annotations

import tomllib
from pathlib import Path

FILENAME = "feature_names.toml"
SUFFIX = "_FEATURE_SUBSET"


def default_name(col: str) -> str:
    """The column without the legacy ``_FEATURE_SUBSET`` suffix."""
    return col[: -len(SUFFIX)] if col.endswith(SUFFIX) else col


def names_path(config_path) -> Path:
    from mufasa.project_layout import project_paths_from_config
    roi = project_paths_from_config(str(config_path)).get("roi_definitions_path")
    base = Path(roi).parent if roi else Path(config_path).parent / "logs" / "measures"
    return base / FILENAME


def read_names(path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    names = data.get("names", {})
    return {str(k): str(v) for k, v in names.items() if str(v).strip()}


def load_names(config_path) -> dict[str, str]:
    return read_names(names_path(config_path))


def save_names(config_path, updates: dict[str, str]) -> Path:
    """Merge ``updates`` ({column: name}) into the project's names. A name
    that is empty or equal to the default removes the entry."""
    from mufasa.project_layout import write_project_toml
    p = names_path(config_path)
    names = read_names(p)
    for col, name in updates.items():
        name = (name or "").strip()
        if not name or name == default_name(col):
            names.pop(col, None)
        else:
            names[col] = name
    p.parent.mkdir(parents=True, exist_ok=True)
    write_project_toml(p, {"names": dict(sorted(names.items()))})
    return p


def display(col: str, names: dict[str, str] | None) -> str:
    """Name to show for ``col``: the user's, else the default."""
    return (names or {}).get(col) or default_name(col)


__all__ = ["FILENAME", "default_name", "display", "load_names", "names_path",
           "read_names", "save_names"]
