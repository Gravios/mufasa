"""
mufasa.utils.feature_view_state
===============================

Feature Visualization state kept between Mufasa sessions (patch 122jq), per
project in ``logs/measures/feature_view.toml``::

    selected = ["Animal_1 movement a (mm)_FEATURE_SUBSET", ...]   # in order
    video = "K7_cam_20251124_1510_A10-0-051"
    colour_by = "rear"
    joint_style = "heatmap"
    jitter = false
    group_by_name_parts = true

Unknown keys are kept; a missing or unreadable file gives ``{}``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

FILENAME = "feature_view.toml"


def state_path(config_path) -> Path:
    from mufasa.utils.feature_names import names_path
    return names_path(config_path).with_name(FILENAME)


def load_state(config_path) -> dict:
    p = state_path(config_path)
    try:
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    sel = data.get("selected", [])
    data["selected"] = [str(c) for c in sel] if isinstance(sel, list) else []
    return data


def save_state(config_path, updates: dict) -> Path:
    """Merge ``updates`` into the saved state (None removes a key)."""
    from mufasa.project_layout import write_project_toml
    p = state_path(config_path)
    data = load_state(config_path) if p.is_file() else {}
    for k, v in updates.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = list(v) if isinstance(v, (list, tuple)) else v
    p.parent.mkdir(parents=True, exist_ok=True)
    write_project_toml(p, data)
    return p


__all__ = ["FILENAME", "load_state", "save_state", "state_path"]
