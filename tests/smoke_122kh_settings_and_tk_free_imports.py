"""
tests/smoke_122kh_settings_and_tk_free_imports.py
=================================================

Patch 122kh: regressions found while bringing the smoke suite up to date.

1. mufasa imports without tkinter. ``utils.errors``, ``utils.lookups``,
   ``roi_tools.roi_utils`` and ``video_processors.video_processing``
   imported tkinter at module level, so on a Python built without Tk
   (common on Linux servers and conda) most of mufasa failed to import.
2. Classifier model paths. ``project.toml`` stores them relative to the
   project; the TOML→INI translator handed them on unresolved
   (``../../abs/model.sav``), so inference opened the wrong path.
3. Directing settings. ``DirectingBodyPartSettingsForm`` read and wrote
   ``project.toml`` with configparser (writing INI text over the TOML),
   and the translator never passed ``bodypart_direction`` on.
"""
from __future__ import annotations

import builtins
import importlib
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


TK_FREE = (
    "mufasa.utils.errors",
    "mufasa.utils.lookups",
    "mufasa.utils.checks",
    "mufasa.roi_tools.roi_utils",
    "mufasa.video_processors.video_processing",
    "mufasa.mixins.config_reader",
)


def _check_tk_free() -> None:
    real_import = builtins.__import__

    def no_tk(name, *a, **k):
        if name == "tkinter" or name.startswith("tkinter.") or name == "PIL.ImageTk":
            raise ModuleNotFoundError("No module named 'tkinter' (blocked by test)")
        return real_import(name, *a, **k)

    saved = {m: sys.modules.pop(m) for m in list(sys.modules)
             if m.startswith("mufasa") or m.startswith("tkinter") or m == "PIL.ImageTk"}
    builtins.__import__ = no_tk
    try:
        for mod in TK_FREE:
            try:
                importlib.import_module(mod)
                ok, err = True, ""
            except Exception as e:      # noqa: BLE001
                ok, err = False, repr(e)
            check(f"{mod} imports without tkinter", ok, detail=err)
        from mufasa.utils.errors import InvalidInputError
        try:
            raise InvalidInputError(msg="x", show_window=True)
        except InvalidInputError:
            shown = True
        except Exception as e:          # noqa: BLE001
            shown = False
            print(repr(e))
        check("an error with show_window=True still raises without Tk", shown)
    finally:
        builtins.__import__ = real_import
        for m in [m for m in sys.modules if m.startswith("mufasa")]:
            del sys.modules[m]
        sys.modules.update(saved)


def _project(tmp: Path) -> Path:
    from mufasa.project_layout import write_project_toml
    p = tmp / "project.toml"
    write_project_toml(p, {
        "project_layout_version": 1,
        "project": {"name": "smoke_122kh"},
        "pose": {"file_type": "csv", "animal_count": 1, "body_parts": ["nose", "tail"]},
        "classifiers": {"targets": ["attack", "grooming"]},
    })
    return p


def _check_model_paths() -> None:
    from mufasa.project_layout import read_project_toml, write_classifier_inference_settings
    from mufasa.utils.toml_to_configparser import read_project_toml_as_configparser
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p = _project(tmp)
        inside = tmp / "models" / "attack" / "model.sav"
        outside = Path(td).parent / "elsewhere" / "grooming.sav"
        write_classifier_inference_settings(p, {
            "attack": {"model_path": str(inside), "threshold": 0.5, "min_bout_ms": 100},
            "grooming": {"model_path": str(outside), "threshold": 0.4, "min_bout_ms": 100},
        })
        stored = read_project_toml(p)["classifier_inference"]["attack"]["model_path"]
        check("project.toml stores the model path relative", not os.path.isabs(stored),
              detail=stored)
        cp = read_project_toml_as_configparser(p)
        m1 = cp.get("SML settings", "model_path_1")
        m2 = cp.get("SML settings", "model_path_2")
        check("translator: model inside the project comes back absolute",
              Path(m1) == inside, detail=m1)
        check("translator: model outside the project comes back absolute",
              Path(m2) == outside, detail=m2)


def _check_directing() -> None:
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mufasa.project_layout import read_project_toml
    from mufasa.ui_qt.forms.addons import DirectingBodyPartSettingsForm as Form
    from mufasa.utils.toml_to_configparser import read_project_toml_as_configparser
    with tempfile.TemporaryDirectory() as td:
        p = _project(Path(td))
        form = Form.__new__(Form)
        form.config_path = str(p)
        check("no direction set yet", Form._current_setting(form) is None)
        Form.target(form, config_path=str(p), bp="tail")
        try:
            data = read_project_toml(p)
            ok = True
        except Exception as e:          # noqa: BLE001
            data, ok = {}, False
            print(repr(e))
        check("project.toml is still TOML after saving", ok)
        check("saved under [directionality]",
              data.get("directionality", {}).get("bodypart_direction") == "tail")
        check("other tables kept", data.get("classifiers", {}).get("targets") == ["attack", "grooming"])
        check("form reads it back", Form._current_setting(form) == "tail")
        cp = read_project_toml_as_configparser(p)
        check("ConfigReader sees [Directionality settings] bodypart_direction",
              cp.get("Directionality settings", "bodypart_direction", fallback="") == "tail")


def main() -> int:
    _check_tk_free()
    _check_model_paths()
    _check_directing()
    print(f"smoke_122kh_settings_and_tk_free_imports: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed")
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
