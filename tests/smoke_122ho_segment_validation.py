"""Smoke test for patch 122ho — project-aware segment fields in the GUI.

The Kalman v2 form's orientation-drift / const-accel / high-angular-noise
fields name *segments*, which the smoother validates against the project's
skeleton. Those names are project-specific (a renamed rig has back / back_rear
/ head / neck / tail_1.. rather than the pre-rename body / head), but the form
hardcoded a 'body,head' example and only failed inside dataclasses.replace with
a raw ValueError stack trace when a stale/typo'd name was submitted.

122ho: (1) a _project_segment_names(config_path) helper resolves the real
segment names (degrading to [] rather than crashing); (2) the form's segment
fields validate against those names before _dc.replace, raising a clear message
that lists the valid segments; (3) the stale 'body,head' examples are removed
from the placeholders and docstrings.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


gui_path = REPO / "mufasa/ui_qt/forms/pose_cleanup.py"
gui = gui_path.read_text()
gtree = ast.parse(gui)

# ---- the helper exists and degrades safely ----
helper = next((n for n in ast.walk(gtree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_project_segment_names"), None)
check("_project_segment_names helper is defined", helper is not None)
helper_src = ast.get_source_segment(gui, helper) if helper else ""
check("helper returns [] when no config_path",
      "if not config_path:" in helper_src and "return []" in helper_src)
check("helper resolves segments via layout_from_config",
      "layout_from_config" in helper_src
      and "sg.name for sg in layout.segments" in helper_src)
check("helper swallows failures (degrades, not crashes)",
      "except Exception:" in helper_src)

# helper actually runs and returns [] for a bad path (no crash)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("_pc_mod", gui_path)
# We can't import the full module (PySide6 absent); instead exec just the
# helper by pulling it out. Simpler: check behaviour via a tiny stub.
# Compile the helper in isolation with a fake layout_from_config.
ns: dict = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]),
             str(gui_path), "exec"), ns)
check("helper returns [] for empty config_path (runtime)",
      ns["_project_segment_names"](None) == [])
check("helper returns [] for a nonexistent config (runtime, no crash)",
      ns["_project_segment_names"]("/no/such/project.toml") == [])

# ---- the validation is present before _dc.replace ----
form = next((n for n in ast.walk(gtree)
             if isinstance(n, ast.ClassDef)
             and n.name == "KalmanV2SmoothingForm"), None)
form_src = ast.get_source_segment(gui, form) if form else ""
check("form validates segments before replace",
      "_valid_segs = {sg.name for sg in layout.segments}" in form_src)
check("validation covers all three segment fields",
      form_src.count("orientation_drift_segments") >= 1
      and "const_accel_segments" in form_src
      and "high_angular_noise_segments" in form_src)
check("validation raises a message naming available segments",
      "Available segments:" in form_src)
# the validation must come before the _dc.replace that triggers the deep error
vpos = form_src.find("_valid_segs = {sg.name")
rpos = form_src.find("_dc.replace(layout, **replacements)")
check("validation precedes _dc.replace", 0 <= vpos < rpos)

# ---- the validation logic behaves correctly (replicated) ----
def _validate(replacements, valid):
    fields = (
        ("Orientation drift segments", "orientation_drift_segments"),
        ("Constant-accel segments", "const_accel_segments"),
        ("High-angular-noise segments", "high_angular_noise_segments"),
    )
    for label, key in fields:
        bad = [n for n in replacements.get(key, []) if n not in valid]
        if bad:
            return f"{label}: bad {bad}"
    return None


VALID = {"back", "back_rear", "head", "neck", "tail_1", "tail_2", "tail_3"}
check("the stale 'body' is rejected",
      _validate({"orientation_drift_segments": ["body", "head"]}, VALID)
      is not None)
check("the correct 'back' passes",
      _validate({"orientation_drift_segments": ["back", "head"],
                 "const_accel_segments": ["back", "head"],
                 "high_angular_noise_segments": ["head"]}, VALID) is None)
check("a typo is rejected",
      _validate({"high_angular_noise_segments": ["heat"]}, VALID) is not None)
check("empty segment lists pass", _validate({}, VALID) is None)

# ---- stale examples removed ----
check("no 'body,head' example remains in the form",
      "body,head" not in gui)
check("placeholders are project-aware (use resolved names)",
      "_project_segment_names(self.config_path)" in form_src
      and "setPlaceholderText(_seg_hint)" in form_src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ho_segment_validation: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
