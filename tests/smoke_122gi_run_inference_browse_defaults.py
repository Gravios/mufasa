"""
tests/smoke_122gi_run_inference_browse_defaults.py
==================================================

Patch 122gi — Run inference model browse: default to the project models
folder, and remember the last folder used.

USER REQUEST
============
"Inference : Run Inference should save the last path used for a save file,
and when browsing for a new file it should open to the models folder of
the project."

BEFORE
======
_on_browse opened QFileDialog at ``edit.text()`` — empty for a fresh row,
so the dialog opened at the current working directory and forgot where you
were every time.

AFTER
=====
_on_browse resolves its start directory via _browse_start_dir, precedence:
  1. the directory of the path already in this row (re-browse in place);
  2. the last folder a model was picked from — saved with QSettings
     (QApplication org/app = "Mufasa"), so it persists across sessions;
  3. the project's models folder via _project_models_dir
     (models/generated_models if present, else models/) — the default for
     a fresh pick;
  4. "" → cwd.
On a successful pick the chosen file's directory is written back to
QSettings, so the next browse (this session or after a restart) reopens
there.

NEW SMOKE: smoke_122gi_run_inference_browse_defaults.py (5 checks)
* run_inference.py parses cleanly
* QSettings is imported
* RunInferenceForm defines _browse_start_dir and _project_models_dir
* _on_browse passes a resolved start dir to getOpenFileName (not the bare
  field text) and persists the picked dir via QSettings().setValue
* _project_models_dir resolves via project_paths_from_config and prefers
  generated_models
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FORM = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "run_inference.py"

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
    src = FORM.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: run_inference.py parses — {e}")
        print("smoke_122gi_run_inference_browse_defaults: 0/5 checks passed")
        return 1
    check("run_inference.py parses cleanly", True)

    check("QSettings is imported", "QSettings" in src)

    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    check(
        "RunInferenceForm defines _browse_start_dir and _project_models_dir",
        "_browse_start_dir" in fns and "_project_models_dir" in fns,
    )

    if "_on_browse" in fns:
        body = ast.unparse(fns["_on_browse"])
        check(
            "_on_browse uses a resolved start dir + persists picked dir via QSettings",
            "_browse_start_dir" in body
            and "getOpenFileName" in body
            and "QSettings().setValue" in body
            and "edit.text()," not in body.replace(" ", ""),  # no longer the bare arg
        )
    else:
        check("_on_browse uses resolved start dir + persists via QSettings", False)

    if "_project_models_dir" in fns:
        body = ast.unparse(fns["_project_models_dir"])
        check(
            "_project_models_dir resolves models folder (prefers generated_models)",
            "project_paths_from_config" in body
            and "models_dir" in body
            and "generated_models" in body,
        )
    else:
        check("_project_models_dir resolves models folder", False)

    print(
        f"smoke_122gi_run_inference_browse_defaults: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
