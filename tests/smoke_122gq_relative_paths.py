"""
tests/smoke_122gq_relative_paths.py
===================================

Patch 122gq — project.toml holds no absolute paths.

Mufasa runs from the top of the project, so stored paths are relativised to
the project root (the folder containing project.toml) on write and resolved
back to absolute on read. Applied to model_path in
[classifier_inference.<name>] (the main stored path).

* relativize_project_path(path, root): absolute -> root-relative; relative
  passes through; cross-drive falls back to as-is.
* resolve_project_path(path, root): relative -> absolute vs root; absolute
  passes through.
* round-trip through write/read_classifier_inference_settings: on-disk value
  is relative, read value is absolute, external paths still round-trip.
"""
import os
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        read_classifier_inference_settings,
        relativize_project_path,
        resolve_project_path,
        write_classifier_inference_settings,
    )

    root = "/proj/root"
    check("relativize: absolute under root -> relative",
          relativize_project_path("/proj/root/models/x.onnx", root)
          == "models/x.onnx")
    check("relativize: already-relative passes through",
          relativize_project_path("models/x.onnx", root) == "models/x.onnx")
    check("resolve: relative -> absolute vs root",
          resolve_project_path("models/x.onnx", root)
          == os.path.normpath("/proj/root/models/x.onnx"))
    check("resolve: absolute passes through",
          resolve_project_path("/opt/x.sav", root) == "/opt/x.sav")

    d = pathlib.Path(tempfile.mkdtemp())
    cfg = d / "project.toml"
    cfg.write_text(f"project_layout_version = {PROJECT_LAYOUT_VERSION}\n")
    mp = str(d / "models" / "generated_models" / "Rear.onnx")
    write_classifier_inference_settings(cfg, {"Rear": {"model_path": mp, "threshold": 0.5}})
    disk = cfg.read_text()
    check("model_path stored relative on disk (no absolute path)",
          "models/generated_models/Rear.onnx" in disk
          and f'"{d}' not in disk)
    got = read_classifier_inference_settings(cfg)
    check("model_path read back resolved to absolute",
          got["Rear"]["model_path"] == os.path.normpath(mp))
    check("model_format still derived correctly (onnx)",
          got["Rear"].get("model_format") == "onnx")

    write_classifier_inference_settings(cfg, {"Attack": {"model_path": "/opt/models/attack.sav"}})
    check("external (outside-project) model_path round-trips",
          read_classifier_inference_settings(cfg)["Attack"]["model_path"]
          == "/opt/models/attack.sav")

    print(f"smoke_122gq_relative_paths: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
