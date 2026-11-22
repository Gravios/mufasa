"""Smoke test for patch 122ld — install dependencies for 3-D pose (PyTorch), and saying what is
missing.

User: Mufasa said PyTorch CUDA is not available; ``python -c "import torch"`` → ModuleNotFoundError
in the mufasa environment (an RTX 5070 Ti, driver showing CUDA 13.2). "update the install deps".

* a ``pose3d`` extra (PyTorch 2.7+, the first with Blackwell kernels); ``gpu`` and ``all`` move
  from 2.4 to 2.7;
* ``gpu.install_hint()`` names what is missing, with the fix: no PyTorch, a CPU-only build, a GPU
  hidden by ``CUDA_VISIBLE_DEVICES``, no GPU seen, a build without kernels for the card (sm_120
  on PyTorch < 2.7); empty when it works;
* ``mufasa-pose3d gpu-check`` prints it; the 3-D form's greyed-out solver choices carry it;
* README and docs say how to install.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import tomllib
import types
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def fake_torch(*, cuda_build="13.0", available=True, cap=(12, 0), archs=("sm_80", "sm_90", "sm_100", "sm_120"),
               version="2.11.0"):
    t = types.SimpleNamespace()
    t.__version__ = version
    t.version = types.SimpleNamespace(cuda=cuda_build)
    t.cuda = types.SimpleNamespace(is_available=lambda: available, get_device_capability=lambda i=0: cap,
                                   get_arch_list=lambda: list(archs),
                                   get_device_name=lambda i=0: "NVIDIA GeForce RTX 5070 Ti")
    return t


def main() -> int:
    from packaging.requirements import Requirement
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    reqs = {k: [Requirement(r) for r in v if r.startswith("torch")] for k, v in extras.items()}
    ok = all(len(reqs.get(k, [])) == 1 and reqs[k][0].specifier.contains("2.7.0")
             and not reqs[k][0].specifier.contains("2.6.0") for k in ("pose3d", "gpu", "all"))
    check(f"extras: pose3d {extras.get('pose3d')}; gpu and all at torch>=2.7 (no 2.4–2.6: no Blackwell kernels)",
          ok and extras["pose3d"] == ["torch>=2.7"])

    from mufasa.pose3d import gpu
    real, env = gpu.torch, os.environ.get("CUDA_VISIBLE_DEVICES")
    hints = {}
    try:
        gpu.torch = None
        hints["none"] = gpu.install_hint()
        gpu.torch = fake_torch(cuda_build=None, available=False)
        hints["cpu build"] = gpu.install_hint()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu.torch = fake_torch(available=False)
        hints["hidden"] = gpu.install_hint()
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        hints["no gpu"] = gpu.install_hint()
        gpu.torch = fake_torch(version="2.6.0", archs=("sm_80", "sm_86", "sm_90"))
        hints["no kernels"] = gpu.install_hint()
        gpu.torch = fake_torch()
        hints["works"] = gpu.install_hint()
    finally:
        gpu.torch = real
        if env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env
    check(f"no PyTorch: {hints['none'][:80]}…", "not installed" in hints["none"] and "mufasa[pose3d]" in hints["none"]
          and "cu128" in hints["none"])
    check(f"CPU-only build: {hints['cpu build'][:70]}…", "CPU-only build" in hints["cpu build"])
    check(f"hidden GPU: {hints['hidden']}", "CUDA_VISIBLE_DEVICES" in hints["hidden"] and "unset" in hints["hidden"])
    check(f"no GPU seen: {hints['no gpu'][:70]}…", "sees no GPU" in hints["no gpu"] and "driver" in hints["no gpu"])
    check(f"RTX 5070 Ti on PyTorch 2.6: {hints['no kernels'][:90]}…",
          "no kernels" in hints["no kernels"] and "sm_120" in hints["no kernels"] and "2.7" in hints["no kernels"])
    check("works: no hint", hints["works"] == "")

    # the CLI
    from mufasa.tools import pose3d_cli
    out = io.StringIO()
    try:
        gpu.torch = None
        with redirect_stdout(out):
            rc = pose3d_cli.main(["gpu-check"])
    finally:
        gpu.torch = real
    check(f"gpu-check without PyTorch: exit {rc}, says what to install", rc == 1 and "mufasa[pose3d]" in out.getvalue())

    # the form
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
    from mufasa.ui_qt.forms.pose3d import Pose3DInferenceForm
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "p"
        root.mkdir()
        ProjectPaths(root).ensure_skeleton()
        write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "d"})
        try:
            gpu.torch = None
            f = Pose3DInferenceForm(config_path=str(root / PROJECT_CONFIG_FILENAME))
        finally:
            gpu.torch = real
        m = f.solver.model()
        off = [(f.solver.itemText(i), m.item(i).isEnabled(), m.item(i).toolTip()) for i in range(f.solver.count())]
    check(f"form without PyTorch: {[(t, e) for t, e, _ in off]}; the greyed-out choices say why",
          not off[1][1] and not off[2][1] and off[3][1] and "mufasa[pose3d]" in off[1][2] and "mufasa[pose3d]" in off[2][2]
          and "not installed" in f.solver.toolTip())

    readme = (REPO / "README.md").read_text()
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("README and docs: the pose3d extra and how to install PyTorch",
          "| `pose3d` |" in readme and ".[qt,pose3d]" in readme and "mufasa[pose3d]" in doc and "cu128" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ld_install_deps: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
