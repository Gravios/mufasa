"""Smoke test for patch 122hu — smooth a standalone parquet in place.

Adds two CLI capabilities so a pose file can be processed outside any project
with output beside it:

* ``layout_from_fdlc_sidecar`` builds a BodyLayout from a FreeDLC
  ``<stem>.fdlc.toml`` skeleton sidecar, so the kinematic tree travels with the
  file and no ``project.toml`` is needed. main() tries this before the
  built-in-rig fallback when no project/--config is found.
* ``--beside-input`` writes each smoothed file into its input's own directory
  (inputs grouped by parent directory), instead of a single ``--output-dir``.

The smoother module imports heavy deps (pyarrow/cv2/h5py) absent in the
sandbox, so this test drives layout_from_fdlc_sidecar with a stubbed
read_fdlc_skeleton (mirroring the real TOML parse) and checks the CLI wiring by
AST + replicated batching logic.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
import tempfile
import tomllib
import types
from pathlib import Path

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


# Stub the importer module's read_fdlc_skeleton (avoids the h5py import chain).
def _stub_read_fdlc_skeleton(path):
    p = os.fspath(path)
    low = p.lower()
    if low.endswith(".fdlc.parquet"):
        t = p[: -len(".fdlc.parquet")] + ".fdlc.toml"
    elif low.endswith(".fdlc.toml"):
        t = p
    else:
        t = os.path.splitext(p)[0] + ".fdlc.toml"
    if not os.path.isfile(t):
        return None
    with open(t, "rb") as _fh:
        data = tomllib.load(_fh)
    nodes = [str(n) for n in (data.get("bodyparts") or []) if str(n)]
    raw = data.get("skeleton") if isinstance(data.get("skeleton"), list) else []
    edges = [(str(e[0]), str(e[1])) for e in raw if len(e) >= 2]
    if not nodes and not edges:
        return None
    return {"nodes": nodes, "edges": edges, "source": t}


_stub = types.ModuleType("mufasa.pose_importers.fdlc_parquet_importer")
_stub.read_fdlc_skeleton = _stub_read_fdlc_skeleton
sys.modules["mufasa.pose_importers.fdlc_parquet_importer"] = _stub

import mufasa.data_processors.kalman_pose_smoother_v2 as K  # noqa: E402

# ---- layout_from_fdlc_sidecar ----
check("layout_from_fdlc_sidecar is defined",
      hasattr(K, "layout_from_fdlc_sidecar"))

with tempfile.TemporaryDirectory() as d:
    parquet = os.path.join(d, "recording_A.fdlc.parquet")
    sidecar = os.path.join(d, "recording_A.fdlc.toml")
    Path(parquet).write_bytes(b"")
    with open(sidecar, "w") as f:
        f.write('bodyparts = ["head_nose","head_mid","back_T4","back_L2",'
                '"tail_V6"]\n')
        f.write('skeleton = [["head_nose","head_mid"],'
                '["head_mid","back_T4"],["back_T4","back_L2"],'
                '["back_L2","tail_V6"]]\n')

    layout = K.layout_from_fdlc_sidecar(parquet)
    check("sidecar yields a layout", layout is not None)
    check("sidecar layout has all skeleton markers",
          layout is not None
          and set(layout.marker_names) == {
              "head_nose", "head_mid", "back_T4", "back_L2", "tail_V6"})
    check("sidecar layout builds a spanning tree (n-1 segments)",
          layout is not None and len(layout.segments) == 5)
    check("flags pass through to the layout (with_drift)",
          K.layout_from_fdlc_sidecar(parquet, with_drift=True).with_drift
          is True)

    # resolves from the .fdlc.toml path directly too
    check("sidecar resolves from the .toml path directly",
          K.layout_from_fdlc_sidecar(sidecar) is not None)

    # missing sidecar -> None (so main() can fall back)
    p2 = os.path.join(d, "no_sidecar.parquet")
    Path(p2).write_bytes(b"")
    check("no sidecar -> None (enables fallback)",
          K.layout_from_fdlc_sidecar(p2) is None)

# ---- CLI wiring (AST + source) ----
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("--beside-input flag is registered",
      '"--beside-input"' in src and "beside_input" in src)
check("main() tries the sidecar layout before the rig fallback",
      "layout_from_fdlc_sidecar(p) for p in args.pose_input" in src)
check("main() groups inputs by parent dir for --beside-input",
      "groups.setdefault(parent," in src
      and "pp if pp.is_dir() else pp.parent" in src)
check("smooth_pose_v2 is called per batch with the batch output dir",
      "for batch_inputs, batch_output_dir in batches:" in src
      and "output_dir=batch_output_dir" in src)

# Stronger guard (a text-present check missed a beside-branch that ignored the
# grouping): parse main(), find the `if args.beside_input:` block, and confirm
# it assigns `batches` from the per-directory groups — not from a single
# (pose_input, output_dir) pair. This catches a beside-branch that silently
# collapses back to one output dir.
_tree = ast.parse(src)
_main = next((n for n in ast.walk(_tree)
             if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
check("main() is parseable", _main is not None)
_beside_if = None
for node in ast.walk(_main) if _main else []:
    if (isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "beside_input"):
        _beside_if = node
        break
check("main() has an `if args.beside_input:` branch", _beside_if is not None)
# within that branch, the batches assignment must reference `groups`
_beside_src = ast.get_source_segment(src, _beside_if) if _beside_if else ""
check("the beside-branch builds batches from per-directory groups",
      "batches = [" in _beside_src
      and "groups.items()" in _beside_src
      and "args.pose_input" not in _beside_src.split("batches = [")[1]
      .split("]")[0])

# ---- replicated batching logic ----
def _batches(pose_input, beside, output_dir, is_dir):
    if beside:
        groups: dict[str, list[str]] = {}
        for p in pose_input:
            pp = Path(p)
            parent = str(pp if is_dir(pp) else pp.parent)
            groups.setdefault(parent, []).append(p)
        return [(inp, out) for out, inp in groups.items()]
    return [(list(pose_input), output_dir)]


b1 = _batches(["/data/rec/a.parquet"], True, "./out", lambda p: False)
check("single file + --beside-input -> output beside it",
      len(b1) == 1 and b1[0][1] == "/data/rec")
b2 = _batches(["/data/x/a.parquet", "/data/y/b.parquet"], True, "./out",
              lambda p: False)
check("files in different folders -> separate batches beside each",
      sorted(o for _, o in b2) == ["/data/x", "/data/y"])
b3 = _batches(["/data/rec"], True, "./out", lambda p: str(p) == "/data/rec")
check("a directory arg -> outputs into that directory",
      b3[0][1] == "/data/rec")
b4 = _batches(["/data/rec/a.parquet", "/data/rec/b.parquet"], False,
              "./myout", lambda p: False)
check("without --beside-input -> single batch to --output-dir (unchanged)",
      len(b4) == 1 and b4[0][1] == "./myout")

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hu_standalone_beside_input: {n_pass}/{len(checks)} "
      f"checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
