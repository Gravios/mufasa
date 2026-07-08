"""
tests/smoke_122gn_fdlc_skeleton.py
==================================

Patch 122gn — import the FreeDLC skeleton from the <stem>.fdlc.toml sidecar.

FreeDLC writes a per-file skeleton sidecar (<stem>.fdlc.toml) next to the
<stem>.fdlc.parquet, in the same folder as the video. This patch:

* read_fdlc_skeleton() resolves the sidecar from either path and parses it
  tolerantly — [skeleton] table or top-level; edges as name pairs or integer
  index pairs (mapped through nodes); missing/malformed -> None (graceful).
* project_layout.write_skeleton/read_skeleton persist a project-wide
  [skeleton] (nodes + name-pair edges) in project.toml.
* FDLCParquetImporter._import_skeleton (called from run()) discovers the
  sidecar, drops edges referencing unknown body-parts, and writes the
  skeleton to project.toml. Absent sidecar -> nodes-only (backward compat).
* pose_viewer main() gains --config to use the project skeleton instead of
  the built-in DEFAULT_SKELETON_EDGES.

Functional checks run in-sandbox (project_layout imports cleanly;
read_fdlc_skeleton is exec'd from source since the module needs heavy deps).
"""
import ast
import os
import sys
import tempfile
import tomllib
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
IMP = REPO / "mufasa" / "pose_importers" / "fdlc_parquet_importer.py"
VIEWER = REPO / "mufasa" / "tools" / "pose_viewer.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _load_reader(src):
    tree = ast.parse(src)
    ns = {"os": os, "warnings": warnings, "tomllib": tomllib}
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("FDLC"):
            exec(compile(ast.Module([n], []), "<c>", "exec"), ns)
        if isinstance(n, ast.FunctionDef) and n.name == "read_fdlc_skeleton":
            exec(compile(ast.Module([n], []), "<f>", "exec"), ns)
    return ns["read_fdlc_skeleton"]


def main():
    src = IMP.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122gn_fdlc_skeleton: 0/9 checks passed")
        return 1
    check("fdlc_parquet_importer.py parses", True)

    rd = _load_reader(src)
    d = Path(tempfile.mkdtemp())

    (d / "a.fdlc.toml").write_text(
        'schema_version=1\nmodel_id="m"\n'
        'bodyparts=["nose","headmid","neck"]\n'
        'skeleton=[["nose","headmid"],["headmid","neck"]]\n'
    )
    r = rd(str(d / "a.fdlc.parquet"))  # resolve sidecar from parquet path
    check("real FreeDLC schema (top-level bodyparts + skeleton list) parsed",
          r is not None and r["nodes"] == ["nose", "headmid", "neck"]
          and r["edges"] == [("nose", "headmid"), ("headmid", "neck")])

    # tolerant fallback: [skeleton] table with nodes/edges also works
    (d / "b.fdlc.toml").write_text('nodes=["a","b","c"]\nedges=[[0,1],[1,2]]\n')
    r2 = rd(str(d / "b.fdlc.toml"))
    check("tolerant fallback: nodes/edges keys + integer index-pair edges",
          r2 is not None and r2["edges"] == [("a", "b"), ("b", "c")])

    check("missing sidecar -> None (graceful, nodes-only fallback)",
          rd(str(d / "none.fdlc.parquet")) is None)

    (d / "bad.fdlc.toml").write_text("== not = valid [[[toml")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bad = rd(str(d / "bad.fdlc.toml"))
    check("malformed sidecar -> None + warning", bad is None and len(w) == 1)

    # project_layout persistence round-trip
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        read_skeleton,
        write_skeleton,
    )
    cfg = d / "project.toml"
    cfg.write_text(f"project_layout_version = {PROJECT_LAYOUT_VERSION}\n")
    write_skeleton(cfg, nodes=["nose", "headmid"], edges=[("nose", "headmid")])
    sk = read_skeleton(cfg)
    check("project.toml [skeleton] round-trips (name-pair tuples)",
          sk and sk["nodes"] == ["nose", "headmid"]
          and sk["edges"] == [("nose", "headmid")]
          and "[skeleton]" in cfg.read_text())

    check("importer wires _import_skeleton (defined + called in run())",
          "_import_skeleton" in src and "self._import_skeleton()" in src)

    check("skeleton imported verbatim from the toml (edges not filtered/dropped)",
          'edges = sk["edges"]' in src
          and "if a in valid and b in valid" not in src)

    viewer = VIEWER.read_text(encoding="utf-8")
    check("pose_viewer main() gains --config -> read_skeleton -> skeleton_edges",
          '"--config"' in viewer and "read_skeleton" in viewer
          and "skeleton_edges=skeleton_edges" in viewer)

    print(f"smoke_122gn_fdlc_skeleton: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
