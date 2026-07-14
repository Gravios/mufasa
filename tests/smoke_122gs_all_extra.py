"""
tests/smoke_122gs_all_extra.py
==============================

Patch 122gs — a robust ``[all]`` install extra.

`pip install -e .[all]` should reliably install everything (Qt front-end,
GPU, ONNX, dev tooling). The previous ``all`` was a self-referential
``mufasa[qt,gpu,onnx,dev]`` extra, which is fragile in editable installs /
on older pip. It is now a flat, self-contained list, kept a superset of the
individual extras.

Checks: parse pyproject.toml and assert ``all`` (1) has no ``mufasa[...]``
self-reference and (2) contains every package from qt/gpu/onnx/dev.
"""
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    check("an [all] extra exists", "all" in extras)
    all_pkgs = extras.get("all", [])

    check("[all] has no self-referential mufasa[...] entry",
          not any("mufasa[" in x for x in all_pkgs))

    all_set = set(all_pkgs)
    for grp in ("qt", "gpu", "onnx", "dev"):
        missing = set(extras.get(grp, [])) - all_set
        check(f"[all] is a superset of [{grp}]", not missing,
              detail=f"missing {missing}")

    print(f"smoke_122gs_all_extra: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
