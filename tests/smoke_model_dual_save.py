"""
tests/smoke_model_dual_save.py
==============================

Patch 122b: cross-validation of the dual-save model helpers in
:mod:`mufasa.project_layout`. Verifies that:

* ``mirror_model_to_global_cache`` puts a copy in
  ``~/.config/mufasa/models/`` and is idempotent on matching hash.
* ``import_model_into_project`` puts a copy in
  ``<project>/models/<name>/model.npz`` with a sensible ``card.toml``,
  is idempotent on matching hash, and refuses to silently overwrite
  a different-content model with the same name.
* ``resolve_v1_project_root`` finds the project root from a
  ``project_config.ini`` sibling of ``project.toml`` (the post-
  migration transient state) and returns ``None`` for plain
  legacy / unknown paths.

PySide6-free; runs headless. Uses a temporary HOME so the test
doesn't pollute the real ``~/.config/mufasa/models/``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mufasa.project_layout import (  # noqa: E402
    ProjectPaths,
    PROJECT_CONFIG_FILENAME,
    MODEL_CARD_FILENAME,
    MODEL_BLOB_FILENAME,
    file_sha256,
    global_model_cache_dir,
    import_model_into_project,
    mirror_model_to_global_cache,
    read_project_toml,
    resolve_v1_project_root,
    write_project_toml,
)


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _fake_model_npz(path: Path, payload: bytes = b"v2-model-blob") -> None:
    """Write a non-empty file standing in for a real npz. The dual-save
    helpers don't care about the file format; they just hash and copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_v1_project(root: Path) -> ProjectPaths:
    """Build a minimal v1 project skeleton at ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(root)
    paths.ensure_skeleton()
    write_project_toml(root / PROJECT_CONFIG_FILENAME, {
        "project_layout_version": 1,
        "project_name": "smoke",
    })
    return paths


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Redirect HOME so the global cache lands in tmp/home/ instead
        # of polluting the developer's real ~/.config/mufasa/models/.
        fake_home = tmp / "home"
        fake_home.mkdir()
        os.environ["HOME"] = str(fake_home)

        # ----------------------------------------------------------
        # 1. mirror_model_to_global_cache — basic copy
        # ----------------------------------------------------------
        src1 = tmp / "external" / "foo.npz"
        _fake_model_npz(src1, b"payload-A")
        cache_path = mirror_model_to_global_cache(src1)
        check(
            "mirror returns a path",
            cache_path is not None,
            detail=f"got {cache_path!r}",
        )
        check(
            "mirror lands in global cache dir",
            cache_path is not None
            and cache_path.parent == global_model_cache_dir().resolve(),
            detail=f"cache={cache_path}",
        )
        check(
            "mirror preserves filename stem",
            cache_path is not None and cache_path.name == "foo.npz",
        )
        check(
            "mirror copies bytes faithfully",
            cache_path is not None
            and cache_path.read_bytes() == b"payload-A",
        )

        # ----------------------------------------------------------
        # 2. mirror is idempotent on matching hash
        # ----------------------------------------------------------
        cache_path_2 = mirror_model_to_global_cache(src1)
        check(
            "mirror idempotent (same path returned)",
            cache_path == cache_path_2,
        )
        # Modify src to a new hash, re-mirror, ensure cache updates.
        _fake_model_npz(src1, b"payload-A-prime")
        cache_path_3 = mirror_model_to_global_cache(src1)
        check(
            "mirror overwrites cache when src changes",
            cache_path_3 is not None
            and cache_path_3.read_bytes() == b"payload-A-prime",
        )

        # ----------------------------------------------------------
        # 3. mirror handles src already inside the cache
        # ----------------------------------------------------------
        in_cache = global_model_cache_dir() / "already_here.npz"
        _fake_model_npz(in_cache, b"in-cache")
        result = mirror_model_to_global_cache(in_cache)
        check(
            "mirror no-ops when src is already in cache",
            result is not None and result.resolve() == in_cache.resolve(),
        )

        # ----------------------------------------------------------
        # 4. import_model_into_project — basic copy + card
        # ----------------------------------------------------------
        proj_root = tmp / "proj_A"
        _make_v1_project(proj_root)

        src2 = tmp / "external" / "kalman_v2_run42.npz"
        _fake_model_npz(src2, b"payload-B")
        in_proj = import_model_into_project(src2, proj_root)
        check(
            "import returns expected in-project path",
            in_proj == proj_root / "models" / "kalman_v2_run42"
                                          / MODEL_BLOB_FILENAME,
            detail=f"got {in_proj}",
        )
        check(
            "import copies bytes faithfully",
            in_proj.read_bytes() == b"payload-B",
        )

        card = in_proj.parent / MODEL_CARD_FILENAME
        check("card.toml exists alongside model.npz", card.is_file())
        card_data = read_project_toml(card) if False else None
        # card.toml is a project-toml file but lacks
        # project_layout_version. Use tomllib directly to read it.
        import tomllib
        with open(card, "rb") as f:
            card_data = tomllib.load(f)
        check(
            "card carries source_path",
            card_data.get("source_path") == str(src2),
        )
        check(
            "card carries sha256 matching the src",
            card_data.get("sha256") == file_sha256(src2),
        )
        check(
            "card carries model_name",
            card_data.get("model_name") == "kalman_v2_run42",
        )

        # ----------------------------------------------------------
        # 5. import is idempotent on matching hash
        # ----------------------------------------------------------
        in_proj_2 = import_model_into_project(src2, proj_root)
        check(
            "import idempotent on matching hash",
            in_proj == in_proj_2 and in_proj.read_bytes() == b"payload-B",
        )

        # ----------------------------------------------------------
        # 6. import refuses to overwrite different content
        # ----------------------------------------------------------
        src3 = tmp / "external" / "kalman_v2_run42.npz"  # same name
        _fake_model_npz(src3, b"payload-C")  # different content
        raised = False
        try:
            import_model_into_project(src3, proj_root)
        except FileExistsError:
            raised = True
        check(
            "import refuses silent overwrite (different content, same name)",
            raised,
        )
        check(
            "in-project model is unchanged after refused import",
            in_proj.read_bytes() == b"payload-B",
        )

        # ----------------------------------------------------------
        # 7. overwrite=True forces replacement
        # ----------------------------------------------------------
        in_proj_3 = import_model_into_project(
            src3, proj_root, overwrite=True,
        )
        check(
            "import with overwrite=True replaces content",
            in_proj_3.read_bytes() == b"payload-C",
        )

        # ----------------------------------------------------------
        # 8. resolve_v1_project_root — happy path
        # ----------------------------------------------------------
        # Transient post-migration state: both project.toml and
        # project_config.ini live at the same root.
        legacy_ini = proj_root / "project_config.ini"
        legacy_ini.write_text("[General settings]\n")
        resolved = resolve_v1_project_root(str(legacy_ini))
        check(
            "resolve finds v1 root from legacy INI at v1 root",
            resolved == proj_root.resolve(),
            detail=f"got {resolved}",
        )

        # ProjectPaths-style: project.toml passed directly.
        resolved_direct = resolve_v1_project_root(
            str(proj_root / PROJECT_CONFIG_FILENAME),
        )
        check(
            "resolve finds v1 root from project.toml",
            resolved_direct == proj_root.resolve(),
        )

        # Pure-legacy: no project.toml anywhere → None
        legacy_only = tmp / "legacy_only"
        legacy_only.mkdir()
        (legacy_only / "project_config.ini").write_text(
            "[General settings]\n",
        )
        resolved_legacy = resolve_v1_project_root(
            str(legacy_only / "project_config.ini"),
        )
        check(
            "resolve returns None for pure-legacy project",
            resolved_legacy is None,
        )

        # None / empty / unrecognized — graceful None
        check(
            "resolve(None) is None",
            resolve_v1_project_root(None) is None,
        )
        check(
            "resolve('') is None",
            resolve_v1_project_root("") is None,
        )
        check(
            "resolve(weird path) is None",
            resolve_v1_project_root("/tmp/no/such/file.txt") is None,
        )

    print(f"smoke_model_dual_save: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed")
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
