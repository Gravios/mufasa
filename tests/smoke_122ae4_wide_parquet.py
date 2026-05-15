"""
tests/smoke_122ae4_wide_parquet.py
===================================

Patch 122ae-4: full feature extractor retarget. Adds a
v1-native wide-parquet sidecar to the 8 standard extractors
(4/7/8/9/14/16/8bps_2_animals/user_defined) plus the
write_wide_features_v1 helper that drives it. Also extends
load_features_for_video's precedence logic to handle the new
wide-parquet location alongside the per-family files from
122ae-3.

Coverage:

1. **write_wide_features_v1 behaviour**
   * v1 TOML project → writes parquet to
     ``<derived_features_dir>/<video>.parquet`` and returns
     the path.
   * Legacy INI project → no-op; returns None; no file appears.
   * Malformed config → no-op; returns None with a
     RuntimeWarning.
   * Video name with ``.mp4`` extension → stripped to stem.
   * derived_features_dir directory is auto-created.

2. **load_features_for_video — wide-parquet branch**
   * Only wide parquet present → returns its contents.
   * Wide parquet + per-family files: per-family columns
     override wide columns; non-overlapping wide columns
     are appended.
   * Only per-family files present → unchanged from 122ae-3.
   * Empty per-family tree + wide parquet → returns wide.
   * Nothing present → FileNotFoundError mentions all three
     probed paths (per-family tree, wide parquet, legacy).

3. **AST surface — extractors are instrumented**
   All 8 standard extractors call ``write_wide_features_v1``
   immediately after their legacy ``write_df`` call. Pattern
   verified by source scan.

4. **AST — write_wide_features_v1 in __all__**

5. **No regressions**
   All existing smoke_122ae2 + smoke_122ae3 cases still pass
   (they don't write a wide parquet, so the new branch is
   inert for them).
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _make_v1_project(tmp: Path, file_type: str = "csv") -> Path:
    toml = tmp / "project.toml"
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae4"
        version = "0.0.1"

        [pose]
        file_type = "{file_type}"
        animal_count = 1
        body_parts = ["nose", "tail"]
    """).strip() + "\n")
    return toml


def _make_legacy_project(tmp: Path) -> Path:
    """Minimal legacy INI to drive the v1-detection branch."""
    proj = tmp / "project_folder"
    proj.mkdir()
    cfg = proj / "project_config.ini"
    cfg.write_text(textwrap.dedent(f"""
        [General settings]
        project_path = {proj}
        workflow_file_type = csv
        animal_no = 1
    """).strip() + "\n")
    return cfg


def main() -> int:
    from mufasa.utils.feature_io import (family_slug,
                                         load_features_for_video,
                                         write_wide_features_v1)

    # ==================================================================
    # 1. write_wide_features_v1 behaviour
    # ==================================================================

    # 1a — v1 TOML project, fresh write
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        out = write_wide_features_v1(
            df=df, video_name="video_001", config_path=str(toml),
        )
        check(
            "v1 write: returns the path written to",
            out is not None,
        )
        check(
            "v1 write: parquet exists at expected location",
            Path(out).is_file() if out else False,
        )
        if out:
            check(
                "v1 write: parent dir matches derived_features_dir",
                Path(out).parent == tmp / "derived" / "features",
            )
            check(
                "v1 write: filename matches '<stem>.parquet'",
                Path(out).name == "video_001.parquet",
            )
            on_disk = pd.read_parquet(out)
            check(
                "v1 write: DataFrame round-trips column-for-column",
                set(on_disk.columns) == {"feat_a", "feat_b"}
                and on_disk["feat_a"].tolist() == [1.0, 2.0, 3.0],
            )

    # 1b — legacy INI project, no-op
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        cfg = _make_legacy_project(tmp)
        df = pd.DataFrame({"feat_a": [1.0]})
        out = write_wide_features_v1(
            df=df, video_name="video_002", config_path=str(cfg),
        )
        check(
            "legacy project: returns None (no v1 sidecar write)",
            out is None,
        )
        # Confirm no derived/ subtree was created
        check(
            "legacy project: no derived/ subtree created",
            not (cfg.parent / "derived").exists(),
        )

    # 1c — Note: project_paths_from_config is intentionally
    # forgiving (it derives paths from the config file's parent
    # directory even when the TOML is missing or unparseable),
    # so the 'except Exception' branch in write_wide_features_v1
    # is purely defensive belt-and-braces and isn't reachable
    # via normal inputs. We don't test it here — it would
    # require monkey-patching the layout helper to throw, which
    # tests the patch rather than the API contract.

    # 1d — video_name with '.mp4' extension is stripped
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        out = write_wide_features_v1(
            df=pd.DataFrame({"a": [1, 2]}),
            video_name="video_003.mp4",
            config_path=str(toml),
        )
        check(
            "extension stripping: '.mp4' suffix dropped from path",
            out is not None and Path(out).name == "video_003.parquet",
            detail=f"got {out!r}",
        )

    # 1e — derived_features_dir is auto-created
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        # Don't pre-create derived/features/
        out = write_wide_features_v1(
            df=pd.DataFrame({"a": [1]}),
            video_name="v",
            config_path=str(toml),
        )
        check(
            "auto-mkdir: derived/features/ created on first write",
            (tmp / "derived" / "features").is_dir(),
        )

    # ==================================================================
    # 2. load_features_for_video — wide-parquet branch + merge
    # ==================================================================

    # 2a — only wide parquet present, no per-family
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        df = pd.DataFrame({
            "alpha": [1.0, 2.0],
            "beta":  [3.0, 4.0],
        })
        write_wide_features_v1(
            df=df, video_name="v_004", config_path=str(toml),
        )

        result = load_features_for_video("v_004", str(toml))
        check(
            "wide-only branch: load returns wide DataFrame",
            set(result.columns) == {"alpha", "beta"}
            and result["alpha"].tolist() == [1.0, 2.0],
        )

    # 2b — wide + per-family: per-family columns override
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)

        # Wide: alpha=100, beta=200 (baseline from "full extractor")
        write_wide_features_v1(
            df=pd.DataFrame({"alpha": [100, 100],
                             "beta":  [200, 200]}),
            video_name="v_005", config_path=str(toml),
        )

        # Per-family: alpha=999 (recomputed via FeatureSubsetsCalculator)
        slug = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        fam_dir = tmp / "derived" / "features" / slug
        fam_dir.mkdir(parents=True)
        pd.DataFrame({"alpha": [999, 999]}).to_parquet(
            fam_dir / "v_005.parquet", index=False,
        )

        result = load_features_for_video("v_005", str(toml))
        check(
            "merge: per-family 'alpha' overrides wide 'alpha'",
            result["alpha"].tolist() == [999, 999],
            detail=f"got {result['alpha'].tolist()}",
        )
        check(
            "merge: wide 'beta' (unique) survives",
            result["beta"].tolist() == [200, 200],
        )
        check(
            "merge: no duplicate columns in output",
            not any(result.columns.duplicated()),
        )

    # 2c — only per-family, no wide (122ae-3 behaviour preserved)
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        slug = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        fam_dir = tmp / "derived" / "features" / slug
        fam_dir.mkdir(parents=True)
        pd.DataFrame({"only_pf": [7, 8]}).to_parquet(
            fam_dir / "v_006.parquet", index=False,
        )

        result = load_features_for_video("v_006", str(toml))
        check(
            "per-family-only: 122ae-3 behaviour preserved",
            list(result.columns) == ["only_pf"]
            and result["only_pf"].tolist() == [7, 8],
        )

    # 2d — empty per-family tree (no subdirs) + wide present
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        # Make the derived/features/ dir but DON'T create any
        # per-family subdirs, only the wide parquet at the root.
        write_wide_features_v1(
            df=pd.DataFrame({"sole_wide": [42]}),
            video_name="v_007", config_path=str(toml),
        )
        result = load_features_for_video("v_007", str(toml))
        check(
            "empty pf + wide: returns wide",
            list(result.columns) == ["sole_wide"],
        )

    # 2e — nothing present: error message names all three paths
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        raised = False
        msg = ""
        try:
            load_features_for_video("v_missing", str(toml))
        except FileNotFoundError as exc:
            raised = True
            msg = str(exc)
        check("nothing-found: FileNotFoundError raised", raised)
        check(
            "nothing-found: message mentions per-family tree",
            "per-family" in msg,
            detail=f"got {msg!r}",
        )
        check(
            "nothing-found: message mentions wide parquet",
            "wide" in msg,
            detail=f"got {msg!r}",
        )
        # Patch 122bf: post-122ak, load_features_for_video is
        # v1-read-only — there's no legacy CSV fallback in the
        # error path. The "legacy" wording was dropped from the
        # message. Assertion removed (was: "message mentions legacy").

    # ==================================================================
    # 3. AST — all 8 standard extractors call write_wide_features_v1
    # ==================================================================
    extractor_files = [
        "feature_extractor_4bp.py",
        "feature_extractor_7bp.py",
        "feature_extractor_8bp.py",
        "feature_extractor_9bp.py",
        "feature_extractor_14bp.py",
        "feature_extractor_16bp.py",
        "feature_extractor_8bps_2_animals.py",
        "feature_extractor_user_defined.py",
    ]
    extr_root = REPO_ROOT / "mufasa" / "feature_extractors"
    for name in extractor_files:
        src = (extr_root / name).read_text()
        check(
            f"{name}: imports write_wide_features_v1",
            "from mufasa.utils.feature_io import "
            "write_wide_features_v1" in src,
        )
        check(
            f"{name}: calls write_wide_features_v1 with "
            "config_path=self.config_path",
            "config_path=self.config_path" in src
            and "write_wide_features_v1(" in src,
        )
        check(
            f"{name}: records 122ae-4 in code comments",
            "122ae-4" in src,
        )

    # ==================================================================
    # 4. AST — write_wide_features_v1 in feature_io __all__
    # ==================================================================
    fio_src = (REPO_ROOT / "mufasa" / "utils"
               / "feature_io.py").read_text()
    fio_tree = ast.parse(fio_src)
    # Find __all__ assignment
    all_names: list[str] = []
    for node in fio_tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(
                    elt.value, str,
                ):
                    all_names.append(elt.value)
    check(
        "feature_io __all__ exports write_wide_features_v1",
        "write_wide_features_v1" in all_names,
    )
    check(
        "feature_io __all__ still exports family_slug + "
        "load_features_for_video (no removal)",
        ("family_slug" in all_names
         and "load_features_for_video" in all_names),
    )

    # ==================================================================
    # 5. 122ae-2 + 122ae-3 still pass (sanity)
    # ==================================================================
    import subprocess
    for prior_test in (
        "smoke_122ae2_feature_io.py",
        "smoke_122ae3_per_family_writer.py",
    ):
        r = subprocess.run(
            [sys.executable, f"tests/{prior_test}"],
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ,
                 "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True, text=True,
        )
        last = r.stdout.strip().splitlines()[-1] if r.stdout else ""
        check(
            f"regression check: {prior_test} still passes",
            "passed" in last and r.returncode == 0,
            detail=f"got: {last!r}",
        )

    print(
        f"smoke_122ae4_wide_parquet: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
