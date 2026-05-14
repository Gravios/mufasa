"""
tests/smoke_122ae5b_non_labeller_readers.py
=============================================

Patch 122ae-5b: migrate the four remaining feature-readers
identified by the 122ae-5 audit. None of them surface to
end-users as urgently as the labellers (which 122ae-5 fixed),
but they all break end-to-end v1 flows otherwise:

  * mufasa/model/inference_batch.py — batch inference scans
    the legacy csv/features_extracted/ directory; for v1
    projects (features under derived/features/) it returned
    Zero files found before this patch.
  * mufasa/roi_tools/ROI_feature_analyzer.py — append-data
    branch reads per-video features from the legacy path.
  * mufasa/third_party_label_appenders/BENTO_appender.py —
    same per-video read pattern.
  * mufasa/data_processors/directing_other_animals_calculator.py —
    append-bool-to-features branch reads per-video features.

Plus a new discovery helper:

  * mufasa.utils.feature_io.list_video_stems_with_features
    — returns the sorted union of video stems for which
    features exist anywhere in the project (wide-parquet
    sidecar, per-family subdirs, OR legacy CSV). Needed by
    inference_batch which previously did a directory scan
    that only found legacy files.

Behavioural coverage:

* list_video_stems_with_features
  - Wide parquet only: stems discovered.
  - Per-family only: stems discovered (deduplicated across
    multiple family subdirs).
  - Legacy CSV only: stems discovered.
  - All three layouts: stems unioned without duplicates.
  - Empty project: returns [].
  - Malformed config: returns [].

AST coverage for the 4 retargeted modules:

* Each imports load_features_for_video.
* Each replaces a read_df(...features_extracted...) call site
  with load_features_for_video at code level.
* Each records 122ae-5b in code/comments.
* inference_batch additionally imports list_video_stems_with_features
  and uses it in the no-features_dir branch.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
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


def _make_v1_project(tmp: Path) -> Path:
    toml = tmp / "project.toml"
    toml.write_text(textwrap.dedent("""
        project_layout_version = 1

        [project]
        name = "smoke_122ae5b"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose", "tail"]
    """).strip() + "\n")
    return toml


def main() -> int:
    from mufasa.utils.feature_io import (family_slug,
                                         list_video_stems_with_features,
                                         write_wide_features_v1)

    # ==================================================================
    # 1. list_video_stems_with_features — discovery helper
    # ==================================================================

    # 1a — Wide parquet only
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        write_wide_features_v1(
            df=pd.DataFrame({"a": [1]}),
            video_name="v_alpha", config_path=str(toml),
        )
        write_wide_features_v1(
            df=pd.DataFrame({"a": [1]}),
            video_name="v_beta", config_path=str(toml),
        )
        stems = list_video_stems_with_features(str(toml))
        check(
            "discovery: wide-parquet-only branch finds stems",
            stems == ["v_alpha", "v_beta"],
            detail=f"got {stems}",
        )

    # 1b — Per-family only
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        slug_a = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        slug_b = family_slug(
            "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)",
        )
        (tmp / "derived" / "features" / slug_a).mkdir(parents=True)
        (tmp / "derived" / "features" / slug_b).mkdir(parents=True)
        for stem in ["v_pf_one", "v_pf_two"]:
            pd.DataFrame({"x": [1]}).to_parquet(
                tmp / "derived" / "features" / slug_a
                / f"{stem}.parquet",
                index=False,
            )
        # v_pf_one ALSO appears in slug_b — should dedupe.
        pd.DataFrame({"y": [1]}).to_parquet(
            tmp / "derived" / "features" / slug_b
            / "v_pf_one.parquet",
            index=False,
        )
        stems = list_video_stems_with_features(str(toml))
        check(
            "discovery: per-family only branch finds stems",
            stems == ["v_pf_one", "v_pf_two"],
            detail=f"got {stems}",
        )

    # 1c — Legacy CSV only
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        for stem in ["v_legacy1", "v_legacy2"]:
            pd.DataFrame({"a": [1]}).to_csv(
                legacy_dir / f"{stem}.csv", index=False,
            )
        # Also drop a hidden file that should be ignored.
        (legacy_dir / ".DS_Store").write_text("")
        stems = list_video_stems_with_features(str(toml))
        check(
            "discovery: legacy-CSV-only branch finds stems",
            stems == ["v_legacy1", "v_legacy2"],
            detail=f"got {stems}",
        )

    # 1d — All three layouts, union without duplicates
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        # Wide-parquet for v_alpha, v_beta
        write_wide_features_v1(
            df=pd.DataFrame({"a": [1]}),
            video_name="v_alpha", config_path=str(toml),
        )
        write_wide_features_v1(
            df=pd.DataFrame({"a": [1]}),
            video_name="v_beta", config_path=str(toml),
        )
        # Per-family for v_beta (also exists in wide) + v_gamma
        slug = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        (tmp / "derived" / "features" / slug).mkdir(parents=True)
        for stem in ["v_beta", "v_gamma"]:
            pd.DataFrame({"x": [1]}).to_parquet(
                tmp / "derived" / "features" / slug
                / f"{stem}.parquet",
                index=False,
            )
        # Legacy CSV for v_alpha (also exists in wide) + v_delta
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        for stem in ["v_alpha", "v_delta"]:
            pd.DataFrame({"a": [1]}).to_csv(
                legacy_dir / f"{stem}.csv", index=False,
            )
        stems = list_video_stems_with_features(str(toml))
        check(
            "discovery: all-three-layouts branch UNIONs stems",
            stems == ["v_alpha", "v_beta", "v_delta", "v_gamma"],
            detail=f"got {stems}",
        )

    # 1e — Empty project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        stems = list_video_stems_with_features(str(toml))
        check(
            "discovery: empty project returns []",
            stems == [],
        )

    # 1f — Malformed config path
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        bad = tmp / "nope" / "missing.toml"   # path doesn't exist
        stems = list_video_stems_with_features(str(bad))
        check(
            "discovery: malformed config returns []",
            stems == [],
        )

    # ==================================================================
    # 2. AST — the 4 retargeted modules
    # ==================================================================
    retargeted = {
        "inference_batch.py":
            REPO_ROOT / "mufasa" / "model" / "inference_batch.py",
        "ROI_feature_analyzer.py":
            REPO_ROOT / "mufasa" / "roi_tools"
            / "ROI_feature_analyzer.py",
        "BENTO_appender.py":
            REPO_ROOT / "mufasa" / "third_party_label_appenders"
            / "BENTO_appender.py",
        "directing_other_animals_calculator.py":
            REPO_ROOT / "mufasa" / "data_processors"
            / "directing_other_animals_calculator.py",
    }
    for name, path in retargeted.items():
        src = path.read_text()
        check(
            f"{name}: imports load_features_for_video",
            "load_features_for_video" in src,
        )
        # The actual call appears at code level (not just imports
        # / comments).
        code_uses = [
            line for line in src.splitlines()
            if "load_features_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        ]
        check(
            f"{name}: load_features_for_video used at code level",
            len(code_uses) >= 1,
            detail=f"got {len(code_uses)} call(s)",
        )
        check(
            f"{name}: records 122ae-5b in code/comments",
            "122ae-5b" in src,
        )

    # inference_batch-specific: uses the discovery helper
    ib_src = retargeted["inference_batch.py"].read_text()
    check(
        "inference_batch: imports list_video_stems_with_features",
        "list_video_stems_with_features" in ib_src,
    )
    check(
        "inference_batch: calls list_video_stems_with_features "
        "at code level",
        any(
            "list_video_stems_with_features(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
            for line in ib_src.splitlines()
        ),
    )

    # ==================================================================
    # 3. No regressions on prior tests' modules — quick spot check
    # ==================================================================
    fio_src = (REPO_ROOT / "mufasa" / "utils"
               / "feature_io.py").read_text()
    fio_tree = ast.parse(fio_src)
    # Confirm list_video_stems_with_features is in __all__
    for node in fio_tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            all_names = [
                e.value for e in node.value.elts
                if isinstance(e, ast.Constant)
                and isinstance(e.value, str)
            ]
            check(
                "feature_io __all__ exports "
                "list_video_stems_with_features",
                "list_video_stems_with_features" in all_names,
            )
            check(
                "feature_io __all__ still exports family_slug + "
                "load_features_for_video + write_wide_features_v1 "
                "(no removal)",
                all(
                    n in all_names
                    for n in (
                        "family_slug",
                        "load_features_for_video",
                        "write_wide_features_v1",
                    )
                ),
            )
            break

    print(
        f"smoke_122ae5b_non_labeller_readers: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
