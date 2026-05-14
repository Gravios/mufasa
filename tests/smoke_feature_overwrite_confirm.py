"""
tests/smoke_feature_overwrite_confirm.py
=========================================

Originally tested that ``FeatureSubsetsCalculator.preflight_check``
ran ``process_one_video`` on a probe video to discover the
columns the run would produce, then compared against existing
files in ``features_extracted/`` / ``targets_inserted/`` for
column-name overlaps.

Patch 122an (B1) removed the column-collision probe along with
the ``append_to_features_extracted`` / ``append_to_targets_inserted``
kwargs that gated it. v1 writes go to per-family parquet under
``derived_features_dir/<family>/`` where collisions are
structurally impossible.

This test now confirms the removed machinery is gone and the
lighter ``save_dir`` filename-collision check survived.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    pf_src = (REPO_ROOT / "mufasa" / "feature_extractors"
              / "feature_subsets.py").read_text()
    # Scope the "no longer called" assertions to the
    # preflight_check method body. _run_sequential /
    # _run_parallel still call process_one_video for the actual
    # per-video execution — that's correct.
    import ast
    tree = ast.parse(pf_src)
    preflight_src = ""
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "preflight_check"):
            preflight_src = ast.unparse(node)
            break
    if not preflight_src:
        print("FAIL: preflight_check method not found")
        return 1
    code_lines = [
        line for line in preflight_src.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code_src = "\n".join(code_lines)

    checks = 0
    passed = 0

    for label, needle in (
        ("process_one_video no longer called from preflight",
         "process_one_video("),
        ("preflight_scratch_ dir construction removed",
         "preflight_scratch_"),
        ("probe column-count diagnostic removed",
         "probe produced"),
        ("append-mode target iteration removed",
         "for label, dir_path in targets"),
        ("self.features_dir column-collision target removed",
         "('features_extracted', self.features_dir)"),
        ("self.targets_folder column-collision target removed",
         "('targets_inserted', self.targets_folder)"),
    ):
        checks += 1
        if needle not in code_src:
            passed += 1
        else:
            print(f"FAIL: {label} — still present in preflight_check")

    # The lighter save_dir filename-collision check SHOULD survive.
    checks += 1
    if "save_dir/" in pf_src and "file exists" in pf_src:
        passed += 1
    else:
        print("FAIL: save_dir filename-collision check missing")

    print(
        f"smoke_feature_overwrite_confirm: "
        f"{passed}/{checks} checks passed"
    )
    return 0 if passed == checks else 1


if __name__ == "__main__":
    sys.exit(main())
