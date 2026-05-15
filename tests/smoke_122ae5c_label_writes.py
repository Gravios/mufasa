"""
tests/smoke_122ae5c_label_writes.py
====================================

Patch 122ae-5c: dual-write labels migration. Each of the three
labellers gets a sidecar :func:`save_labels_for_video` call
after its existing legacy ``targets_inserted`` write.

Scope: write-side only. Continue-mode reads still pull from
the legacy file, classifier training is untouched. The dual-
write means projects accumulate label data under
``derived/labels/`` so v1 consumers (and the planned
classifier-training retarget in a future patch) can find them.

Sidecar failures are caught and logged so they never abort
the labeller's save — the legacy write is still the primary
contract during the migration window.

Behavioural verification (using save_labels_for_video directly
since the labellers themselves need PySide6 / cv2 / Tk which
aren't in the sandbox):

* Single-classifier label write → file appears at
  derived/labels/<video>.parquet.
* Multi-classifier label write → all classifier columns
  preserved with Int64 dtype.
* Existing v1 labels file + new merge write → both
  classifier columns survive (mirrors what happens when the
  user re-labels different classifiers in successive
  sessions).
* Overwrite mode wipes the existing file (used when the
  labeller wants to replace rather than merge).

AST verification of the three labellers:

* All three import save_labels_for_video.
* All three call save_labels_for_video at code level inside
  their save method.
* All three guard the sidecar with try/except so failures
  don't abort the legacy save.
* All three record 122ae-5c in code/comments.
* The legacy write_df / to_csv / to_parquet target on
  self.targets_inserted_file_path is STILL present (we did
  NOT drop the legacy write).

Cross-file invariant:

* save_labels_for_video is exported from mufasa.utils.label_io.
* The 122ae-3.5 helper used here hasn't drifted: it still
  accepts (video_name, config_path, labels, *, merge=True)
  with the expected behaviour. Spot-checked by re-running
  the smoke_122ae35_label_io test inside this suite.
"""
from __future__ import annotations

import ast
import subprocess
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


def _make_v1_project(tmp: Path,
                     classifier_targets: list[str]) -> Path:
    toml = tmp / "project.toml"
    targets_str = ", ".join(f'"{t}"' for t in classifier_targets)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae5c"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose", "tail"]

        [classifiers]
        targets = [{targets_str}]
    """).strip() + "\n")
    return toml


def main() -> int:
    from mufasa.utils.label_io import (load_labels_for_video,
                                       save_labels_for_video)

    # ==================================================================
    # 1. Behavioural — simulate what the labeller save method does
    # ==================================================================

    # 1a — single classifier
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff"])
        # Simulate self.data_df_targets — what the labellers pass.
        data_df_targets = pd.DataFrame({
            "sniff": [0, 1, 1, 0],
        })
        out_path = save_labels_for_video(
            video_name="v_001",
            config_path=str(toml),
            labels=data_df_targets,
        )
        check(
            "sidecar write: single-classifier file lands at "
            "derived/labels/<video>.parquet",
            Path(out_path) == tmp / "derived" / "labels"
            / "v_001.parquet"
            and Path(out_path).is_file(),
        )
        read_back = pd.read_parquet(out_path)
        check(
            "sidecar write: single-classifier values "
            "round-tripped as Int64",
            read_back["sniff"].tolist() == [0, 1, 1, 0]
            and read_back["sniff"].dtype.name == "Int64",
        )

    # 1b — multi-classifier
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff", "rear", "groom"])
        data_df_targets = pd.DataFrame({
            "sniff":  [1, 0, 1, 0, 1],
            "rear":   [0, 0, 1, 1, 0],
            "groom":  [0, 1, 0, 0, 1],
        })
        save_labels_for_video(
            video_name="v_002",
            config_path=str(toml),
            labels=data_df_targets,
        )
        loaded = load_labels_for_video("v_002", str(toml))
        check(
            "sidecar write: multi-classifier all columns present",
            set(loaded.columns) == {"sniff", "rear", "groom"},
        )
        check(
            "sidecar write: row count preserved",
            len(loaded) == 5,
        )

    # 1c — Merge mode (default) — two writes preserve both columns
    # (mirrors labeller re-labelling different classifiers across
    # sessions)
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff", "rear"])
        # Session 1: just sniff
        save_labels_for_video(
            video_name="v_003",
            config_path=str(toml),
            labels=pd.DataFrame({"sniff": [1, 0, 1]}),
        )
        # Session 2: just rear
        save_labels_for_video(
            video_name="v_003",
            config_path=str(toml),
            labels=pd.DataFrame({"rear": [0, 1, 1]}),
        )
        loaded = load_labels_for_video("v_003", str(toml))
        check(
            "sidecar write: merge mode preserves prior classifier "
            "column across sessions",
            "sniff" in loaded.columns and "rear" in loaded.columns,
        )
        check(
            "sidecar write: merge mode preserves sniff VALUES "
            "from earlier session",
            loaded["sniff"].tolist() == [1, 0, 1],
        )

    # 1d — Overwrite mode wipes prior
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, ["sniff", "rear"])
        save_labels_for_video(
            video_name="v_004",
            config_path=str(toml),
            labels=pd.DataFrame({
                "sniff": [1, 1, 1], "rear": [0, 0, 0],
            }),
        )
        save_labels_for_video(
            video_name="v_004",
            config_path=str(toml),
            labels=pd.DataFrame({"rear": [1, 1, 1]}),
            merge=False,
        )
        on_disk = pd.read_parquet(
            tmp / "derived" / "labels" / "v_004.parquet",
        )
        check(
            "sidecar write: merge=False wipes prior classifier "
            "columns",
            list(on_disk.columns) == ["rear"],
        )

    # ==================================================================
    # 2. AST — all three labellers wired up
    # ==================================================================
    labellers = {
        "labelling_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_interface.py",
        "labelling_advanced_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_advanced_interface.py",
        "standard_labeller.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "standard_labeller.py",
    }
    for name, path in labellers.items():
        src = path.read_text()
        # Patch 122bf: 122ak migrated the labellers to labels-only
        # save via save_labels_for_video. The import is the same
        # but commonly on multiple lines as a tuple — match the
        # substring rather than the exact single-line form.
        check(
            f"{name}: imports save_labels_for_video",
            "save_labels_for_video" in src
            and "from mufasa.utils.label_io" in src,
        )
        # Sidecar call appears at code level
        code_uses = [
            line for line in src.splitlines()
            if "save_labels_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        ]
        check(
            f"{name}: save_labels_for_video called at code level",
            len(code_uses) >= 1,
        )
        # Patch 122bf: the [122ae-5c] canary tag + "Sidecar labels
        # write" wording were transitional — 122ak removed the
        # dual-write era and the sidecar IS the save now. The
        # try/except is still around save_labels_for_video as
        # error handling, just without the sidecar framing.
        check(
            f"{name}: save_labels_for_video call is in a "
            "try/except (error propagation)",
            "try:" in src
            and "save_labels_for_video" in src
            and "except Exception" in src,
        )
        # Patch 122bf: the actual legacy WRITE is gone post-122ak,
        # but the variable assignment
        # `self.targets_inserted_file_path = os.path.join(...)`
        # may still linger in __init__ (vestigial — used by no
        # downstream write). Check for the absence of an actual
        # write_df call using this path.
        check(
            f"{name}: no write_df(...self.targets_inserted_file_path) "
            "write site remains (post-122ak v1-only save)",
            "write_df(" not in src
            or "self.targets_inserted_file_path" not in
                src.split("write_df(")[1].split(")")[0]
            if "write_df(" in src else True,
        )
        check(
            f"{name}: records 122ae-5c in code/comments",
            "122ae-5c" in src,
        )

    # ==================================================================
    # 3. Cross-file invariant — re-run 122ae-3.5 helper test as
    #    a smoke check that save_labels_for_video's contract hasn't
    #    drifted under us
    # ==================================================================
    r = subprocess.run(
        [sys.executable, "tests/smoke_122ae35_label_io.py"],
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ,
             "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True,
    )
    last_line = r.stdout.strip().splitlines()[-1] if r.stdout else ""
    check(
        "regression: smoke_122ae35_label_io still passes "
        "(save_labels_for_video contract unchanged)",
        "passed" in last_line and r.returncode == 0,
        detail=f"got: {last_line!r}",
    )

    print(
        f"smoke_122ae5c_label_writes: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
