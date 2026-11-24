# D3. Testing and patches

## The smoke tests

Mufasa's tests are smoke tests in `tests/smoke_*.py`, about 350 of them, one or more per patch.
Each is a plain script: it builds what it needs (a project, a video, a pose file, often
synthetic), runs the feature end to end, and prints one line per check:

```text
$ python tests/smoke_122lv_cli_commands.py
PASS  mufasa help (and --help): the commands, list's subjects, the other tools, the workbench's options
PASS  mufasa help list: its options
…
smoke_122lv_cli_commands: 24/24 checks passed
```

The exit code is 0 only when every check passes. A check's text says what was checked and, where it
helps, the numbers it found. A failure then explains itself.

### Writing one

```python
"""Smoke test for patch 122xx — what it covers.

User: "the request, in their words".

* what each check establishes, one bullet each.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # Qt without a display
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# … build a project in tempfile.mkdtemp(), run the feature, check …
check(f"the result: {value:.1f} mm (expected about 12)", abs(value - 12) < 0.5)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122xx_name: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
```

* **A temporary project.** Build it in a temporary folder with
  `mufasa.utils.config_creator.ProjectConfigCreator(...)`, and get its paths with
  `project_paths_from_config`.
* **Synthetic data with a known answer.** A video of frames whose grey level is the frame number,
  a rat walking at a known speed, a box flaring a known angle. Then check the numbers, not only
  that the code ran.
* **Optional dependencies.** When the test needs MuJoCo, PyTorch or ultralytics and it isn't
  installed, pass with a check that says so ("MuJoCo not installed: skipped").
* **Qt.** Set `QT_QPA_PLATFORM=offscreen`, create one `QApplication`, and call `processEvents()`
  after acting on widgets. Grab widgets (`widget.grab()`) to check that they drew.
* **Documentation.** End with a check that the docs mention the feature. That keeps the docs and
  the code together.

### Running them

One test:

```bash
QT_QPA_PLATFORM=offscreen python tests/smoke_122lr_viewer_videos.py
```

All of them, two at a time, with a log per test:

```bash
mkdir -p /tmp/suite
ls tests/smoke_*.py | QT_QPA_PLATFORM=offscreen xargs -P 2 -I{} sh -c \
    'python {} > /tmp/suite/$(basename {} .py).log 2>&1; echo "$? $(basename {} .py)"' \
    | tee /tmp/suite/_status.txt
grep -v '^0 ' /tmp/suite/_status.txt          # the failures
```

The whole suite takes 40–60 minutes. Two tests fail without data that isn't in the repository:
`smoke_feature_kernels_realdata` and `smoke_feature_parallel_verify`. A test that times out under
load can pass on its own; rerun it alone before treating it as a failure.

**Which tests to run for a change.** A small change limited to one area needs only the tests that
use it (`grep -l module_name tests/*.py`). Run the full suite for changes to shared code:

* `project_layout.py`;
* the workbench (`workbench.py`, `frame_scrubber.py`);
* the 3-D pipeline;
* anything with a test of its own that guards the whole codebase (the lint and invariant tests
  below).

## Lint

[ruff](https://docs.astral.sh/ruff/) with these rules on every changed file:

```bash
ruff check --select E713,E703,E714,SIM114,SIM300,UP024,UP017,UP010,UP009,UP021,B013,F,E9,I001,UP FILES
```

`tests/smoke_122g?_ruff_*.py` check the codebase as a whole, and `docs/lint_status.md` describes where
it stands.

## Invariants the tests guard

These hold across the codebase, and tests fail if they are broken:

* **Only `mufasa/legacy_layout.py` knows the old SimBA layout.** No other module may name the old
  config file, which is checked by 122eg, 122eh, 122ei and 122eq. Paths go through
  `project_layout`.
* **The command line imports no Qt** (122lv).
* **The docs index**, `docs/README.md`, lists the docs (122cn).
* **The README describes the current state** (122iw): the commands, the pages, the extras.
* **The sidebar's page order** follows the pipeline (122d0, 122jh).

## How changes are made

Mufasa changes in numbered **patches**, each one reviewable change:

1. **One patch, one concern.** A feature, or a fix, with its test and its documentation. A patch
   is named by its number and a tag (`0159` / `122lx`) and described in `CHANGELOG.md`:
   * the request, quoted;
   * what changed;
   * the test and its check count.
2. **The commit message** has a title line and sections headed `== What ==`. It ends with the
   authorship lines.
3. **Tests before the commit.** Run the new test and the tests of what it touches, and the full
   suite for shared code. The commit message says what was run.
4. **Delivered as a patch file.** `git format-patch -1 --start-number N` makes it, and
   `git am` on a clean copy of the previous state checks that it applies.
   * Leave out files whose only change is their line endings (CRLF): this means
     `mufasa/video_processors/extract_frames.py` and
     `mufasa/assets/lookups/feature_extraction_headers.csv`.
5. **Docs with the change.** User-facing changes update the relevant chapter of this guide and the
   topic doc (`docs/pose3d.md`, `docs/cli.md`, …). New docs go into `docs/README.md`.

## Where to read next

* `CHANGELOG.md`: what each patch did, newest first. This is the quickest way to find where a
  feature came from.
* The topic docs (`docs/pose3d.md`, `docs/biomech.md`, `docs/kalman_smoother_design.md`, …): the
  methods behind the features, with the numbers they give.
* `docs/workflow_audit.md` and `docs/workflows.md`: how the workbench's pages, forms and back ends
  connect, and which parts are still being ported.
