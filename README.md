# Mufasa

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![GUI: PySide6 / Qt 6.8](https://img.shields.io/badge/GUI-PySide6%2FQt%206.8-green.svg)](https://www.qt.io/)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-orange.svg)](#)

Mufasa is a fork of [SimBA (sgoldenlab/simba)](https://github.com/sgoldenlab/simba) targeting Linux, Python ≥ 3.11, and PySide6 (Qt 6.8). The legacy Tkinter GUI has been retired; the Qt workbench is the only supported entry surface.

It provides:

* **Pose-data import** from many trackers (DLC, SLEAP, MARS, TRK, FaceMap, YOLO, SuperAnimal-TopView, SimBA blob).
* **Pipeline stages** — outlier correction, feature extraction, supervised classifier training and inference, ROI analysis, directionality, classification visualization.
* **Two project layouts:**
  * Legacy SimBA (`project_config.ini`-driven) — preserved for compatibility with existing projects.
  * v1 (`project.toml`-driven, run-id-scoped) — the new layout with proper run separation and cleaner directory grouping.

## Running Mufasa

After `pip install -e .`, three console entry points are installed:

| Command | What it does |
|---|---|
| `mufasa` | Smart launcher with environment diagnostics; opens the Qt workbench. |
| `mufasa-workbench` | Direct workbench launch (no env diagnostics). |
| `mufasa-chooser` | Legacy Qt chooser / standalone popups. |
| `mufasa-migrate-project` | Legacy SimBA layout → v1 layout migration tool. |

The legacy `mufasa-tk` Tkinter launcher was **Removed in patch 122d4**; the `python -m mufasa.SimBA` entry point followed in patch 122d5. The Qt workbench replaces both.

Migrate a legacy SimBA-style project to v1:

```bash
mufasa-migrate-project /path/to/legacy_project          # dry run
mufasa-migrate-project /path/to/legacy_project --commit # actually move files
```

(Same code is also reachable as `python -m mufasa.cli.migrate_project` for installs where the console script isn't on `$PATH`.)

See [`docs/migration_guide.md`](docs/migration_guide.md) for the full workflow.

## Project layout

Mufasa supports two project layouts; detection is by config-file extension (`.toml` → v1, `.ini` → legacy).

**v1 layout:**

```
my_project/
├── project.toml                ← layout marker + metadata
├── sources/
│   ├── videos/
│   ├── pose/
│   ├── annotations/            (optional, for training)
│   └── video_info.csv
├── derived/
│   ├── outlier_corrected/<run_id>/
│   ├── features/<run_id>/
│   ├── labels/<run_id>/
│   ├── classifications/        (flat; one parquet per video)
│   ├── directionality/
│   └── frames/{extracted,annotated}/
├── models/
└── logs/
    └── measures/ROI_definitions.h5
```

**Legacy SimBA layout** is preserved for backward compatibility; existing `project_folder/`-style projects continue to work without modification.

Backend code that needs layout-agnostic path resolution should use the `mufasa.project_layout.project_paths_from_config(config_path)` helper, which returns a dict of layout-correct paths for either layout.

See [`docs/v1_project_layout.md`](docs/v1_project_layout.md) for the full v1 reference (directory contents, run-id semantics, the path-abstraction layer for backend code).

## Installation

```bash
git clone https://github.com/<your-fork>/mufasa.git
cd mufasa
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.11 or later. Optional dependencies: GPU-enabled PyTorch / TensorFlow / FFmpeg for video-heavy workflows.

After install, run `mufasa` to open the workbench. On first launch the smart launcher checks for missing native deps (OpenCV, FFmpeg, etc.) and prints what's missing if anything; the workbench still loads regardless.

## Documentation

Doc index: [`docs/README.md`](docs/README.md).

User-facing:

| Doc | When to read |
|---|---|
| [`docs/v1_project_layout.md`](docs/v1_project_layout.md) | Creating a v1 project, understanding the directory layout. |
| [`docs/migration_guide.md`](docs/migration_guide.md) | Moving an existing legacy project to v1. |
| [`docs/workflow_recipes.md`](docs/workflow_recipes.md) | Eleven end-to-end recipes for common experimental setups. |
| [`docs/data_source_guides.md`](docs/data_source_guides.md) | Per-tracker import guide (DLC, SLEAP, MARS, etc.). |

Developer-facing:

| Doc | When to read |
|---|---|
| [`docs/workflows.md`](docs/workflows.md) | Codebase audit: which workflow lives where, port status, known gaps. |
| [`docs/hardwired_paths_audit.md`](docs/hardwired_paths_audit.md) | Path-abstraction triage rules + the 107-hit audit. |
| [`docs/workflow_audit.md`](docs/workflow_audit.md) | Topological view of the system (pages, forms, backends, pipeline). |
| [`docs/simba_death_cascade.md`](docs/simba_death_cascade.md) | Tk surface removal plan (now complete). |
| [`docs/qt_workbench_known_issues.md`](docs/qt_workbench_known_issues.md) | QWI tracker. |

## Relationship to SimBA

Mufasa is a soft fork of SimBA — the scientific intent, behavioral-analysis pipeline, and core algorithms are identical. The differences are infrastructural:

* **GUI:** Qt (PySide6) instead of Tkinter. The Tk surface is entirely removed.
* **Project layout:** v1 layout with per-run separation under `derived/<stage>/<run_id>/`.
* **Platform:** Linux + Python 3.11+ only. Windows / macOS / older Pythons not supported.
* **No PyPI distribution.** Mufasa is install-from-source. The SimBA `simba-uw-tf-dev` PyPI package is unaffected and remains the canonical SimBA distribution.

If you're starting a new project and want the cross-platform GUI, PyPI distribution, and the broader SimBA tutorial ecosystem, use SimBA. If you're on Linux and want Qt + the v1 layout, use Mufasa.

## Citation

If you use Mufasa in published work, **cite the original SimBA paper** — Mufasa is a re-skin of SimBA's pipeline, not a separate scientific contribution.

```bibtex
@article{Goodwin2024,
  author    = {Goodwin, Nastacia L. and Choong, Jia J. and Hwang, Sophia and
               Pitts, Kayla and Bloom, Liana and Islam, Aasiya and
               Zhang, Yizhe Y. and Szelenyi, Eric R. and Tong, Xiaoyu and
               Newman, Emily L. and Miczek, Klaus and Wright, Hayden R. and
               McLaughlin, Ryan J. and Norville, Zane C. and Eshel, Neir and
               Heshmati, Mitra and Nilsson, Simon R. O. and Golden, Sam A.},
  title     = {Simple Behavioral Analysis (SimBA) as a platform for
               explainable machine learning in behavioral neuroscience},
  journal   = {Nature Neuroscience},
  volume    = {27},
  pages     = {1411--1424},
  year      = {2024},
  doi       = {10.1038/s41593-024-01649-9},
  publisher = {Nature Publishing Group},
  url       = {https://www.nature.com/articles/s41593-024-01649-9},
}
```

Optional acknowledgment for Mufasa-specific features (v1 layout, Qt workbench, etc.) is welcome but not required.

## License

GPL v3 — see [`LICENSE`](LICENSE).

Inherited from SimBA's GPL v3 license; this is a derivative work.

## Acknowledgments

Mufasa builds on the work of the SimBA project at the [Golden Lab, University of Washington](https://goldenneurolab.com/). Mufasa-specific changes are documented in the patch series (see `docs/simba_death_cascade.md` for the v1 migration history).

Original SimBA contributors:

* [Simon Nilsson](https://github.com/sronilsson)
* [Jia Jie Choong](https://github.com/inoejj)
* [Sophia Hwang](https://github.com/sophihwang26)

See [github.com/sgoldenlab/simba](https://github.com/sgoldenlab/simba) for the upstream project, the full contributor list, and the original SimBA documentation, tutorials, and scientific references.
