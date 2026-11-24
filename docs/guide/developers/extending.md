# D2. Extending Mufasa

Recipes for the common additions. Each one ends with a smoke test
([D3](testing-and-patches.md)).

## A workbench form

A form is an `OperationForm` subclass in `mufasa/ui_qt/forms/`, placed in a page's section:

```python
# mufasa/ui_qt/forms/my_step.py
from PySide6.QtWidgets import QFormLayout, QSpinBox

from mufasa.project_layout import ProjectPaths, project_paths_from_config
from mufasa.ui_qt.workbench import OperationForm


class MyStepForm(OperationForm):
    title = "My step"
    description = "What it does, in one sentence (shown under the title)."
    section_id = "my_step"            # optional: provenance (✓ / ○); add it to SECTIONS too

    def build(self) -> None:
        f = QFormLayout()
        self.window = QSpinBox(self)
        self.window.setRange(1, 1000)
        self.window.setValue(15)
        f.addRow("Window (frames)", self.window)
        self.body_layout.addLayout(f)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")        # shown to the user, nothing runs
        return {"config_path": self.config_path, "window": self.window.value()}

    def target(self, *, config_path: str, window: int) -> None:
        root = ProjectPaths(project_paths_from_config(config_path)["project_root"])
        out = root.stage_run_dir("my_step")                   # derived/my_step/<run id>/
        ...                                                    # the work: runs in the background
        self._last_run_id = out.name                          # lets provenance record the run
```

Then put it on a page:

```python
# mufasa/ui_qt/pages/features_page.py
from mufasa.ui_qt.forms.my_step import MyStepForm
...
page.add_section("My step", [(MyStepForm, {})])
```

Keep these in mind:

* **The work goes in `target`, not in the form.** Put the computation in a Qt-free module
  (`mufasa/data_processors/…`, `mufasa/pose3d/…`) and call it from `target`. Then the command
  line, scripts and tests can use it too.
* **Validate in `collect_args`.** Raise `RuntimeError` with a message the user can act on; the
  base class shows it and runs nothing.
* **`target` runs in a worker thread.** It must not touch widgets. Read every value in
  `collect_args`.
* **Provenance.** Set `section_id` and add a `SectionSpec` to `mufasa.section_provenance.SECTIONS`
  (its page, title, dependencies, `detect_path`), then set `self._last_run_id` in `target`. A
  producer whose output later stages read from `outlier_corrected` also sets
  `publish_source_stage` and `publish_target_stage` (as Kalman v2 does).
* **Paths come from `project_layout`.** Never build them by hand, and never name the old layout's
  config file outside `legacy_layout.py`. A test checks this.

A section that isn't a form (a launcher, an editor) is added with
`page.add_section_widget(title, factory)`. The factory is called when the section is first opened,
with `config_path=` if it takes it, and returns the widget.

## A page

```python
# mufasa/ui_qt/pages/my_page.py
from mufasa.ui_qt.workbench import WorkflowPage


def build_my_page(workbench, config_path: str | None = None) -> WorkflowPage:
    page = workbench.add_page("My page", icon_name="tools")
    page.add_section("My step", [(MyStepForm, {})])
    return page
```

Call `build_my_page(wb, config_path=project_config_path)` from `mufasa/ui_qt/workbench_app.py`, at
its place in the sidebar's order, which follows the pipeline.

## A feature family

Feature families live in `mufasa/feature_extractors/`:

1. **Name it.** Add a constant for its name (upper case, with units: `"MY FEATURE (MM)"`) and add
   it to `FEATURE_FAMILIES` in `feature_subsets.py`. The name becomes the output folder's slug,
   `derived/features/<slug>/`.
2. **Write the kernel.** Add a pure function to `feature_subset_kernels.py`. It takes the pose
   DataFrame and parameters, and returns a `dict[str, np.ndarray]` mapping each output column's
   name to one value per frame. Keep it free of project state, so it can be tested on synthetic
   pose.
3. **Dispatch it.** Add a branch to `process_one_video` in `feature_subset_orchestration.py`
   (`elif family == MY_FEATURE: fam_result = compute_my_feature(...)`). If the family's values
   can be negative, add it to `SIGNED_FAMILIES`. A missing value is then kept as NaN rather than
   the legacy −1.
4. **Show it.** **Features › Compute Features** lists `FEATURE_FAMILIES`. A family needing ROIs (its
   name containing `ROI`) goes in the ROI group automatically.

The 3-D families (`features_3d.py`) are the model for a family that reads another file (the 3-D
pose) instead of the 2-D pose.

## A command-line command

`mufasa <command>` commands are in `mufasa/cli/commands.py`:

```python
def cmd_hello(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mufasa hello", description="Say hello to a project.")
    ap.add_argument("-p", "--project")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg, how = resolve_project(args.project)          # --project, the current folder, the last opened
    if not _need(cfg, how):
        return 2
    print(f"Hello, {_root(cfg)}")
    return 0

COMMANDS["hello"] = (cmd_hello, "say hello to the project")
```

Then add `"hello"` to the `COMMANDS` tuple in `mufasa/cli/workbench_launcher.py`. The launcher
keeps its own copy so that it imports nothing until a command is asked for, and a test checks the
two lists agree. `mufasa help` lists the new command automatically.

A new `mufasa list` subject is a function `(args, cfg, how) -> int` added to `LIST_SUBJECTS`.
Support `--json` by building a list of dicts and calling `_emit(args, items, render)`.

Keep commands Qt-free: a test imports `mufasa.cli.commands` and checks PySide6 isn't loaded.

## A pipeline step

`mufasa run <step>` steps are `mufasa-pose3d` subcommands under pipeline names:

1. Add the subcommand to `mufasa/tools/pose3d_cli.py` (an argparse subparser and a handler). Its
   work goes in a Qt-free module (`mufasa/pose3d/…` or `mufasa/biomech/…`).
2. Add it to `STEPS` in `mufasa/cli/run.py`: `"my-step": ("my-subcommand", "what it does")`.
3. For it to be part of `mufasa run all`, add it to `CHAIN` and teach `run_all` its file names
   (`<stem>_my.parquet`) and inputs, so it can tell when it is up to date.

## A tracker format

Importers are in `mufasa/pose_importers/`. The workbench's list of formats is `POSE_IMPORT_ROUTES` in
`mufasa/ui_qt/forms/pose_import.py`. An importer writes one file per video into `sources/pose/`,
with columns `<marker>_x`, `<marker>_y` and `<marker>_p` (the likelihood). Then everything
downstream works unchanged. FreeDLC parquet files also carry the skeleton in their metadata. See
`smoke_122kz_pose_file_contract.py` for the contract a pose file must meet.
