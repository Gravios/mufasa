# Tk surface audit

**Generated:** post-patch 122bq (May 2026).
**Purpose:** Document the state of Mufasa's legacy Tk surface (`mufasa/SimBA.py` + `mufasa/ui/`) and identify what's required to remove it, since the Qt workbench (`mufasa` / `mufasa-workbench` entry points) is the recommended UI going forward.
**Status of this doc:** This audit is a **planning artifact**, not a removal decision. The Tk surface is currently live and reachable via the `mufasa-tk` entry point. Removal requires the migrations described in §5.

---

## 1. Two parallel UI surfaces

Mufasa ships with two independent UIs that share backend code:

| Surface | Entry point | Code location | State |
|---|---|---|---|
| **Tk (legacy)** | `mufasa-tk = "mufasa.SimBA:main"` | `mufasa/SimBA.py`, `mufasa/ui/` | maintained, but slated for removal once Qt port is feature-complete |
| **Qt (current)** | `mufasa = "mufasa.cli.workbench_launcher:main"`, `mufasa-workbench = "mufasa.ui_qt.workbench_app:main"`, `mufasa-chooser = "mufasa.ui_qt.app:main"` | `mufasa/ui_qt/` | active development; 14 workbench pages |

Both surfaces drive the same backend (`mufasa/data_processors/`, `mufasa/plotting/`, `mufasa/feature_extractors/`, etc.). The choice of surface only affects how the user invokes a backend — not what runs.

---

## 2. Tk surface inventory

96 non-`__init__.py` files in `mufasa/ui/`, plus `mufasa/SimBA.py` (the launcher). Classified by importer audit:

| Status | Count | Meaning |
|---|---:|---|
| **LOAD-BEARING-FOR-QT** | 1 | Qt code imports from this Tk file. Must be ported or replaced before removal. |
| **TK-REACHABLE** | 85 | Reachable via `mufasa/SimBA.py` (Tk launcher) or via backend modules that launch Tk popups. |
| **TK-INTERNAL-ONLY** | 8 | Imported only by other `mufasa/ui/` files. Dies with its parents. |
| **UNREFERENCED** | 2 | Not imported by anything in the codebase. Safe to delete now (modulo dynamic-import / script-execution caveats). |

### 2a. LOAD-BEARING-FOR-QT (1 file)

| File | Imported by | Notes |
|---|---|---|
| `mufasa/ui/px_to_mm_ui.py` | `mufasa/ui_qt/forms/video_utilities.py` | Qt calibration form launches the Tk pixel-to-mm dialog. Single Qt-side coupling to the Tk surface. |

**Implication:** until this is ported, deleting `mufasa/ui/` breaks the Qt workbench's video-calibration flow.

### 2e. Pop-ups orphan re-audit (added 122co)

A targeted re-audit of `mufasa/ui/pop_ups/` (81 files post-122ck, after the 5 cue-light deletions) for true orphans — files where no symbol they define is imported anywhere in the codebase.

**Finding: zero true orphans.** Every single pop-up file is referenced by at least one importer somewhere in `mufasa/`. The bulk of those importers is `SimBA.py` (the Tk launcher), which imports ~80 pop-up classes for its menu-callback wiring.

**Implication for Tier-4 strategy:** `mufasa/ui/pop_ups/` cannot be incrementally drained the same way `cue_light_tools/` was. Files in pop_ups/ get deleted in only two ways:

1. **Cluster-deletion** — when a group of related pop-ups all have Qt replacements and `SimBA.py` is edited to drop the cluster (the 122ck cue-light cleanup pattern: 5 files dropped, 3 SimBA.py breadcrumb-comments left).
2. **Cascade-deletion** — when `SimBA.py` itself is deleted in Tier-4 close-out, the ~80 transitively-imported files become orphans in a single move.

There is no "delete now, no consumers" pop-up file. The "many likely orphans" hypothesis floated in earlier planning was wrong.

#### Methodology note: AST traversal vs regex matching

The first pass of this audit used regex (`\bimport\b.*\b{class_name}\b`) and reported 37 of 81 files as orphans — a 45% false-positive rate. The regex pattern failed on **multi-line imports**, which `SimBA.py` uses extensively:

```python
# Continuation-line style (used in SimBA.py for ~80 pop-up imports)
from mufasa.ui.pop_ups.animal_directing_other_animals_pop_up import \
    AnimalDirectingAnimalPopUp
```

```python
# Parenthesized style (used in some places for grouped imports)
from mufasa.ui.pop_ups.foo_pop_up import (
    FooPopUp,
    BarPopUp,
)
```

In both cases, the class name lives on a different line from the `import` keyword. A line-bound regex with `re.compile(r"\bimport\b.*\bClass\b")` (no `DOTALL` flag) misses both forms.

**Correct approach:** AST traversal. `ast.parse(src)` normalizes both single-line and multi-line imports into the same `ImportFrom` node shape; `node.names` gives the imported names regardless of source formatting. Every future cross-file import audit in this repo should use AST, not regex.

The corrected AST-based audit re-ran in under a second and reported 0 orphans — a 1-line code change (regex → AST loop) that completely flipped the conclusion.

### 2f. Companion audits (added 122cp)

Two further AST orphan-audits run after §2e succeeded for `pop_ups/`. Both produce zero orphans but reveal different consumer-graph shapes worth documenting before future Tier-4 work.

#### 2f.1. `mufasa/ui/` non-popup files (8 files)

| File | Importer cluster | Notes |
|---|---|---|
| `blob_quick_check_interface.py` | UI (`ui/blob_tracker_ui.py`) | Tk-internal chain |
| `blob_tracker_ui.py` | UI_POPUP (`ui/pop_ups/initialize_blob_tracking_pop_up.py`) | Tk-internal chain |
| `get_tree_view.py` | UI_POPUP (`ui/pop_ups/print_video_meta_popup.py`) | Tk-internal chain |
| `machine_model_settings_ui.py` | SIMBA | Reached only via SimBA.py menus |
| `px_to_mm_ui.py` | **QT + UI_POPUP** | LOAD-BEARING-FOR-QT (§2a); the single Qt → Tk coupling point |
| `tkinter_functions.py` | EVERYTHING (520 import edges) | The central Tk module; the dependency target of every Tk-importer audit |
| `utils.py` | BACKEND + SIMBA | Consumed by `roi_tools/roi_ui.py` + `roi_ui_mixin.py` (§3d Bucket 3 deferral) and SimBA.py |
| `video_timelaps.py` | UI_POPUP (`ui/pop_ups/video_processing_pop_up.py`) | Tk-internal chain |

**0 orphans.** The picture matches §2a's count of 1 LOAD-BEARING-FOR-QT file — `px_to_mm_ui.py` confirmed. The remaining 7 files split between SimBA-reachable (`machine_model_settings_ui.py`), backend-reachable via Qt-ROI dialog chain (`utils.py`), the central Tk module itself (`tkinter_functions.py`), and internal Tk-chains (4 files).

Implication: same as §2e for `pop_ups/`. None of these 8 files is independently deletable. Each goes when its parent work item completes (SimBA.py deletion + Qt ROI dialog port + Tier-4 finale).

#### 2f.2. `mufasa/unsupervised/pop_ups/` (13 files)

**0 orphans, but with a notable property: the entire 13-file cluster is self-contained.** Each file's only importer is `mufasa/unsupervised/unsupervised_main.py`. No SimBA.py reference, no Qt-side coupling, no backend module reaches in.

This makes the unsupervised cluster the cleanest possible cascade-deletion target. When Tier 3b ships the Qt port and `unsupervised_main.py` is replaced (or deleted), all 13 files become orphans in a single move — no SimBA.py surgical edits needed (unlike the 122ck cue-light cleanup or the future ui/pop_ups bulk delete). The closed-cluster shape is what `backend_audit.md` §3d Bucket 2 was implicitly describing for the "dies with Tier 3b" line.

### 2g. ROI Tk cluster re-audit (added 122cq)

A targeted re-audit of `roi_tools/roi_ui_mixin.py` (the §3d Bucket 3 deferral) found that the original 122ck audit was wrong. The four Qt-side "ROI_ui" references that motivated the deferral are **all docstrings**, not real code dependencies:

| Qt file | Line | Reference |
|---|---:|---|
| `ui_qt/dialogs/roi_video_table.py` | 11 | docstring: ":class:`ROI_ui` canvas in a subprocess" |
| `ui_qt/dialogs/roi_video_table.py` | 40 | docstring: "Subprocess-launched ``ROI_ui`` writes…" |
| `ui_qt/dialogs/roi_video_table.py` | 407 | docstring: "Replaces the previous subprocess-launched ROI_ui" |
| `ui_qt/forms/roi.py` | 37 | docstring: "is still the OpenCV-based ``ROI_ui``" |

All four are historical pointers explaining what each Qt port replaces, not actual `from … import ROI_ui` statements. An AST audit of real `ImportFrom` nodes for the `ROI_ui` symbol returns just two consumers, both Tk:

```
ui/blob_tracker_ui.py:13        from mufasa.roi_tools.roi_ui import ROI_ui
ui/pop_ups/roi_video_table_pop_up.py:8   from mufasa.roi_tools.roi_ui import ROI_ui
```

Meanwhile, the actual Qt ROI surface (`ui_qt/dialogs/roi_canvas.py` + `roi_define_panel.py`) imports from `mufasa.roi_tools.roi_logic` directly. `roi_logic.py` is the UI-framework-independent extraction (671 lines, no Tk/Qt imports) explicitly designed so Qt and Tk panels can both build on the same primitives. The Qt panels never touched `ROI_mixin` at all.

`roi_ui_mixin.py` + `roi_ui.py` are reclassified from `backend_audit.md` §3d Bucket 3 → Bucket 2 in 122cq. Bucket 3 is now drained (originally had this one entry).

**Update (122cr): cluster deleted.** Following the reclassification, 122cr executed the cluster-deletion. Six files removed:

* `mufasa/roi_tools/roi_ui_mixin.py` (1263 lines)
* `mufasa/roi_tools/roi_ui.py` (178 lines)
* `mufasa/ui/blob_tracker_ui.py` (Tk blob tracker UI)
* `mufasa/ui/blob_quick_check_interface.py` (orphan-after-cascade — consumed only by `blob_tracker_ui.py`)
* `mufasa/ui/pop_ups/roi_video_table_pop_up.py` (Tk ROI table popup)
* `mufasa/ui/pop_ups/initialize_blob_tracking_pop_up.py` (Tk blob tracker launcher)

Five surgical `SimBA.py` edits:
* Lines 162–163: import of `InitializeBlobTrackerPopUp` → breadcrumb-comment.
* Line 193: import of `ROIVideoTable` → breadcrumb-comment.
* Line 400: `start_new_ROI = SimbaButton(...)` → breadcrumb-comment.
* Line 404: `start_new_ROI.grid()` → breadcrumb-comment.
* Line 869: `blob_tracking_menu.add_command(...)` (the "Perform blob tracking" entry) → breadcrumb-comment. The sibling `BlobVisualizerPopUp` entry below stays.

Qt replacements verified before deletion:
* `ROIVideoTable` → `ROIManageForm` (in `mufasa/ui_qt/forms/roi.py`).
* `InitializeBlobTrackerPopUp` → `BlobTrackerInitLauncher` (in `mufasa/ui_qt/forms/addons.py`, wired into `addons_page.py:55`).

Same pattern as the 122ck cue-light cleanup (6 files + SimBA.py edits) but with the audit-corrected dependency picture from 122cq.

**Caveat: subprocess-launched popups (122cr discovery; 122cs/ct partial port).** After the 122cr cluster-deletion, the AST orphan-audit reported 4 files in `mufasa/ui/pop_ups/` as newly-orphan. 2 remain post-122ct:

* `duplicate_rois_by_source_target_popup.py`
* `import_roi_csv_popup.py`
* ~~`min_max_draw_size_popup.py`~~ ✓ **DELETED 122ct** (Qt-native port at `ui_qt/dialogs/min_max_draw_size.py`)
* ~~`roi_size_standardizer_popup.py`~~ ✓ **DELETED 122cs** (Qt-native port at `ui_qt/dialogs/roi_size_standardizer.py`)

These are NOT real orphans. They're kept alive by the Qt dialog `ui_qt/dialogs/roi_video_table.py:491-513`, which has four file-menu actions (`_action_standardize`, `_action_duplicate`, `_action_import_csv`, `_action_min_max_draw_size`) that launch the popups via subprocess using string-literal imports:

```python
def _action_standardize(self) -> None:
    self._launch_tk_popup(
        "from mufasa.ui.pop_ups.roi_size_standardizer_popup import ROISizeStandardizerPopUp\n"
        "ROISizeStandardizerPopUp(config_path=sys.argv[1])\n"
    )
```

The import is a string literal passed to a subprocess Python interpreter — it's a real runtime dependency, but `ast.parse(roi_video_table.py)` reports it as just a `Constant` string, not an `ImportFrom`. The 4 popups appear orphan to the AST audit but aren't. Deleting them would break the Qt dialog at runtime.

This is the fourth audit-methodology lesson (alongside 122co's AST > regex + walk-functions-too + 122cq's docstring-vs-import distinction): **string-literal subprocess imports are real runtime dependencies that no AST audit can catch**. Code like `subprocess.run(["python", "-c", "from foo import bar; bar()"])` is invisible to import analyses. The honest check before deletion is: does the file appear in *any* string anywhere in the codebase?

When the four file-menu actions get ported to Qt-native implementations (out of scope per the comment at `roi_video_table.py:488-489`), the subprocess strings will disappear and the 4 popups can finally be deleted alongside.

#### Methodology lesson: distinguish docstring references from code imports

This is the third audit-methodology lesson in the recent run:

1. 122co: AST > regex (multi-line imports break line-bound regex).
2. 122co (second pass): walk both `ast.ClassDef` AND `ast.FunctionDef` (some pop-ups use the function-style idiom).
3. 122cq (this section): only `ast.ImportFrom` nodes count as real dependencies. Sphinx-style `:class:`…\`` references in docstrings look like code references at a glance but aren't.

The audit anti-pattern is "grep for the symbol name." Even with AST, walking `ast.walk(tree)` and matching the symbol name in arbitrary `ast.Name` nodes would catch the same false positives — variable references, attribute access, type annotations in strings, comments parsed via something other than the `tokenize` module. The honest test is: does an `ast.ImportFrom` node have an `alias.name` matching the target? If not, it's not a real consumer.

The 122ck audit looked at "grep results" rather than parsed imports. The correction here is to pin this lesson alongside the AST > regex lesson in §7.

### 2b. UNREFERENCED (2 files)

| File | Lines | Defines | Notes |
|---|---:|---|---|
| `mufasa/ui/pop_ups/helpers.py` | 13 | `restart_roi_video_table` | Small helper, no importers detected. Verify no dynamic import before deleting. |
| `mufasa/ui/user_defined_pose_creator.py` | 156 | `PoseConfigCreator` | Standalone pose-config creator. Verify no string-based reference in a launcher table before deleting. |

### 2c. TK-INTERNAL-ONLY (8 files)

These are imported only by other `mufasa/ui/` files (their callers are themselves Tk-reachable). They die transitively when their parents are removed.

```
blob_quick_check_interface.py
blob_tracker_ui.py
get_tree_view.py
pop_ups/duplicate_rois_by_source_target_popup.py
pop_ups/import_roi_csv_popup.py
pop_ups/min_max_draw_size_popup.py
pop_ups/roi_size_standardizer_popup.py
video_timelaps.py
```

### 2d. TK-REACHABLE (85 files)

Mostly `mufasa/ui/pop_ups/*.py`. All importable from `mufasa/SimBA.py`'s menu commands. A small subset is also imported by backend modules (see §3b).

The full list is omitted here for length; reproduce with:

```bash
python tools/tk_surface_audit.py --status TK-REACHABLE  # if/when scripted
```

For now, the per-file status is in `/tmp/tk_audit.txt` (regeneratable from the AST audit pass at the bottom of this doc).

---

## 3. Cross-surface coupling

### 3a. `mufasa/ui/tkinter_functions.py` — the foundation

| Importer kind | Count | Files |
|---|---:|---|
| Qt code | **0** | (none) |
| `mufasa/SimBA.py` | 1 | the Tk launcher |
| Other Tk files | 85 | every Tk popup |
| **Backend modules** | **25** | mixins, labelling, video_processors, roi_tools, cue_light_tools, etc. |

The Qt code does NOT depend on `tkinter_functions.py` — the Qt widgets (in `mufasa/ui_qt/widgets.py`) are independent ports.

**But 25 backend modules do import from it.** Removing `tkinter_functions.py` would break those backends unless they're migrated first.

A sample of the backend importers:

```
mufasa/mixins/annotator_mixin.py
mufasa/mixins/train_model_mixin.py
mufasa/mixins/pop_up_mixin.py
mufasa/cue_light_tools/cue_light_main_popup.py
mufasa/labelling/labelling_interface.py
mufasa/labelling/standard_labeller.py
mufasa/video_processors/video_processing.py
mufasa/video_processors/batch_process_menus.py
mufasa/roi_tools/roi_ruler.py
mufasa/roi_tools/roi_ui_mixin.py
...15 more
```

These backend modules use `MufasaDropDown`, `MufasaLabel`, `SimbaButton`, etc. as part of their popups. They need to be either:

- Migrated to use Qt widgets (`mufasa.ui_qt.widgets`), or
- Refactored to separate "backend logic" from "UI invocation" so the UI part can live next to the Qt code.

### 3b. Backend modules launching Tk pop-ups

A subset of TK-REACHABLE files is imported by backend modules:

| File | Backend importer |
|---|---|
| `pop_ups/cue_light_clf_analyzer_popup.py` | `cue_light_tools/cue_light_main_popup.py` |
| `pop_ups/cue_light_data_analyzer_popup.py` | `cue_light_tools/cue_light_main_popup.py` |
| `pop_ups/cue_light_movement_analyzer_popup.py` | `cue_light_tools/cue_light_main_popup.py` |
| `pop_ups/cue_light_visualizer_popup.py` | `cue_light_tools/cue_light_main_popup.py` |

`cue_light_main_popup.py` itself is in `cue_light_tools/`, not in `ui/` — it's backend code that imports Tk popups. The Qt workbench provides a separate Cue-light form (under Add-ons), so this Tk-side cue-light path is parallel functionality.

---

## 4. Removal-dependency graph

Order matters. Each step unlocks the next.

```
                    ┌─────────────────────────────────┐
                    │ Step 1 — Port px_to_mm_ui.py    │
                    │   to Qt; update                  │
                    │   ui_qt/forms/video_utilities.py │
                    └────────────────┬─────────────────┘
                                     │ now 0 Qt → Tk
                                     v
       ┌─────────────────────────────────────────────────────┐
       │ Step 2 — Migrate the 25 backend modules that         │
       │   import from mufasa.ui.tkinter_functions.            │
       │   Either: (a) move their Tk popup invocations         │
       │   into the Qt workbench, or (b) refactor to            │
       │   separate backend logic from UI invocation.          │
       └───────────────────────────┬─────────────────────────┘
                                   │
                                   v
       ┌─────────────────────────────────────────────────────┐
       │ Step 3 — Migrate cue_light_main_popup.py +           │
       │   the 4 cue-light Tk popups it imports.               │
       │   The Qt workbench's Add-ons page already has         │
       │   cue-light forms; reconcile.                          │
       └───────────────────────────┬─────────────────────────┘
                                   │
                                   v
       ┌─────────────────────────────────────────────────────┐
       │ Step 4 — Remove the mufasa-tk entry point from        │
       │   pyproject.toml. Users now reach mufasa only         │
       │   through the Qt workbench (mufasa /                  │
       │   mufasa-workbench / mufasa-chooser).                  │
       └───────────────────────────┬─────────────────────────┘
                                   │
                                   v
       ┌─────────────────────────────────────────────────────┐
       │ Step 5 — Delete mufasa/SimBA.py and the entire        │
       │   mufasa/ui/ tree (96 files). Drop the                │
       │   backward-compat aliases in errors.py /              │
       │   tkinter_functions.py / etc.                          │
       └─────────────────────────────────────────────────────┘
```

Each step is a multi-file patch. Steps 2 and 3 are the substantive work; Step 1 is small, Step 4 is one-line, Step 5 is mechanical bulk deletion.

---

## 5. Implications for ongoing patches

### What this means for the SimBA → Mufasa rebranding lane

**Don't rename Tk-only code.** Renaming code about to be deleted is wasted effort, and back-compat aliases for vanishing code become permanent dead weight.

Patches 122bn (UI widget class renames) and 122bo (exception class renames) **already happened** and shipped backward-compat aliases. They're sunk cost. Don't revert; just stop adding to the pile.

Patches that should NOT be written:
- Renaming `SimbaButton`/`SimbaCheckbox`/`SimBALabel`/`SimBARadioButton` (the UI factory functions). Tk-only. 702 refs, all in code slated for removal.
- Updating Tk launcher UI labels in `mufasa/SimBA.py` ("SIMBA CONFIG FILE: " → "PROJECT CONFIG FILE: ", `self.root.title("SimBA")` → `"Mufasa"`, etc.). The file is going away.

Patches that ARE still worthwhile:
- Backend module cleanup that helps Step 2 (separating backend logic from Tk invocation).
- Anything in `mufasa/ui_qt/`.
- Anything in `mufasa/data_processors/`, `mufasa/feature_extractors/`, `mufasa/plotting/`, etc.

### What this means for documentation

`workflow_recipes.md` describes recipes that may invoke Tk popups (e.g., R8 unsupervised discovery). That recipe currently says:

> Use `UnsupervisedGUI` (Tk popup) or call backends programmatically

After Tk removal, that recipe will need updating to: "call backends programmatically" (until a Qt unsupervised page is added).

---

## 6. Recommended next steps

In priority order, low-risk first:

1. **Add deprecation banners** to `mufasa/SimBA.py`, `mufasa/ui/__init__.py`, and `mufasa/ui/tkinter_functions.py`. Module-level docstring + `DeprecationWarning` on import. Signals to contributors not to touch these.

2. **Port `px_to_mm_ui.py` to Qt**. Smallest substantive migration. Removes the only Qt → Tk coupling.

3. **Audit the 25 backend importers of `tkinter_functions.py`**. For each, decide: migrate to Qt widgets, or split off the Tk-popup invocation into a separate file that lives next to the Tk surface (so the backend module itself becomes UI-free).

4. **Delete `mufasa/ui/pop_ups/helpers.py` and `mufasa/ui/user_defined_pose_creator.py`** (the two UNREFERENCED files). Trivial; just verify there's no dynamic import via string lookup first.

5. After Steps 2–3 are done, plan Steps 4–5 of §4 — the actual removal.

---

## 7. Audit methodology

The audit is reproducible. For each `.py` file in `mufasa/`:

```python
import ast
def get_from_imports(f):
    tree = ast.parse(f.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
    return imports
```

For each file in `mufasa/ui/`, find every file `f` such that `mod_of(target) in get_from_imports(f)`. Bucket the importers by their location (Qt / SimBA.py / backend / ui-internal). Status follows from which buckets have entries.

Pure AST-based; no runtime/dynamic-import detection. The two UNREFERENCED files should be cross-checked against string-based dynamic loads before deletion.

**Always use AST, never regex** (lesson learned in §2e's pop-ups orphan re-audit). A line-bound regex misses multi-line imports — both continuation-line (`import \`) and parenthesized (`import (\n …\n)`) forms. `SimBA.py` uses both extensively; a regex audit reported 37 false-positive orphans before the AST rerun corrected the count to 0.

**Only `ast.ImportFrom` nodes count as real dependencies** (lesson learned in §2g's ROI Tk cluster re-audit). Sphinx-style `:class:`…\`` references in docstrings look like code references when grep'd but aren't actual `from … import …` statements. The 122ck audit treated a docstring `:class:\`ROI_ui\`` reference as evidence of Qt → Tk dependency; the 122cq correction showed this was wrong. The honest check: walk every file's `ast.ImportFrom` nodes and ask whether any `alias.name` matches the target symbol. Anything else (string match, comment, docstring, type-annotation-in-string) is noise.

**String-literal subprocess imports are runtime dependencies no AST audit can catch** (lesson learned in §2g's 122cr deletion). Patterns like `subprocess.run(["python", "-c", "from foo import bar; bar()"])` look like just string `Constant` nodes to AST, but the imported file IS a real runtime consumer. The 122cr ROI Tk cluster-deletion almost-deleted 4 popup files that appeared orphan to AST but were actually kept alive by string-literal imports in `ui_qt/dialogs/roi_video_table.py`. **Before any cluster-deletion, grep the codebase for the target filename in ANY string context, not just imports.** The blind spot is unfixable in pure AST; the mitigation is awareness + a string-search pre-flight check.

---

## 8. Caveats

- **Dynamic imports not detected.** `importlib.import_module("mufasa.ui.X")` calls or string-based dispatch tables would be invisible to this audit. The two UNREFERENCED files in particular need a string-grep cross-check.
- **`__init__.py` re-exports not traced.** If `mufasa/ui/__init__.py` re-exports something from a sub-module, an importer of `mufasa.ui` would appear to reach the file indirectly. The audit treats each file independently. Spot-checked: `mufasa/ui/__init__.py` is empty / minimal.
- **Some "TK-REACHABLE via backend" files may have parallel Qt implementations.** The `cue_light_*` popups are an example — they're imported by `cue_light_main_popup.py` (backend) but the Qt workbench has its own Cue-light forms. The Tk path is parallel functionality, not unique.
- **The 85 TK-REACHABLE count is `>= 85`, not `= 85`.** Some files counted here are also reachable via the backend modules — they appear under both categories in the raw audit. Bucketing is by primary reachability, not exclusive.
- **The audit is post-122bn / post-122bo.** Both patches renamed Tk-side identifiers and added back-compat aliases. Those aliases are now permanent debt for code that's being deleted. (Sunk cost; not actionable here, just noted.)
- **`mufasa-chooser` entry point is intermediate.** It's defined as `mufasa.ui_qt.app:main` (a Qt chooser), but it launches Qt pop-ups, not Tk ones — so it doesn't keep the Tk surface alive. It will outlive the Tk removal.
