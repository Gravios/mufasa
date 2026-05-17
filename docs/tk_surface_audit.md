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

---

## 8. Caveats

- **Dynamic imports not detected.** `importlib.import_module("mufasa.ui.X")` calls or string-based dispatch tables would be invisible to this audit. The two UNREFERENCED files in particular need a string-grep cross-check.
- **`__init__.py` re-exports not traced.** If `mufasa/ui/__init__.py` re-exports something from a sub-module, an importer of `mufasa.ui` would appear to reach the file indirectly. The audit treats each file independently. Spot-checked: `mufasa/ui/__init__.py` is empty / minimal.
- **Some "TK-REACHABLE via backend" files may have parallel Qt implementations.** The `cue_light_*` popups are an example — they're imported by `cue_light_main_popup.py` (backend) but the Qt workbench has its own Cue-light forms. The Tk path is parallel functionality, not unique.
- **The 85 TK-REACHABLE count is `>= 85`, not `= 85`.** Some files counted here are also reachable via the backend modules — they appear under both categories in the raw audit. Bucketing is by primary reachability, not exclusive.
- **The audit is post-122bn / post-122bo.** Both patches renamed Tk-side identifiers and added back-compat aliases. Those aliases are now permanent debt for code that's being deleted. (Sunk cost; not actionable here, just noted.)
- **`mufasa-chooser` entry point is intermediate.** It's defined as `mufasa.ui_qt.app:main` (a Qt chooser), but it launches Qt pop-ups, not Tk ones — so it doesn't keep the Tk surface alive. It will outlive the Tk removal.
