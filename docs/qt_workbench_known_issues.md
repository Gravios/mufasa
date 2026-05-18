# Qt workbench — known issues

**Status:** issues surfaced during real-world workbench use; tracked here so dedicated patches can fix them without losing context. Not blockers for the SimBA.py death cascade (the legacy entry-point retirement) — these are post-cascade follow-on work.

| ID | Severity | Status | Component |
|---|---|---|---|
| QWI-1 | High | Open | ROI: Apply-all fails on v1 projects |
| QWI-2 | Medium | Open | Features: destination label shows raw HTML markup |
| QWI-3 | Medium | Open | Features: max_workers=0 crash on empty project |
| QWI-4 | Medium | ✓ Fixed 122d0 | Workbench: page ordering (Annotation before Classifier) |

---

## QWI-1 — ROI: Apply-all fails on v1 projects

**Reproduce:** Open Qt workbench → ROI → Definitions → draw a ROI on one video → click "APPLY TO ALL" on the same row in the video table.

**Error:**
```
Apply-all failed
Could not apply ROIs: NotDirectoryError: NOT A DIRECTORY ERROR:
Could not find the videos directory in the Mufasa project.
SimBA expected a directory at location:
/data/testing/mufasa/test-<id>/videos
```

**Root cause:** `mufasa/roi_tools/roi_utils.py:462` in `multiply_ROIs()`:

```python
videos_dir = os.path.join(project_path, "videos")
```

Hard-codes the **legacy SimBA layout** (`<project>/videos/`). The v1 layout stores videos at `<project>/sources/videos/` (per `config_reader.py:367` v1 branch). The legacy assumption is baked into `multiply_ROIs()`, so any v1 project gets the wrong path even though `ConfigReader` would resolve it correctly.

The Qt UI (`roi_video_table.py` → `_apply_all`) calls `multiply_ROIs()` directly and inherits the bad assumption.

**Recommended fix:**

Replace the hard-coded `videos_dir` computation in `multiply_ROIs()` with layout-agnostic resolution. Either:

(a) Instantiate `ConfigReader(config_path=...)` and use `reader.video_dir` (which has the v1-vs-legacy branch built in).

(b) Use `project_metadata_from_config()` helper from `mufasa/utils/v1_meta.py` (or wherever the layout resolver lives).

Approach (a) is the smaller diff. Same root function still works for legacy users since `ConfigReader` resolves the legacy path correctly when it sees a legacy-shaped project.

The error message wording — "SimBA expected" — is also stale; should be "Mufasa expected" to match current branding. Cosmetic but worth doing in the same patch.

**Severity rationale:** ROI Apply-all is a primary workflow for any project with > 1 video. Blocks v1 users. High.

---

## QWI-2 — Features: destination label shows raw HTML markup

**Reproduce:** Qt workbench → Features → Compute feature subsets → look at the "Destination" radio buttons.

**Symptom:** The "Write per-family parquet to ..." radio button renders as raw text:

```
Write per-family parquet to <code>derived/features/<lt;familygt;/<lt;videogt;.parquet</code> <i>(recommended, v1-native)</i>
```

The `<code>`, `<i>`, and entity-encoded `&lt;` / `&gt;` are visible as literal characters instead of being rendered as styled text.

**Root cause:** `mufasa/ui_qt/forms/features.py:198-201` (and similar lines):

```python
QLabel(
    "<code>derived/features/&lt;family&gt;/&lt;video&gt;.parquet</code>"
    " <i>(recommended, v1-native)</i>", self,
)
```

The string contains HTML markup, but the widget is likely a `QRadioButton` (not a `QLabel`), and `QRadioButton.text` is plain-text-only by default. Or it's a `QLabel` whose parent layout-context overrides the text format.

Other affected lines in the same file: 95, 96, 188, 200, 201, 283, 294, 468.

**Recommended fix:**

Option A — Strip the HTML markup (simpler):
* Replace `<code>...</code>` and `<i>...</i>` with plain text.
* Use Unicode replacements where needed (`«family»`, `〈video〉`, etc.) or simply spell things out.

Option B — Force rich-text rendering:
* For each affected widget, call `setTextFormat(Qt.RichText)`.
* But this only works for `QLabel` — `QRadioButton.text` doesn't support rich text. Would need to either wrap in a `QLabel` next to the radio button, or use a different widget tree.

Option A is the pragmatic fix; Option B is structurally cleaner but more code-churn.

**Severity rationale:** Cosmetic but visible on a primary workflow page. Confuses users about what the actual path format is. Medium.

---

## QWI-3 — Features: `max_workers=0` crash on empty project

**Reproduce:** Qt workbench → Features → Compute feature subsets → Run (on a project with 0 eligible videos for feature extraction).

**Error:**
```
ValueError: max_workers must be greater than 0
```

Traceback ends at:
```
File ".../feature_extractors/feature_subsets.py", line 650, in _run_parallel
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
File "/usr/lib/python3.12/concurrent/futures/process.py", line 679, in __init__
    raise ValueError("max_workers must be greater than 0")
```

**Root cause:** `mufasa/feature_extractors/feature_subsets.py:643-650`:

```python
n_videos = len(self.data_paths)
n_workers = min(self.n_workers, n_videos)
print(f"Running feature extraction on {n_videos} videos across {n_workers} workers...")
failures: list[tuple[str, str]] = []
with ProcessPoolExecutor(max_workers=n_workers) as pool:
```

When `n_videos == 0`, `n_workers = min(self.n_workers, 0) = 0`. `ProcessPoolExecutor(max_workers=0)` raises immediately.

**Recommended fix:** short-circuit before pool creation:

```python
n_videos = len(self.data_paths)
if n_videos == 0:
    print("Feature extraction: no eligible videos. Nothing to do.")
    return
n_workers = max(1, min(self.n_workers, n_videos))
```

The `max(1, ...)` is belt-and-braces. The short-circuit catches the actual empty-project case cleanly.

The Qt form should also display a clearer message — "No eligible videos in this project. Add videos via Data Import first." — instead of letting the backend ValueError bubble up.

**Severity rationale:** Crashes the run on otherwise-valid empty-state. Easy fix but exposes the no-video-state which is common during fresh project setup. Medium.

---

## QWI-4 — Workbench: page ordering (Annotation before Classifier) — ✓ Fixed 122d0

**Symptom:** Sidebar shows Annotation tab above Classifier tab, but the Annotation workflow **requires** classifiers to be defined first:

```
Could not open labeller
No classifiers defined in the project. Add at least one via the
Classifier → Manage page before labelling.
```

User has to scroll PAST Annotation to find Classifier → Manage, then back up.

**Root cause:** `mufasa/ui_qt/workbench_app.py:71-75` calls `build_annotation_page` before `build_classifier_page`.

**Fix landed in 122d0:** swap the call order so Classifier appears above Annotation in the sidebar:

```python
# Was:
build_annotation_page(wb, config_path=...)  # line 71-72
build_classifier_page(wb, config_path=...)  # line 74-75
# Now:
build_classifier_page(wb, config_path=...)
build_annotation_page(wb, config_path=...)
```

**Severity rationale:** UX/discoverability issue. Doesn't block workflows but creates a backtrack. Medium.

The fix is a 2-line swap; the cost of NOT fixing it is every new user discovers the order mismatch on first annotation attempt.

---

## Discovery context

These issues surfaced during user testing of the Qt workbench on a real v1 project (`/data/testing/mufasa/test-20260427/`) while the Tk-deletion cascade work (122cz onwards) was in progress. They're orthogonal to the cascade — the cascade removes the legacy Tk path; these are bugs in the **already-shipped Qt workbench code**.

None of them block Stage B execution of the death cascade. Tracking them here so they don't get lost.

## Suggested patch sequence

| Patch | Bug | Scope | Estimated size |
|---|---|---|---|
| **122d0** (this patch) | QWI-4 | Swap page order in workbench_app.py | 2 lines |
| (later) | QWI-3 | Short-circuit + max() in feature_subsets._run_parallel | 5 lines + Qt-side message |
| (later) | QWI-2 | Strip HTML markup from QRadioButton labels in features.py | ~10 lines, ~5 widgets |
| (later) | QWI-1 | Layout-agnostic videos_dir in multiply_ROIs | ~5 lines + audit other multiply_ROIs siblings |

The cascade work (Stage A → B → C) can proceed in parallel with these fixes; they don't share files.
