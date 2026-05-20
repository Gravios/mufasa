# ROI tool — enhancements proposal

**Audience:** maintainers planning ROI tool improvements.

**Scope:** audit of current ROI tool capabilities; proposed design for two specific feature gaps surfaced by real-user testing.

**Date:** post-patch 122dj.

---

## Audit summary

| Capability | Status | Implementation |
|---|---|---|
| Draw rect / circle / polygon | works | `roi_canvas.py` — hand-rolled QPainter-based canvas with `_DrawMode` state machine (PENDING → DRAGGING → idle) |
| Apply ROI from one video to all others | **fixed in 122d9 + 122da** | `roi_utils.multiply_ROIs` v1-routed through `project_paths_from_config` |
| Reset a video's ROIs | works (v1 + legacy) | `roi_utils.reset_video_ROIs` (also 122da-fixed) |
| Save / persist ROIs | works | `<root>/logs/measures/ROI_definitions.h5` (path identical for both layouts) |
| **Apply to "similar" subset of videos** | **NOT IMPLEMENTED** | only all-or-nothing |
| **Select + drag-to-adjust placed ROI** | **NOT IMPLEMENTED** | dragging is creation-only; placed shapes are static |
| Standardize ROI size across videos | works | `roi_size_standardizer.py` (separate dialog) |
| Import ROIs from CSV | works | `import_roi_csv.py` |
| Duplicate ROIs source → target | works | `duplicate_rois_source_target.py` |

The two gaps directly surfaced by user testing are **subset-apply** (motivated by users with multiple experimental conditions / arena layouts needing different ROI shapes per group) and **drag-to-adjust** (motivated by the friction of reset-then-redraw for small ROI tweaks).

## The QPainter vs QGraphicsScene question

`roi_canvas.py` (28 KB) is a hand-rolled `QPainter`-based canvas. It does NOT use `QGraphicsScene` + `QGraphicsItem`. This means:

* `ItemIsMovable` / `ItemIsSelectable` flags aren't available — they belong to `QGraphicsItem`, not `QWidget`.
* Drag-to-adjust requires implementing hit-testing + a per-shape selection state machine manually within the existing `mousePressEvent` / `mouseMoveEvent` / paint loop.
* This is more code than the QGraphicsScene equivalent would be, but **a full rewrite to QGraphicsScene is not the right path** — `roi_canvas.py` has ~1000 lines of working draw-state-machine + frame-coordinate-mapping logic that would all need to be reimplemented. Adding 200 lines for select-and-drag within the existing structure is a much smaller risk envelope.

Both proposals below stay within the existing QPainter framework.

---

## Proposal 1 — Apply ROI to selected videos (suggested patch: 122dk)

### Problem

Users with two or more experimental conditions / arena layouts need different ROI shapes per group. Currently the only options are:
* "Apply to all" → propagates one shape to every video (wrong for the secondary condition).
* Per-video draw → tedious for projects with many videos per condition.

### Proposed UI

A new button on the ROI define panel, between "Apply to all" and "Reset":

```
[Reset]  [Apply to all]  [Apply to selected…]  [Save]  [Save & close]
```

Clicking "Apply to selected…" opens a modal **multi-select dialog**:

```
┌─ Apply ROI to selected videos ────────────────────────────┐
│                                                            │
│  Source video: cond_A_mouse1.mp4                          │
│  Source ROIs: 1 rectangle ("home_zone")                   │
│                                                            │
│  Filter by name:  [cond_A_____________]                    │
│  [Select all]  [Clear]  [Invert]                          │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ☐ cond_A_mouse1.mp4   (source — already has this ROI)│ │
│  │ ☑ cond_A_mouse2.mp4                                  │ │
│  │ ☑ cond_A_mouse3.mp4                                  │ │
│  │ ☐ cond_B_mouse1.mp4                                  │ │
│  │ ☐ cond_B_mouse2.mp4                                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  3 of 5 videos selected                                   │
│                                                            │
│                              [Cancel]  [Apply to 3 videos]│
└────────────────────────────────────────────────────────────┘
```

Key features:
* **Filter by name** — substring match, updates list incrementally as the user types. Helps when there are 50+ videos.
* **Select all / Clear / Invert** — convenience for common patterns.
* **Source video appears as disabled / pre-checked-and-disabled** — can't apply to itself; users see it for orientation only.
* **Live count** — "3 of 5 videos selected" → button label updates.

### Backend changes

`mufasa/roi_tools/roi_utils.py` gains a new function:

```python
def multiply_ROIs_to_subset(
    *,
    config_path: str,
    source_video: str,
    target_videos: list[str],
) -> None:
    """Copy ROIs from source_video to each target_video.

    Parameters
    ----------
    config_path : str
        Project config path (v1 .toml or legacy .ini).
    source_video : str
        Filename of the video whose ROIs to copy.
    target_videos : list[str]
        Filenames of videos to receive the copied ROIs. The
        source video is silently dropped from this list if
        present.
    """
```

Internally this can either (a) call `multiply_ROIs` once per target with the existing one-source-to-all helper restricted, or (b) load the HDF once, mutate in memory, and write back. Option (b) is cleaner (one disk write) but requires more code; (a) is one-line wrapper-style. Recommend (b) for performance with large projects.

Alternative API: extend `multiply_ROIs` to accept an optional `target_videos: list[str] | None` parameter, where `None` keeps the current "apply to all" behavior. Backward compatible. Probably the cleaner choice.

### Cost

* Backend: ~30 LoC (new function or parameter on `multiply_ROIs`).
* Dialog: ~150 LoC (new file `apply_roi_subset_dialog.py`).
* Wiring: ~20 LoC (new button + click handler in `roi_define_panel.py`).
* Tests: ~80 LoC (AST + behavioral mocks where possible).
* **Total: ~280 LoC.** One patch.

### Risk

Low. The backend is a small extension to existing tested code. The dialog is a standard Qt multi-select pattern. The risk is mostly in the per-row checkbox rendering when there are many videos — a `QListWidget` with item-level checkboxes handles 1000+ rows fine; no virtualization needed.

---

## Proposal 2 — Drag to adjust placed ROI (suggested patch: 122dl)

### Problem

Currently, once a shape is committed (mouse released after creation), it's static. Adjusting position or size requires:
1. Reset the entire video's ROIs.
2. Redraw from scratch.

This is friction-heavy for small adjustments (e.g., "move the food zone 20 pixels left").

### Proposed UX

Introduce a **select mode** on the canvas:

* New `_DrawMode.SELECT` value alongside the existing `IDLE / RECT_PENDING / RECT_DRAGGING / ...` set.
* Activated by an "Edit" or "Select" toggle button on the panel (sibling of Rect / Circle / Polygon).
* In SELECT mode:
  * Mouse click on a placed shape → that shape becomes the **selected** shape (visual: bold outline; resize handles at corners for rect / endpoint for circle / vertices for polygon).
  * Click + drag the body of the shape → the shape moves with the cursor.
  * Click + drag a corner handle (rect) / edge handle (circle) → the shape resizes.
  * Click + drag a vertex (polygon) → that vertex moves.
  * Click on empty space → deselection.
  * Press Delete key → selected shape removed.
* On mouse release after a drag, the shape's new coords are committed to the in-memory model (which the panel's Save button writes back to HDF).

### Implementation outline

Within the existing `roi_canvas.py` QPainter framework:

```python
# In roi_canvas.py:

class _DrawMode(Enum):
    IDLE = "idle"
    RECT_PENDING = "rect_pending"
    RECT_DRAGGING = "rect_dragging"
    CIRCLE_PENDING = "circle_pending"
    CIRCLE_DRAGGING = "circle_dragging"
    POLY_PENDING = "poly_pending"
    POLY_VERTEXING = "poly_vertexing"
    SELECT = "select"               # NEW
    SHAPE_MOVING = "shape_moving"   # NEW
    HANDLE_DRAGGING = "handle_dragging"  # NEW (resize)


class _EditState:
    """Tracks the current selection / drag state."""
    selected_shape_idx: int | None = None
    selected_handle: str | None = None  # "n", "se", "vertex_3", etc.
    drag_offset: tuple[float, float] | None = None
    pre_drag_coords: dict | None = None  # for undo


class ROICanvas(QWidget):
    # ... existing fields ...

    def _hit_test(self, frame_pt: tuple[float, float]) -> tuple[int | None, str | None]:
        """Return (shape_idx, handle) for the topmost shape under
        the cursor, or (None, None) if no hit.

        Iterates shapes back-to-front (matches z-order rendering).
        For each: check handles first (small click target); then
        the body.
        """
        # ~30 LoC
        ...

    def mousePressEvent(self, ev):
        # ... existing creation modes ...
        elif self._mode == _DrawMode.SELECT:
            shape_idx, handle = self._hit_test(frame_pt)
            if shape_idx is not None:
                self._edit_state.selected_shape_idx = shape_idx
                self._edit_state.selected_handle = handle
                self._edit_state.pre_drag_coords = (
                    self._shapes[shape_idx].copy()
                )
                if handle is None:
                    # Body click → move whole shape
                    self._edit_state.drag_offset = (
                        frame_pt[0] - self._shapes[shape_idx].center[0],
                        frame_pt[1] - self._shapes[shape_idx].center[1],
                    )
                    self._mode = _DrawMode.SHAPE_MOVING
                else:
                    self._mode = _DrawMode.HANDLE_DRAGGING
            else:
                # Click on empty space → deselect
                self._edit_state.selected_shape_idx = None
            self.update()

    def mouseMoveEvent(self, ev):
        # ... existing creation modes ...
        elif self._mode == _DrawMode.SHAPE_MOVING:
            shape = self._shapes[self._edit_state.selected_shape_idx]
            shape.set_center(
                frame_pt[0] - self._edit_state.drag_offset[0],
                frame_pt[1] - self._edit_state.drag_offset[1],
            )
            self.update()
        elif self._mode == _DrawMode.HANDLE_DRAGGING:
            shape = self._shapes[self._edit_state.selected_shape_idx]
            shape.set_handle(self._edit_state.selected_handle, frame_pt)
            self.update()

    def mouseReleaseEvent(self, ev):
        # Commit the change to the shape model (already done by
        # mouseMove); just transition back to SELECT idle.
        if self._mode in (_DrawMode.SHAPE_MOVING,
                          _DrawMode.HANDLE_DRAGGING):
            self._mode = _DrawMode.SELECT
            self._emit_change()  # notify panel — Save button enables

    def keyPressEvent(self, ev):
        if (ev.key() == Qt.Key_Delete
                and self._edit_state.selected_shape_idx is not None):
            del self._shapes[self._edit_state.selected_shape_idx]
            self._edit_state.selected_shape_idx = None
            self.update()
            self._emit_change()
```

The painter (already present) needs ~30 lines added to render selection chrome:

```python
def paintEvent(self, ev):
    # ... existing shape rendering ...
    for idx, shape in enumerate(self._shapes):
        # Existing rendering
        self._draw_shape(painter, shape)
        # NEW: selection chrome
        if idx == self._edit_state.selected_shape_idx:
            self._draw_selection_chrome(painter, shape)
            self._draw_handles(painter, shape)
```

### Cost

* `roi_canvas.py`: ~150 LoC (hit-test + new modes + key handling + render chrome).
* `roi_define_panel.py`: ~20 LoC (Edit / Select toggle button + wiring).
* Shape model: ~30 LoC (`set_center`, `set_handle` methods on Rect / Circle / Polygon classes — may already exist; check).
* Tests: ~80 LoC.
* **Total: ~280 LoC.** One patch.

### Risk

Medium. The hit-test logic is the only piece with real ambiguity — what counts as "on the handle" vs "on the body" at the edge needs care. Tolerance values matter (8-pixel handle hit-radius is typical). The QPainter framework is well-understood after working on `roi_canvas.py` for prior patches; the state machine extends naturally.

Polygon handle-dragging (moving individual vertices) is the trickiest sub-feature. Recommend deferring polygon vertex-drag to a follow-up if it adds too much per-vertex hit-test logic — the patch can ship with rect + circle drag-and-resize and polygon move-only (drag the body, no per-vertex adjust).

---

## Implementation order recommendation

**122dk first** (Apply to selected videos):
* Solves the immediate two-conditions pain.
* Smaller and lower-risk than drag-to-adjust.
* Adds clear user value with minimal canvas-state changes.

**122dl second** (drag-to-adjust):
* UX improvement; not blocking any specific workflow.
* Bigger change to canvas internals; better to land after subset-apply has been exercised on real projects so canvas-level confidence is high.

Both fit cleanly into the existing 122dx patch series rhythm.

---

## What this proposal does NOT cover

* **Rotated rectangles.** Mufasa's rectangles are axis-aligned. Adding rotation would be substantially more code (rotation handle + transform math) and isn't asked for. Skip.
* **Per-condition ROI templates.** A higher-level abstraction where users define "condition A layout" and "condition B layout" once, then assign each video to a condition. Powerful but a separate design conversation; subset-apply handles the immediate pain and the template idea can be a future generalization.
* **Undo / redo** within the canvas. The proposed drag-to-adjust commits changes immediately; reverting requires resetting and redrawing. A proper undo stack is a much larger feature (per-shape diff history, batching, etc.) and not in scope.
* **3D ROI support.** Out of scope; 3D pose data is itself carved out as a separate future concern.

---

## References

* Current canvas: [`mufasa/ui_qt/dialogs/roi_canvas.py`](../mufasa/ui_qt/dialogs/roi_canvas.py) — 28 KB; hand-rolled QPainter; `_DrawMode` state machine.
* Current panel: [`mufasa/ui_qt/dialogs/roi_define_panel.py`](../mufasa/ui_qt/dialogs/roi_define_panel.py) — 31 KB; Apply / Reset / Save buttons at L328.
* Per-row table: [`mufasa/ui_qt/dialogs/roi_video_table.py`](../mufasa/ui_qt/dialogs/roi_video_table.py) — 24 KB; per-video Draw / Reset / Apply-all buttons.
* Backend: [`mufasa/roi_tools/roi_utils.py`](../mufasa/roi_tools/roi_utils.py) — `multiply_ROIs`, `reset_video_ROIs`.
* Recent fixes: patches 122d9 + 122da (Apply-all v1 routing).
