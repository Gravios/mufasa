# Plan: Native Qt drawing canvas for the ROI definition panel

**Status**: deferred — see context below
**Estimated scope**: 800–1200 lines, 2–3 iteration cycles
**Replaces**: the OpenCV-based popup currently driven by
`mufasa.video_processors.roi_selector{,_circle,_polygon}` and isolated
in a subprocess by `_run_selector_sync` in
`mufasa/ui_qt/dialogs/roi_define_panel.py`

---

## Why this is deferred

The OpenCV popup works. It opens in a subprocess (cleanly avoids the
host Qt context), displays the actual video frame, accepts mouse input,
and returns captured geometry as JSON. ~200 ms spawn latency per draw.
User flagged the popup as aesthetically annoying but not blocking.
Decision was to ship the subprocess fix and revisit only if the popup
becomes a real friction point during sustained ROI work.

If you're picking this up later: **don't do it unless you have a
concrete UX complaint to anchor the work**. Reinventing OpenCV's
drawing canvas in Qt is the kind of project that's easy to start,
hard to finish, and where the result is easily worse than what's
being replaced. SimBA, DeepLabCut, EthoVision all use OpenCV
popups for this exact reason.

---

## What stays the same

**Don't touch**:

- `mufasa/roi_tools/roi_logic.py` — the `ROILogic` class that owns
  ROI state, persistence, frame buffer. It's UI-framework-independent.
  The new Qt drawing widget calls `logic.add_rectangle()` /
  `add_circle()` / `add_polygon()` exactly as the existing code does.
- `mufasa/video_processors/roi_selector*.py` — keep the OpenCV
  classes. They're load-bearing for non-Qt callers (notebooks,
  scripts) and removing them is unrelated to this work.
- `mufasa/ui_qt/dialogs/roi_define_panel.py`'s overall layout
  (tool palette / video list / preview / shape table / save bar).
  Only the **preview area** and the **draw flow** change.
- The H5 file format and downstream analyzers. Nothing here touches
  persistence.

**Replace**:

- `_run_selector_sync` and `_run_selector_in_process` in
  `roi_define_panel.py`. After this work they're dead code; remove.
- `_PreviewLabel` (the read-only `QLabel` showing the current frame).
  Becomes an interactive `QGraphicsView` or custom `QWidget`.
- `_on_draw_clicked` flow. No more subprocess spawn — clicking
  Draw → enters drawing mode on the existing preview widget.

---

## Architecture

### New file: `mufasa/ui_qt/dialogs/roi_canvas.py` (~500 lines)

A `QWidget` (or `QGraphicsView`) that:

1. Renders the current frame with all defined ROI overlays.
2. Accepts mouse events for drawing new shapes.
3. Manages per-shape-type state machines (idle / drawing / committing).
4. Translates between widget pixels and image pixels for shapes
   that need to land on the original image's coordinate system.
5. Emits a `shape_committed` signal carrying the captured geometry
   when the user finishes drawing.

**Choose between two implementations**:

#### Option A: Custom `QWidget` with `paintEvent` (recommended)

Simpler. The widget paints the frame as a `QPixmap` via `QPainter`,
then paints the existing ROIs and the in-progress shape on top.
Mouse events come in as widget coordinates; convert to image
coordinates via the cached scaling factor.

Pros: ~500 lines total. Total control over every pixel. No surprises
from `QGraphicsScene` performance.

Cons: have to manually invalidate (`update()`) on every mouse move,
which can cause flicker on slow connections. Mitigate by only
calling `update()` from the rect that changed (use the
`update(QRect)` overload).

#### Option B: `QGraphicsScene` + `QGraphicsView` + `QGraphicsItem`

More code (~700 lines). Each ROI becomes a `QGraphicsItem`.
Drawing creates new items; the scene auto-renders them.

Pros: items can be selected, dragged, deleted by the framework.
Useful if you want to add post-draw editing later.

Cons: layered abstractions. Coordinate transforms are split between
the view's transform and the scene's. Overkill for the actual
requirements (draw, save, done — no editing).

**Pick A.** Editing existing ROIs by dragging them isn't in scope and
shouldn't be — that's `interactive_modifier_ui.py`'s job.

### Mouse coordinate plumbing

The preview displays the frame scaled to fit the widget while
preserving aspect ratio. The widget is, say, 1024×768; the frame is
600×338. `QPixmap.scaled(self.size(), Qt.KeepAspectRatio,
Qt.SmoothTransformation)` produces a, say, 1024×577 scaled pixmap
centered with black bars top/bottom.

For mouse coordinate translation, cache on every paint:

```python
@dataclass
class _ImageMapping:
    scale: float            # frame_px → widget_px
    offset_x: int           # widget x where frame_px x=0 lands
    offset_y: int           # widget y where frame_px y=0 lands
    frame_w: int            # original frame width in px
    frame_h: int            # original frame height in px

def _widget_to_frame(self, wx: int, wy: int) -> Optional[Tuple[int, int]]:
    m = self._mapping
    if m is None: return None
    fx = (wx - m.offset_x) / m.scale
    fy = (wy - m.offset_y) / m.scale
    if 0 <= fx < m.frame_w and 0 <= fy < m.frame_h:
        return int(fx), int(fy)
    return None  # mouse is in the black bars; ignore
```

Use `_widget_to_frame` in every `mousePressEvent` /
`mouseMoveEvent` / `mouseReleaseEvent`. If it returns `None`,
treat as "off-canvas" — clamp or ignore depending on context.

### State machines

Use a small enum for active mode:

```python
class _DrawMode(Enum):
    IDLE = "idle"               # not drawing; preview is interactive
    RECT_DRAGGING = "rect"      # mouse down, capturing rectangle
    CIRCLE_DRAGGING = "circle"  # mouse down, capturing radius
    POLY_VERTEXING = "poly"     # accumulating polygon vertices
```

Activated when user clicks Draw → in the panel. Each mode has its
own mouse-event handlers and key handlers. Key handlers:

- `IDLE`: no-op
- `RECT_DRAGGING`, `CIRCLE_DRAGGING`: ESC commits if size > 0,
  cancels otherwise
- `POLY_VERTEXING`: ESC / Q / Space close polygon if ≥3 vertices,
  cancel otherwise; Backspace removes last vertex

Always reset to `IDLE` after commit/cancel.

### Drawing flow (replaces subprocess)

In `ROIDefinePanel`:

```python
def _on_draw_clicked(self):
    # All the existing validation: name not empty, no duplicates, etc.
    name = ...
    kind = self._selected_kind()
    color = ...

    # Hand off to canvas. Canvas takes over mouse/keyboard until done.
    self.canvas.start_draw(kind=kind, name=name, color=color,
                           thickness=..., marker=...)
    # Return immediately. Canvas emits shape_committed when user
    # finishes; we wire that signal in __init__.

def _on_shape_committed(self, kind: str, name: str, geom: dict, ...):
    # geom is already in frame-pixel coordinates
    if kind == RECTANGLE:
        self.logic.add_rectangle(name=name, top_left=geom["top_left"],
                                  bottom_right=geom["bottom_right"], ...)
    elif kind == CIRCLE: ...
    elif kind == POLYGON: ...
    self._dirty = True
    self._sync_table()
    # Canvas re-renders itself by querying logic.rendered_frame()
    self.canvas.refresh()
```

The canvas has no knowledge of `ROILogic`. It just emits geometry.
The panel translates that into logic mutations. Clean separation.

### Polygon vertex simplification

The OpenCV version uses Shapely:

```python
self.polygon = Polygon(vertices).simplify(tolerance=20,
                                            preserve_topology=True)
```

Two options:

1. **Skip it.** User clicks vertices; what they click is what they
   get. Simpler. If users complain about jitter, add it later.
2. **Reuse Shapely.** It's already a Mufasa dependency
   (`shapely.geometry.Polygon`). Three lines:

```python
from shapely.geometry import Polygon
poly = Polygon(vertices).simplify(tolerance=20, preserve_topology=True)
simplified = list(poly.exterior.coords)[:-1]   # drop closing dup
```

**Do option 2.** Three lines, behavior matches existing. Done.

---

## Implementation steps

### Step 1: Build the canvas widget in isolation (200 lines)

Single file `roi_canvas.py`. Widget that:

- Holds a BGR ndarray and renders it via `paintEvent`
- Has `set_frame(bgr_array)` to update the displayed frame
- Has `_mapping` cached after every paint
- Has `_widget_to_frame()` helper

Test by instantiating in a tiny PySide6 demo script. Frame should
display, resize cleanly, mouse events should fire (just `print()`
the converted coords for now — no shape drawing yet).

### Step 2: Add rectangle drawing (100 lines)

- `start_draw(kind="rectangle", ...)` sets mode to
  `RECT_DRAGGING_PENDING`
- `mousePressEvent` while pending → mode becomes `RECT_DRAGGING`,
  records start point
- `mouseMoveEvent` while dragging → updates end point, calls `update()`
- `paintEvent` while dragging → draws live rectangle preview
- `mouseReleaseEvent` while dragging → commits, emits signal,
  resets to `IDLE`
- ESC at any point → cancels, resets to `IDLE`

Test by drawing a few rectangles in the demo script. Verify the
emitted coords are in frame-pixel space (e.g. drawing in the middle
of a 600×338 frame should give coords in the 100–500 range, not
the 200–800 range of the displayed widget).

### Step 3: Add circle drawing (50 lines)

Same pattern as rectangle. `mousePressEvent` records center; drag
extends radius; release commits.

### Step 4: Add polygon drawing (100 lines)

Different state machine — accumulates vertices on click, closes on
ESC/Q/Space.

- `mousePressEvent` while pending → mode becomes `POLY_VERTEXING`,
  records first vertex
- `mousePressEvent` while vertexing → appends vertex
- `mouseMoveEvent` while vertexing → updates "rubber band line"
  showing where the next click will go
- `keyPressEvent` for ESC/Q/Space while vertexing → if ≥3 vertices,
  apply Shapely simplification, commit, emit signal; else cancel
- `keyPressEvent` Backspace → pop last vertex
- `paintEvent` → draws all committed vertices, edges between them,
  rubber-band line from last vertex to current mouse pos

This is the trickiest of the three. Test edge cases:

- 2 vertices, hit ESC → cancel (not enough for a polygon)
- 3 vertices in a line (degenerate polygon) → Shapely will collapse
  to a 0-area polygon; `preserve_topology=True` should handle, but
  test
- Backspace at 0 vertices → no-op, don't crash
- Backspace at 1 vertex → return to mode-pending state

### Step 5: Integrate into `ROIDefinePanel` (100 lines)

- Replace `_PreviewLabel` with the new `ROICanvas`
- Wire `start_draw` and `shape_committed` signal
- Delete `_run_selector_sync` and `_run_selector_in_process`
- Remove `subprocess`, `pickle`, `tempfile` imports
- The existing validation in `_on_draw_clicked` (name checks, etc.)
  stays but is followed by `canvas.start_draw(...)` instead of the
  subprocess call
- The existing `_on_selector_done` becomes `_on_shape_committed`
  with minor tweaks

### Step 6: Test the integration (manual)

- Draw rectangle, circle, polygon — all three from the same panel
  session
- Switch videos via PgUp/PgDn while in IDLE mode → works as before
- Try to switch videos while drawing (mode != IDLE) → cancel the
  draw first, or block the switch
- Resize the panel window mid-draw → coords should still translate
  correctly to frame pixels (this is the most likely subtle bug)
- Save → ROIs should land in `ROI_definitions.h5` exactly as before
- Close + reopen panel on the same video → ROIs should reappear
  (logic doesn't change; this is just verifying we didn't break
  the load path)

### Step 7: Smoke tests

`tests/smoke_roi_canvas.py` — ~200 lines:

- Coordinate transform with various mappings (no PySide6 needed,
  just test `_widget_to_frame` math)
- Mock mouse-event sequences and verify the state machine
  transitions (still no PySide6 — the state machine logic should
  be testable in isolation if we structure it right)
- Polygon simplification edge cases (degenerate inputs, exact
  outputs)

The widget itself can't be headlessly tested without `pytest-qt` or
similar; visual testing on `nphy-069` is the actual verification.

---

## Failure modes to watch for

These have bitten me on similar Qt work in this project. Listing
them so they're cheap to catch:

1. **Widget pixel coords vs. image pixel coords**. If you draw a
   rectangle in the displayed widget, the coords need to be the
   image pixels, not widget pixels. Off-by-one bugs here are
   visible in the saved H5 file (rectangles slightly larger or
   smaller than what the user drew).

2. **Aspect-preserving scaling and black bars**. The widget is
   1024×768, frame is 600×338, scaled pixmap is 1024×577 with
   95.5px black bars top and bottom. If user clicks in the black
   bar, `_widget_to_frame` should return None and the click should
   be ignored. Easy to forget.

3. **`update()` performance**. On a long, slow-rendering frame
   (4K, ThinLinc), `update()` on every mouse move is a flicker
   bomb. Use `update(QRect)` with the changed region only.

4. **Focus stealing**. The `Draw →` button has focus when clicked.
   If the canvas doesn't grab focus, ESC/Q won't be received.
   Call `canvas.setFocus()` after `start_draw()`.

5. **Resize during draw**. If the panel resizes mid-draw, the
   `_mapping` cache becomes stale. Recompute it in `paintEvent` or
   listen to `resizeEvent` and clear in-progress geometry that's
   in widget coords (everything should be stored in frame coords;
   widget-coord state should only live within a single mouse
   sequence).

6. **High-DPI displays**. Qt's `devicePixelRatio` may be 2x on
   HiDPI. Mouse events come in logical coords, not physical.
   `QPixmap` for the frame may need to account for this.
   Test on a HiDPI monitor.

7. **Polygon "rubber band" repaint**. Drawing a line from the
   last committed vertex to the current mouse position requires
   `mouseMoveEvent` to call `update()`. Don't forget — without
   it the rubber band only redraws on click.

8. **Modifier keys**. Polygon close on Space conflicts with the
   normal use of Space in Qt forms. Make sure the canvas has
   keyboard focus when in `POLY_VERTEXING` mode and steals Space
   only then.

---

## Done criteria

The work is finished when:

1. User clicks Draw → in the unified panel; **no popup window
   appears**.
2. User clicks/drags in the panel's preview area to define a
   shape; the shape draws live as they drag.
3. ESC commits rectangle/circle; ESC/Q/Space commits polygon.
4. Resulting `ROI_definitions.h5` is byte-identical to what the
   OpenCV version would have produced for the same input
   (verified by comparing `read_roi_data` output).
5. All existing tests still pass (`smoke_roi_logic.py` — 20/20).
6. New `smoke_roi_canvas.py` exercises the coordinate transform
   and state machine.
7. The 30-patch stack still applies cleanly through this work
   (or this becomes a single new patch on top of the existing
   stack).

---

## What this enables (and what it doesn't)

**Enabled** by native Qt:

- Single-window UX (the original ask)
- Faster draw flow (~200 ms saved per shape — not significant
  unless drawing dozens of ROIs in rapid succession)
- Foundation for in-place ROI editing later (`QGraphicsItem`-style
  selection, drag-to-resize, etc.) if Option B was chosen.

**Not enabled**:

- Anything in `interactive_modifier_ui.py`. That's a separate
  module that handles existing-ROI editing (move, resize, rename
  after-the-fact). It's already its own thing; this work doesn't
  touch it.
- Faster-than-OpenCV drawing performance. They'll be similar.
- Any new ROI shape types. Triangle, ellipse, free-form
  brush — none of those are in scope.

---

## Cost summary

| Item | Estimate |
|---|---|
| Lines of code | 800–1200 (final patch size) |
| Files touched | 3 (`roi_canvas.py` new; `roi_define_panel.py` modified;
  test new) |
| Time for me to write the patch | ~2–3 hours of focused work |
| Iteration cycles after first ship | 2–3 (history says this is
  the floor for UI patches in this project) |
| Risk of making things worse | Real — the OpenCV popup works.
  A buggy native Qt canvas is worse than a working subprocess. |
| Risk of stalling out | Real — half-finished GUI rewrites are
  the canonical failed-project shape. |

If you come back to this and decide to do it, **commit to finishing**.
Don't ship a partial implementation. The OpenCV version stays
load-bearing until the Qt version handles all three shape types,
all the edge cases, and all the existing tests pass.
