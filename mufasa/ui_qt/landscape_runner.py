"""
mufasa.ui_qt.landscape_runner
=============================

Feature Visualization → **Landscape…** / **Animate…** (patch 122jk).

Both run the command-line tools on the features selected in the tree:

* **Landscape** — ``mufasa-landscape --sources features --columns-file …``:
  joint PDFs of the selected pairs (``jpdf_top.png``) and a t-SNE of the
  selected features (``tsne.png``), pooled over every session.
* **Animate** — the landscape above (or an earlier run on the same
  selection, reused), then ``mufasa-landscape-animate --through-tsne`` with
  explicit pairs, so the GIF goes t-SNE → pair 1 → t-SNE → pair 2 → … →
  t-SNE.

Before either runs, :class:`FeatureNamesDialog` shows the selected
features in an editable table — column name | name in figures — pre-filled
from, and saved back to, the project's ``logs/measures/feature_names.toml``
(:mod:`mufasa.utils.feature_names`), so the names are remembered. Patch
122jl adds a Scale column (linear / log10) per feature: log10 features are
analysed and plotted as log10 by both tools, also remembered.

The tools run as child processes (:class:`QProcess`), so the GUI stays
responsive and their log streams into :class:`LandscapeRunDialog`, which
then shows the figures and the animation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QMovie, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

LANDSCAPE_MODULE = "mufasa.tools.feature_landscape"
ANIMATE_MODULE = "mufasa.tools.landscape_animate"
SELECTION_FILE = "columns.txt"


# --------------------------------------------------------------------------- #
# Runs and pairs
# --------------------------------------------------------------------------- #
def landscape_root(config_path) -> Path:
    return Path(config_path).parent / "derived" / "plots" / "feature_landscape"


def _lines(p: Path) -> list[str]:
    return sorted(ln.strip() for ln in p.read_text().splitlines() if ln.strip()) \
        if p.is_file() else []


def find_reusable_run(config_path, cols: list[str]) -> Path | None:
    """Newest landscape run made from exactly this selection (any order),
    with the same log10 scales, that has what the animation needs; None if
    there is none."""
    from mufasa.utils.feature_names import load_scales
    root = landscape_root(config_path)
    if not root.is_dir():
        return None
    want = sorted(cols)
    sc = load_scales(config_path)
    want_log = sorted(c for c in cols if sc.get(c) == "log10")
    for run in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        sel = run / SELECTION_FILE
        if not sel.is_file():
            continue
        got = sorted(ln.strip() for ln in sel.read_text().splitlines() if ln.strip())
        logged = run / "log10_columns.txt"
        if want_log and not logged.is_file():
            continue
        if got == want and _lines(logged) == want_log and all(
                (run / f).is_file() for f in
                ("tsne.parquet", "tsne_samples.parquet", "pairs.csv")):
            return run
    return None


def animation_pairs(cols: list[str], pairs_csv: Path | None, n: int,
                    max_per_feature: int = 2) -> list[tuple[str, str]]:
    """Pairs to animate. Two features → that pair. More → the best-ranked
    pairs of the run's ``pairs.csv`` (mufasa-landscape's ordering), each
    feature in at most ``max_per_feature`` of them; without a table, the
    selection taken two at a time."""
    if len(cols) == 2:
        return [(cols[0], cols[1])]
    out: list[tuple[str, str]] = []
    if pairs_csv is not None and Path(pairs_csv).is_file():
        table = pd.read_csv(pairs_csv)
        use = {c: 0 for c in cols}
        for a, b in zip(table.feature_a, table.feature_b):
            if a not in use or b not in use:
                continue
            if use[a] >= max_per_feature or use[b] >= max_per_feature:
                continue
            out.append((a, b))
            use[a] += 1
            use[b] += 1
            if len(out) >= n:
                break
    if not out:
        out = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1, 2)][:n]
    return out


def landscape_argv(config_path, columns_file: Path, out: Path, names_file: Path | None,
                   *, tsne_sample: int, frames_per_session: int, n_plot: int) -> list[str]:
    argv = ["-m", LANDSCAPE_MODULE, "--project", str(config_path),
            "--sources", "features", "--columns-file", str(columns_file),
            "--out", str(out), "--tsne-sample", str(tsne_sample),
            "--frames-per-session", str(frames_per_session), "--n-plot", str(n_plot)]
    if names_file is not None:
        argv += ["--names-file", str(names_file)]
    return argv


def animate_argv(run: Path, pairs: list[tuple[str, str]], out: Path,
                 names_file: Path | None, *, theme: str, mp4: bool, color: str,
                 n_points: int = 4000, sampling: str = "random",
                 intro_frames: int = 40, hold_frames: int | None = None,
                 tsne_hold_frames: int | None = None,
                 transition_frames: int | None = None, outliers: str = "none",
                 outlier_value: float | None = None,
                 outliers_tsne: bool = False) -> list[str]:
    argv = ["-m", ANIMATE_MODULE, str(run), "--through-tsne", "--theme", theme,
            "--color", color, "--out", str(out), "--n-points", str(n_points),
            "--sampling", sampling, "--intro-frames", str(intro_frames)]
    for flag, v in (("--hold-frames", hold_frames), ("--tsne-hold-frames", tsne_hold_frames),
                    ("--transition-frames", transition_frames)):
        if v is not None:
            argv += [flag, str(v)]
    if outliers != "none":   # patch 122jp
        argv += ["--outliers", outliers]
        if outlier_value is not None:
            argv += ["--outlier-percentile" if outliers == "percentile" else "--outlier-mad",
                     f"{outlier_value:g}"]
        if outliers_tsne:
            argv.append("--outliers-tsne")
    for a, b in pairs:
        argv += ["--pair", a, b]
    if names_file is not None:
        argv += ["--names-file", str(names_file)]
    if mp4:
        argv.append("--mp4")
    return argv


# --------------------------------------------------------------------------- #
# Pair matrix (patch 122jr)
# --------------------------------------------------------------------------- #
class PairMatrix(QWidget):
    """Feature × feature grid of tick boxes: the cell in row *a*, column *b*
    animates the joint space with *a* on x and *b* on y (the mirror cell
    swaps the axes). The diagonal is unused. Ticked cells show their place
    in the animation, which follows the order they were ticked in."""

    changed = Signal()

    def __init__(self, cols: list[str], labels: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.cols = list(cols)
        self._order: list[tuple[str, str]] = []
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Tick the joint spaces to animate — rows: feature on the x axis, "
                      "columns: feature on the y axis. Numbers give the order (the order you tick them); "
                      "the animation goes t-SNE → 1 → t-SNE → 2 → …", self)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        n = len(self.cols)
        self.grid = QTableWidget(n, n, self)
        self.grid.setSelectionMode(QAbstractItemView.NoSelection)
        self.grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for r in range(n):
            for c in range(n):
                it = QTableWidgetItem("")
                it.setTextAlignment(Qt.AlignCenter)
                if r == c:
                    it.setFlags(Qt.NoItemFlags)
                    it.setBackground(self.palette().mid())
                else:
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                self.grid.setItem(r, c, it)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.grid.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.set_labels(labels)
        self.grid.itemChanged.connect(self._on_item)
        lay.addWidget(self.grid, 1)
        brow = QHBoxLayout()
        self.best_btn = QPushButton("Best-ranked", self)
        brow.addWidget(self.best_btn)
        upper = QPushButton("All pairs (upper triangle)", self)
        upper.clicked.connect(lambda: self.set_pairs(
            [(self.cols[i], self.cols[j]) for i in range(n) for j in range(i + 1, n)]))
        brow.addWidget(upper)
        clear = QPushButton("Clear", self)
        clear.clicked.connect(lambda: self.set_pairs([]))
        brow.addWidget(clear)
        brow.addStretch()
        self.count = QLabel("", self)
        brow.addWidget(self.count)
        lay.addLayout(brow)
        self._refresh()

    def set_labels(self, labels: list[str]) -> None:
        """Header the rows and columns with the names used in the figures
        (as typed in the names table). Column headers wrap onto several
        lines so long names do not widen the grid; the column name is in
        the tooltip."""
        self.labels = list(labels)
        self.grid.setHorizontalHeaderLabels([_wrap(lb, 14) for lb in labels])
        self.grid.setVerticalHeaderLabels(list(labels))
        for i, (c, lb) in enumerate(zip(self.cols, labels)):
            tip = f"{lb}\n({c})"
            self.grid.horizontalHeaderItem(i).setToolTip(tip + " — y axis")
            self.grid.verticalHeaderItem(i).setToolTip(tip + " — x axis")

    def _cell(self, a: str, b: str) -> QTableWidgetItem:
        return self.grid.item(self.cols.index(a), self.cols.index(b))

    def _on_item(self, it: QTableWidgetItem) -> None:
        if self._busy or it.row() == it.column():
            return
        p = (self.cols[it.row()], self.cols[it.column()])
        if it.checkState() == Qt.Checked and p not in self._order:
            self._order.append(p)
        elif it.checkState() != Qt.Checked and p in self._order:
            self._order.remove(p)
        self._refresh()

    def set_pairs(self, pairs) -> None:
        """Tick exactly these pairs, in this order."""
        self._busy = True
        self._order = []
        for p in pairs:
            p = tuple(p)
            if p[0] in self.cols and p[1] in self.cols and p[0] != p[1] and p not in self._order:
                self._order.append(p)
        for r in range(len(self.cols)):
            for c in range(len(self.cols)):
                if r != c:
                    self.grid.item(r, c).setCheckState(
                        Qt.Checked if (self.cols[r], self.cols[c]) in self._order
                        else Qt.Unchecked)
        self._busy = False
        self._refresh()

    def pairs(self) -> list[tuple[str, str]]:
        return list(self._order)

    def _refresh(self) -> None:
        self._busy, was = True, self._busy
        for r in range(len(self.cols)):
            for c in range(len(self.cols)):
                if r != c:
                    self.grid.item(r, c).setText("")
        for k, (a, b) in enumerate(self._order, 1):
            self._cell(a, b).setText(str(k))
        self._busy = was
        n = len(self._order)
        self.count.setText(f"{n} joint space(s) · {1 + 2 * n if n else 0} spaces with t-SNE")
        self.changed.emit()


class PairPreview(QWidget):
    """Joint PDF (2-D log density) of one feature pair, for the pair
    matrix (patch 122jt)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("", self)
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(260, 260)
        lay.addWidget(self.canvas, 1)
        self.pair: tuple[str, str] | None = None

    def show_message(self, text: str) -> None:
        self.pair = None
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                         color="#777777")
        self.title.setText("")
        self.canvas.draw_idle()

    def draw(self, x: np.ndarray, y: np.ndarray, name_x: str, name_y: str, k: int) -> None:
        from mufasa.tools.feature_plots import plot_joint
        self.pair = (name_x, name_y)
        self.figure.clear()
        plot_joint(self.figure, x, y, name_x, name_y, style="heatmap", bins=70)
        n = int((np.isfinite(x) & np.isfinite(y)).sum())
        self.title.setText(f"<b>Pair {k}</b> (last ticked): {name_x} × {name_y} · "
                           f"{n:,} frames, all videos")
        self.canvas.draw_idle()


# --------------------------------------------------------------------------- #
# Names (and options) before a run
# --------------------------------------------------------------------------- #
class FeatureNamesDialog(QDialog):
    """Editable table of the selected features: column | name in figures.

    Pre-filled from the project's saved names; on OK the edits are saved
    (a name equal to the default, or empty, is forgotten). ``mode`` adds the
    options of that run ("landscape" or "animate")."""

    def __init__(self, parent: QWidget | None, config_path, cols: list[str],
                 mode: str = "landscape") -> None:
        super().__init__(parent)
        from mufasa.utils.feature_names import (
            SCALES,
            default_name,
            display,
            load_names,
            load_scales,
            scale_of,
        )
        self.config_path, self.cols, self.mode = config_path, list(cols), mode
        self.setWindowTitle("Feature names for the figures")
        self.resize(820, 520)
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Names used for these features in the figures, and the scale each is "
            "analysed and plotted on (log10: values ≤ 0 become missing). Tick Jitter "
            "for near-discrete features: their points are spread by a little noise "
            "in the figures only. All of it is remembered for this project "
            "(logs/measures/feature_names.toml).", self)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        saved = load_names(config_path) if config_path else {}
        scales = load_scales(config_path) if config_path else {}
        from mufasa.utils.feature_names import load_jitter
        jittered = load_jitter(config_path) if config_path else set()
        self.table = QTableWidget(len(self.cols), 4, self)
        self.table.setHorizontalHeaderLabels(["Feature (column)", "Name in figures",
                                              "Scale", "Jitter"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.scale_cbs: list[QComboBox] = []
        for r, c in enumerate(self.cols):
            src = QTableWidgetItem(default_name(c))
            src.setFlags(src.flags() & ~Qt.ItemIsEditable)
            src.setToolTip(c)
            src.setData(Qt.UserRole, c)
            self.table.setItem(r, 0, src)
            name = QTableWidgetItem(display(c, saved))
            name.setToolTip("Double-click or start typing to rename")
            self.table.setItem(r, 1, name)
            sc = QComboBox(self.table)
            sc.addItems(list(SCALES))
            sc.setCurrentText(scale_of(c, scales))
            sc.setToolTip("linear, or log10 (values ≤ 0 become missing)")
            self.table.setCellWidget(r, 2, sc)
            self.scale_cbs.append(sc)
            jt = QTableWidgetItem("")
            jt.setFlags((jt.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            jt.setCheckState(Qt.Checked if c in jittered else Qt.Unchecked)
            jt.setToolTip("Spread near-discrete values by uniform noise of ±0.4 of the "
                          "step between them — figures only; the analysis uses the "
                          "real values")
            self.table.setItem(r, 3, jt)
        names_tab = QWidget(self)
        nl = QVBoxLayout(names_tab)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(self.table, 1)

        brow = QHBoxLayout()
        reset = QPushButton("Reset selected to column names", self)
        reset.clicked.connect(self._reset_selected)
        brow.addWidget(reset)
        brow.addStretch()
        brow.addWidget(QLabel("Set all scales:", self))
        for s in ("linear", "log10"):
            b = QPushButton(s, self)
            b.clicked.connect(lambda _c=False, s=s: [cb.setCurrentText(s)
                                                     for cb in self.scale_cbs])
            brow.addWidget(b)
        nl.addLayout(brow)
        self.matrix: PairMatrix | None = None
        if mode == "animate":
            # Patch 122jr: pick the animated pairs in a feature × feature matrix
            from PySide6.QtWidgets import QTabWidget
            self.tabs = QTabWidget(self)
            self.tabs.addTab(names_tab, "Names, scales, jitter")
            self.matrix = PairMatrix(self.cols, [self.table.item(r, 1).text()
                                                 for r in range(len(self.cols))], self)
            # Patch 122jt: JPDF of the last ticked pair beside the matrix
            self.preview = PairPreview(self)
            split = QSplitter(Qt.Horizontal, self)
            split.addWidget(self.matrix)
            split.addWidget(self.preview)
            split.setStretchFactor(0, 1)
            split.setStretchFactor(1, 1)
            self.tabs.addTab(split, "Pairs to animate")
            self.table.itemChanged.connect(self._names_edited)
            self.matrix.changed.connect(self._pairs_changed)
            from PySide6.QtCore import QTimer
            self._pv_timer = QTimer(self)
            self._pv_timer.setSingleShot(True)
            self._pv_timer.setInterval(150)
            self._pv_timer.timeout.connect(self.update_preview)
            self._frame: pd.DataFrame | None = None
            self.tabs.currentChanged.connect(lambda _i: self._pv_timer.start())
            for cb in self.scale_cbs:
                cb.currentTextChanged.connect(lambda *_a: self._pv_timer.start())
            lay.addWidget(self.tabs, 1)
        else:
            lay.addWidget(names_tab, 1)

        form = QFormLayout()
        self.tsne_sample = QSpinBox(self)
        self.tsne_sample.setRange(500, 100000)
        self.tsne_sample.setSingleStep(1000)
        self.tsne_sample.setValue(10000)
        self.tsne_sample.setToolTip("Frames embedded by t-SNE (sampled over all sessions)")
        form.addRow("t-SNE frames:", self.tsne_sample)
        self.frames_per_session = QSpinBox(self)
        self.frames_per_session.setRange(100, 1000000)
        self.frames_per_session.setSingleStep(500)
        self.frames_per_session.setValue(3000)
        self.frames_per_session.setToolTip("Frames read from each session (uniform sample)")
        form.addRow("Frames per session:", self.frames_per_session)
        self.n_plot = QSpinBox(self)
        self.n_plot.setRange(1, 200)
        self.n_plot.setValue(24)
        self.n_plot.setToolTip("Joint PDFs drawn (best-ranked pairs of the selection)")
        form.addRow("Pairs plotted:", self.n_plot)
        if mode == "animate":
            self.n_pairs = QSpinBox(self)
            self.n_pairs.setRange(1, 20)
            self.n_pairs.setValue(4 if len(self.cols) > 2 else 1)
            self.n_pairs.setEnabled(len(self.cols) > 2)
            self.n_pairs.setToolTip("Joint spaces in the animation (best-ranked pairs); "
                                    "with two features selected, that one pair")
            form.addRow("Best-ranked joint spaces:", self.n_pairs)
            self._ranked = None
            run = find_reusable_run(config_path, self.cols) if config_path else None
            if run is not None:
                self._ranked = run / "pairs.csv"
            self.matrix.best_btn.clicked.connect(
                lambda: self.matrix.set_pairs(animation_pairs(
                    self.cols, self._ranked, self.n_pairs.value())))
            self.matrix.best_btn.setToolTip(
                "Tick the best-ranked pairs (up to 'Joint spaces')"
                + ("" if self._ranked else " — no landscape of this selection yet, so "
                   "the features are paired in order"))
            saved = []
            if config_path:
                from mufasa.utils.feature_view_state import load_state
                saved = [tuple(p) for p in load_state(config_path).get("animation_pairs", [])
                         if isinstance(p, list) and len(p) == 2
                         and p[0] in self.cols and p[1] in self.cols and p[0] != p[1]]
            self.matrix.set_pairs(saved or animation_pairs(self.cols, self._ranked,
                                                           self.n_pairs.value()))
            self.color = QComboBox(self)
            self.color.addItems(["tsne", "session"])
            self.color.setToolTip("Colour points by their t-SNE position or by session")
            form.addRow("Colour by:", self.color)
            # Patch 122jm: points, how they are picked, and the session intro
            self.n_points = QSpinBox(self)
            self.n_points.setRange(100, 100000)
            self.n_points.setSingleStep(500)
            self.n_points.setValue(4000)
            self.n_points.setToolTip("Points animated, taken from the t-SNE frames "
                                     "(at most that many)")
            form.addRow("Points in animation:", self.n_points)
            self.sampling = QComboBox(self)
            self.sampling.addItems(["random", "balanced", "even"])
            self.sampling.setToolTip(
                "random: uniformly (dense regions dominate, as in the data) · "
                "balanced: the same number from every session · "
                "even: spread over the t-SNE map, so sparse regions stay visible")
            form.addRow("Sampling:", self.sampling)
            # Patch 122jp: outliers left out of the animated points
            orow = QHBoxLayout()
            self.outliers = QComboBox(self)
            self.outliers.addItems(["none", "percentile", "mad"])
            self.outliers.setToolTip(
                "Leave out frames that are outliers in any animated feature, before "
                "the points are sampled. percentile: outside the given % tails; "
                "mad: farther than the given number of scaled MADs from the median")
            orow.addWidget(self.outliers)
            from PySide6.QtWidgets import QDoubleSpinBox
            self.outlier_value = QDoubleSpinBox(self)
            self.outlier_value.setDecimals(2)
            orow.addWidget(self.outlier_value)
            self.outliers_tsne = QCheckBox("also t-SNE coordinates", self)
            orow.addWidget(self.outliers_tsne)
            orow.addStretch()
            self._outlier_defaults = {"percentile": 1.0, "mad": 5.0}

            def _sync(m: str) -> None:
                on = m != "none"
                self.outlier_value.setEnabled(on)
                self.outliers_tsne.setEnabled(on)
                if m == "percentile":
                    self.outlier_value.setRange(0.01, 25.0)
                    self.outlier_value.setSuffix(" % each tail")
                elif m == "mad":
                    self.outlier_value.setRange(1.0, 50.0)
                    self.outlier_value.setSuffix(" × MAD")
                if on:
                    self.outlier_value.setValue(self._outlier_defaults[m])

            self.outliers.currentTextChanged.connect(_sync)
            _sync("none")
            form.addRow("Remove outliers:", orow)
            self.intro_frames = QSpinBox(self)
            self.intro_frames.setRange(0, 1000)
            self.intro_frames.setValue(40)
            self.intro_frames.setToolTip("First frames: the t-SNE map coloured by session, "
                                         "then the colours change to 'Colour by' (0 = none)")
            form.addRow("Intro frames (by session):", self.intro_frames)
            # Patch 122jo: timing in frames (the GIF runs at 20 fps)
            self.hold_frames = QSpinBox(self)
            self.hold_frames.setRange(0, 2000)
            self.hold_frames.setValue(30)
            self.hold_frames.setToolTip("Frames paused on each joint feature space "
                                        "(20 frames = 1 s)")
            form.addRow("Pause on each feature space (frames):", self.hold_frames)
            self.tsne_hold_frames = QSpinBox(self)
            self.tsne_hold_frames.setRange(0, 2000)
            self.tsne_hold_frames.setValue(30)
            self.tsne_hold_frames.setToolTip("Frames paused each time on the t-SNE map")
            form.addRow("Pause on t-SNE (frames):", self.tsne_hold_frames)
            self.transition_frames = QSpinBox(self)
            self.transition_frames.setRange(2, 2000)
            self.transition_frames.setValue(40)
            self.transition_frames.setToolTip("Frames per move between spaces")
            form.addRow("Transition (frames):", self.transition_frames)
            self.theme = QComboBox(self)
            self.theme.addItems(["dark", "light"])
            form.addRow("Theme:", self.theme)
            self.mp4 = QCheckBox("Also write an MP4", self)
            form.addRow("", self.mp4)
            self.reuse = QCheckBox("Reuse an earlier landscape of this selection", self)
            self.reuse.setChecked(True)
            self.reuse.setToolTip("Skip re-running mufasa-landscape (and its t-SNE) when "
                                  "a run on exactly these features exists")
            form.addRow("", self.reuse)
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        bb.button(QDialogButtonBox.Ok).setText(
            "Generate landscape" if mode == "landscape" else "Generate animation")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        self._ok = bb.button(QDialogButtonBox.Ok)
        if self.matrix is not None:
            self._pairs_changed()
        lay.addWidget(bb)

    def _reset_selected(self) -> None:
        from mufasa.utils.feature_names import default_name
        rows = {i.row() for i in self.table.selectedIndexes()} or set(range(len(self.cols)))
        for r in rows:
            self.table.item(r, 1).setText(default_name(self.cols[r]))

    def names(self) -> dict[str, str]:
        """{column: name in figures} as edited."""
        return {self.cols[r]: self.table.item(r, 1).text().strip()
                for r in range(len(self.cols))}

    def jitter(self) -> dict[str, bool]:
        """{column: jitter in figures?} as ticked."""
        return {c: self.table.item(r, 3).checkState() == Qt.Checked
                for r, c in enumerate(self.cols)}

    def scales(self) -> dict[str, str]:
        """{column: "linear" | "log10"} as chosen."""
        return {c: cb.currentText() for c, cb in zip(self.cols, self.scale_cbs)}

    def _names_edited(self, item: QTableWidgetItem) -> None:
        if self.matrix is not None and item.column() == 1:
            self.matrix.set_labels([self.table.item(r, 1).text()
                                    for r in range(len(self.cols))])
        if self.matrix is not None:
            self._pv_timer.start()

    # -- JPDF preview (patch 122jt) ------------------------------------ #
    PREVIEW_PER_VIDEO = 3000

    def preview_frame(self) -> pd.DataFrame:
        """The selected features pooled over every video (at most
        PREVIEW_PER_VIDEO frames each, -1 read as missing), read once."""
        if self._frame is None:
            parts = []
            if self.config_path:
                from mufasa.tools.feature_landscape import clean_columns
                from mufasa.utils.feature_io import (
                    list_video_stems_with_features,
                    load_feature_columns,
                )
                for v in list_video_stems_with_features(self.config_path):
                    try:
                        df = load_feature_columns(v, self.config_path, self.cols)
                    except Exception:
                        continue
                    if len(df) > self.PREVIEW_PER_VIDEO:
                        df = df.sample(self.PREVIEW_PER_VIDEO, random_state=0)
                    parts.append(df)
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if len(frame.columns):
                frame = clean_columns(frame.apply(pd.to_numeric, errors="coerce"),
                                      log=lambda *_a, **_k: None, drop_full_circle=False)
            self._frame = frame
        return self._frame

    def update_preview(self) -> None:
        """Draw the JPDF of the last ticked pair with the names, scales and
        jitter currently set in the names table."""
        if self.matrix is None or self.tabs.currentIndex() != 1:
            return
        prs = self.matrix.pairs()
        if not prs:
            self.preview.show_message("Tick a pair to see its joint distribution.")
            return
        a, b = prs[-1]
        frame = self.preview_frame()
        missing = [c for c in (a, b) if c not in frame.columns]
        if missing:
            self.preview.show_message("No values for: " + ", ".join(missing))
            return
        from mufasa.utils.feature_names import apply_scales, jitter_values
        names = self.names()
        sc = {c: v for c, v in self.scales().items() if v == "log10"}
        jit = self.jitter()
        sub, _ = apply_scales(frame[[a, b]], sc, log=lambda *_a, **_k: None)
        va, vb = (jitter_values(sub[c].to_numpy(np.float64), seed=k + 1) if jit.get(c)
                  else sub[c].to_numpy(np.float64) for k, c in enumerate((a, b)))
        label = lambda c: ("log10 " if c in sc else "") + (names.get(c) or c)  # noqa: E731
        self.preview.draw(va, vb, label(a), label(b), len(prs))

    def _pairs_changed(self) -> None:
        if getattr(self, "_pv_timer", None) is not None:
            self._pv_timer.start()
        n = len(self.matrix.pairs())
        self.tabs.setTabText(1, f"Pairs to animate ({n})" if n else "Pairs to animate")
        bb = getattr(self, "_ok", None)
        if bb is not None:
            bb.setEnabled(n > 0)

    def pairs(self) -> list[tuple[str, str]]:
        """The ticked pairs in animation order (animate mode)."""
        return self.matrix.pairs() if self.matrix is not None else []

    def accept(self) -> None:
        if self.config_path:
            from mufasa.utils.feature_names import save_names
            save_names(self.config_path, self.names(), self.scales(), self.jitter())
            if self.matrix is not None:
                from mufasa.utils.feature_view_state import load_state, save_state
                keep = [list(p) for p in load_state(self.config_path).get(
                    "animation_pairs", []) if not (set(p) <= set(self.cols))]
                save_state(self.config_path, {
                    "animation_pairs": keep + [list(p) for p in self.pairs()]})
        super().accept()

    def options(self) -> dict:
        o = dict(tsne_sample=self.tsne_sample.value(),
                 frames_per_session=self.frames_per_session.value(),
                 n_plot=self.n_plot.value())
        if self.mode == "animate":
            o.update(pairs=self.pairs(), n_pairs=self.n_pairs.value(), theme=self.theme.currentText(),
                     color=self.color.currentText(), mp4=self.mp4.isChecked(),
                     reuse=self.reuse.isChecked(), n_points=self.n_points.value(),
                     sampling=self.sampling.currentText(),
                     intro_frames=self.intro_frames.value(),
                     hold_frames=self.hold_frames.value(),
                     tsne_hold_frames=self.tsne_hold_frames.value(),
                     transition_frames=self.transition_frames.value(),
                     outliers=self.outliers.currentText(),
                     outlier_value=(self.outlier_value.value()
                                    if self.outliers.currentText() != "none" else None),
                     outliers_tsne=self.outliers_tsne.isChecked())
        return o


# --------------------------------------------------------------------------- #
# Running, with the log and the results
# --------------------------------------------------------------------------- #
class LandscapeRunDialog(QDialog):
    """Runs steps one after another as child Python processes; a step is a
    callable returning argv (after ``python``) or None to skip, so later
    steps can depend on earlier outputs. Shows the log, then the files."""

    def __init__(self, parent: QWidget | None, title: str, steps, out_dir: Path,
                 *, python: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 760)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._steps = list(steps)
        self._python = python or sys.executable
        self.out_dir = Path(out_dir)
        self.failed = False
        self.finished_ok = False
        lay = QVBoxLayout(self)
        self.status = QLabel("Starting…", self)
        lay.addWidget(self.status)
        split = QSplitter(Qt.Vertical, self)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        split.addWidget(self.log)
        res = QWidget(self)
        rl = QHBoxLayout(res)
        rl.setContentsMargins(0, 0, 0, 0)
        self.files = QListWidget(res)
        self.files.setMaximumWidth(260)
        self.files.currentItemChanged.connect(self._preview)
        self.files.itemDoubleClicked.connect(
            lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.UserRole))))
        rl.addWidget(self.files)
        self.view = QLabel("Results appear here when the run finishes.", res)
        self.view.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea(res)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.view)
        rl.addWidget(scroll, 1)
        split.addWidget(res)
        split.setSizes([260, 500])
        lay.addWidget(split, 1)
        brow = QHBoxLayout()
        self.open_btn = QPushButton("Open folder", self)
        self.open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir))))
        brow.addWidget(self.open_btn)
        brow.addStretch()
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop)
        brow.addWidget(self.stop_btn)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        brow.addWidget(close)
        lay.addLayout(brow)
        self._movie: QMovie | None = None
        self.proc: QProcess | None = None
        self._i = -1

    # -- process control ---------------------------------------------- #
    def start(self) -> None:
        self._next()

    def _next(self) -> None:
        self._i += 1
        while self._i < len(self._steps):
            try:
                argv = self._steps[self._i]()
            except Exception as exc:   # a step's own setup failed
                self._append(f"[error] {exc}")
                self._done(False)
                return
            if argv is not None:
                break
            self._i += 1
        else:
            self._done(True)
            return
        self.status.setText(f"Step {self._i + 1} of {len(self._steps)} running…")
        self._append("$ python " + " ".join(_quote(x) for x in argv))
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = p.processEnvironment()
        if env.isEmpty():
            from PySide6.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("MPLBACKEND", "Agg")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self.proc = p
        p.start(self._python, argv)

    def _read(self) -> None:
        if self.proc is not None:
            txt = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
            for ln in txt.rstrip("\n").splitlines():
                self._append(ln)

    def _on_error(self, err) -> None:  # noqa: ANN001
        if err == QProcess.FailedToStart:
            self._append(f"[error] could not start {self._python}")
            self._done(False)

    def _on_finished(self, code: int, _status) -> None:  # noqa: ANN001
        self._read()
        if code != 0:
            self._append(f"[error] step {self._i + 1} exited with code {code}")
            self._done(False)
            return
        self._next()

    def stop(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self._steps = self._steps[: self._i + 1]
            self.proc.kill()
            self._append("[stopped]")

    def closeEvent(self, ev) -> None:  # noqa: ANN001
        self.stop()
        super().closeEvent(ev)

    def _append(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _done(self, ok: bool) -> None:
        self.finished_ok, self.failed = ok, not ok
        self.stop_btn.setEnabled(False)
        self.status.setText(f"Done — {self.out_dir}" if ok else
                            "Failed — see the log above.")
        self.show_results()

    # -- results -------------------------------------------------------- #
    def result_files(self) -> list[Path]:
        if not self.out_dir.is_dir():
            return []
        order = {".gif": 0, ".png": 1, ".mp4": 2, ".pdf": 3, ".csv": 4}
        fs = [p for p in self.out_dir.iterdir() if p.suffix.lower() in order]
        return sorted(fs, key=lambda p: (order[p.suffix.lower()],
                                         -p.stat().st_mtime, p.name))

    def show_results(self) -> None:
        self.files.clear()
        for p in self.result_files():
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            self.files.addItem(it)
        if self.files.count():
            self.files.setCurrentRow(0)

    def _preview(self, cur: QListWidgetItem | None, _prev=None) -> None:  # noqa: ANN001
        if self._movie is not None:
            self._movie.stop()
            self._movie = None
        if cur is None:
            return
        p = Path(cur.data(Qt.UserRole))
        suf = p.suffix.lower()
        if suf == ".gif":
            self._movie = QMovie(str(p))
            self.view.setMovie(self._movie)
            self._movie.start()
        elif suf == ".png":
            pm = QPixmap(str(p))
            w = max(400, self.view.parentWidget().width() - 20)
            self.view.setPixmap(pm.scaledToWidth(min(w, pm.width()),
                                                 Qt.SmoothTransformation)
                                if not pm.isNull() else pm)
        else:
            self.view.setText(f"{p.name}\n\nDouble-click to open with the system viewer.")


def _wrap(text: str, width: int) -> str:
    """Break ``text`` at spaces into lines of about ``width`` characters."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width, break_long_words=False)) or text


def _quote(s: str) -> str:
    return f'"{s}"' if any(ch in s for ch in " ()'\"") else s


# --------------------------------------------------------------------------- #
# Entry points used by the Feature Visualization form
# --------------------------------------------------------------------------- #
def run_landscape(parent: QWidget | None, config_path, cols: list[str], *,
                  ask: bool = True, options: dict | None = None,
                  start: bool = True) -> LandscapeRunDialog | None:
    """Names dialog, then mufasa-landscape on ``cols``; returns the run
    dialog (None if cancelled)."""
    from mufasa.project_layout import generate_run_id
    from mufasa.utils.feature_names import names_path
    opts = options or {}
    if ask:
        d = FeatureNamesDialog(parent, config_path, cols, "landscape")
        if d.exec() != QDialog.Accepted:
            return None
        opts = {**d.options(), **opts}
    opts = {"tsne_sample": 10000, "frames_per_session": 3000, "n_plot": 24, **opts}
    out = landscape_root(config_path) / generate_run_id()
    out.mkdir(parents=True, exist_ok=True)
    sel = out / SELECTION_FILE
    sel.write_text("\n".join(cols) + "\n")
    nf = names_path(config_path)
    names_file = nf if nf.is_file() else None
    steps = [lambda: landscape_argv(config_path, sel, out, names_file,
                                    tsne_sample=opts["tsne_sample"],
                                    frames_per_session=opts["frames_per_session"],
                                    n_plot=opts["n_plot"])]
    dlg = LandscapeRunDialog(parent, f"Landscape — {len(cols)} features", steps, out)
    dlg.show()
    if start:
        dlg.start()
    return dlg


def run_animation(parent: QWidget | None, config_path, cols: list[str], *,
                  ask: bool = True, options: dict | None = None,
                  start: bool = True) -> LandscapeRunDialog | None:
    """Names dialog, then (unless an earlier run on this selection is
    reused) mufasa-landscape, then mufasa-landscape-animate --through-tsne
    on the best-ranked pairs of the selection."""
    from mufasa.project_layout import generate_run_id
    from mufasa.utils.feature_names import names_path
    opts = options or {}
    if ask:
        d = FeatureNamesDialog(parent, config_path, cols, "animate")
        if d.exec() != QDialog.Accepted:
            return None
        opts = {**d.options(), **opts}
    opts = {"tsne_sample": 10000, "frames_per_session": 3000, "n_plot": 24,
            "n_pairs": 4, "theme": "dark", "color": "tsne", "mp4": False,
            "reuse": True, "n_points": 4000, "sampling": "random", "intro_frames": 40,
            "hold_frames": 30, "tsne_hold_frames": 30, "transition_frames": 40,
            "outliers": "none", "outlier_value": None, "outliers_tsne": False, **opts}
    nf = names_path(config_path)
    names_file = nf if nf.is_file() else None
    run = find_reusable_run(config_path, cols) if opts["reuse"] else None
    steps = []
    if run is None:
        run = landscape_root(config_path) / generate_run_id()
        run.mkdir(parents=True, exist_ok=True)
        sel = run / SELECTION_FILE
        sel.write_text("\n".join(cols) + "\n")
        steps.append(lambda: landscape_argv(config_path, sel, run, names_file,
                                            tsne_sample=opts["tsne_sample"],
                                            frames_per_session=opts["frames_per_session"],
                                            n_plot=opts["n_plot"]))

    def _animate() -> list[str]:
        pairs = (list(opts["pairs"]) if opts.get("pairs") else
                 animation_pairs(cols, run / "pairs.csv", opts["n_pairs"]))
        k, out = 1, run / "animation.gif"
        while out.exists():
            out = run / f"animation_{k}.gif"
            k += 1
        return animate_argv(run, pairs, out, names_file, theme=opts["theme"],
                            mp4=opts["mp4"], color=opts["color"],
                            n_points=opts["n_points"], sampling=opts["sampling"],
                            intro_frames=opts["intro_frames"],
                            hold_frames=opts["hold_frames"],
                            tsne_hold_frames=opts["tsne_hold_frames"],
                            transition_frames=opts["transition_frames"],
                            outliers=opts["outliers"], outlier_value=opts["outlier_value"],
                            outliers_tsne=opts["outliers_tsne"])

    steps.append(_animate)
    dlg = LandscapeRunDialog(parent, f"Animation — {len(cols)} features", steps, run)
    if len(steps) == 1:
        dlg._append(f"[reusing landscape run {run.name}]")
    dlg.show()
    if start:
        dlg.start()
    return dlg


__all__ = ["FeatureNamesDialog", "LandscapeRunDialog", "PairMatrix", "PairPreview",
           "animate_argv", "animation_pairs", "find_reusable_run", "landscape_argv",
           "landscape_root", "run_animation", "run_landscape"]
