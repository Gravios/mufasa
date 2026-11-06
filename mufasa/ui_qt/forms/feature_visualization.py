"""
mufasa.ui_qt.forms.feature_visualization
========================================

Features → **Feature Visualization** (patch 122je).

Pick a video and any computed features — shown as a tree, one branch per
feature family (tick a family to take all its shown features; search
filters across families) — then:

* **Distributions** — a histogram per feature, for this video or pooled
  over all videos. With *Colour by* set to a behaviour, labelled frames
  are drawn against the rest and each panel shows the feature's AUC for
  that behaviour.
* **Joint** — two features as a heatmap (2-D density), a scatter, or a
  scatter over the heatmap; behaviour frames overlaid.
* **Over time** — the whole session per feature with behaviour bouts
  shaded; double-click a time to open the video there.
* **Landscape… / Animate…** — the selection through mufasa-landscape
  (joint PDFs + t-SNE over all sessions) and mufasa-landscape-animate
  (t-SNE → pair → t-SNE → next pair GIF); a table to name the features in
  the figures comes first, and the names are remembered (122jk).
* **Open with video** — the video with the selected features as live
  traces under it (click / drag the traces to move, wheel to zoom); open
  viewers follow the selection as features are ticked / unticked (122jk).

Only the selected columns are read (parquet column projection), so this
stays quick with hundreds of features and dozens of sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm

MAX_DIST_FEATURES = 12
MAX_TIME_FEATURES = 8
MAX_VIEWER_FEATURES = 16


class FeatureVisualizationForm(OperationForm):
    title = "Feature Visualization"
    description = (
        "Look at computed features: distributions (optionally split by a "
        "behaviour, with the AUC per feature), two features jointly, whole "
        "sessions over time, or live alongside the video."
    )

    # ------------------------------------------------------------------ #
    def build(self) -> None:
        self.run_btn.setVisible(False)
        self._checked: list[str] = []          # selection order kept
        self._catalog: dict[str, list[str]] = {}
        self._videos: dict[str, str | None] = {}

        row = QHBoxLayout()
        row.addWidget(QLabel("Video:", self))
        self.video_cb = QComboBox(self)
        self.video_cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_cb.currentIndexChanged.connect(self._on_video_changed)
        row.addWidget(self.video_cb, 3)
        row.addWidget(QLabel("Scope:", self))
        self.scope_cb = QComboBox(self)
        self.scope_cb.addItems(["This video", "All videos"])
        self.scope_cb.setToolTip("Distributions and joint plots: this video, or "
                                 "all videos with features pooled")
        row.addWidget(self.scope_cb)
        row.addWidget(QLabel("Colour by:", self))
        self.colour_cb = QComboBox(self)
        self.colour_cb.setToolTip("Split frames by a labelled behaviour")
        row.addWidget(self.colour_cb, 1)
        refresh = QPushButton("↻", self)
        refresh.setToolTip("Re-scan videos and features")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        self.body_layout.addLayout(row)

        frow = QHBoxLayout()
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search features (space-separated words)…")
        self.search.setClearButtonEnabled(True)
        # debounce: re-filter 200 ms after the last keystroke (big trees)
        from PySide6.QtCore import QTimer
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(self._apply_filter)
        self.search.textChanged.connect(lambda _t: self._search_timer.start())
        frow.addWidget(self.search, 3)
        from PySide6.QtWidgets import QCheckBox
        self.compact_cb = QCheckBox("Group by name parts", self)
        self.compact_cb.setChecked(True)
        self.compact_cb.setToolTip(
            "Inside each family, group features by the parts of their names "
            "they share (usually the first body part), dropping the wording "
            "every feature of the family has in common")
        self.compact_cb.toggled.connect(lambda _on: self._build_tree())
        frow.addWidget(self.compact_cb)
        for text, slot in (("Expand all", lambda: self.tree.expandAll()),
                           ("Collapse all", lambda: self.tree.collapseAll()),
                           ("Clear selection", self._clear_selection)):
            b = QPushButton(text, self)
            b.clicked.connect(slot)
            frow.addWidget(b)
        self.body_layout.addLayout(frow)

        # Patch 122jf: features as a tree — one branch per feature family.
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        from PySide6.QtWidgets import QHeaderView
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(180)
        self.tree.setMaximumHeight(300)
        self.tree.itemChanged.connect(self._on_item_changed)
        # expansion remembered by node path across rebuilds and view toggles
        self._expanded: set[tuple] = set()
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.itemCollapsed.connect(self._on_collapsed)
        self._syncing = False
        self.body_layout.addWidget(self.tree)
        self.sel_label = QLabel("", self)
        self.sel_label.setWordWrap(True)
        self.sel_label.setStyleSheet("color: palette(placeholder-text);")
        self.body_layout.addWidget(self.sel_label)

        brow = QHBoxLayout()
        self.dist_btn = QPushButton("Distributions", self)
        self.dist_btn.clicked.connect(self.plot_distributions)
        self.joint_btn = QPushButton("Joint (2 features)", self)
        self.joint_btn.clicked.connect(self.plot_joint)
        self.time_btn = QPushButton("Over time", self)
        self.time_btn.clicked.connect(self.plot_over_time)
        self.video_btn = QPushButton("Open with video ▸", self)
        self.video_btn.clicked.connect(lambda: self.open_viewer())
        # Patch 122jk: the selection through mufasa-landscape / -animate
        self.landscape_btn = QPushButton("Landscape…", self)
        self.landscape_btn.setToolTip(
            "mufasa-landscape on the selected features, pooled over all sessions: "
            "joint PDFs of their pairs and a t-SNE (asks for figure names first)")
        # clicked(bool) would land in ``ask`` — call with no arguments
        self.landscape_btn.clicked.connect(lambda: self.generate_landscape())
        self.animate_btn = QPushButton("Animate…", self)
        self.animate_btn.setToolTip(
            "GIF: t-SNE → joint space 1 → t-SNE → joint space 2 → … of the "
            "selected features (runs the landscape first if needed)")
        self.animate_btn.clicked.connect(lambda: self.generate_animation())
        for b in (self.dist_btn, self.joint_btn, self.time_btn):
            brow.addWidget(b)
        brow.addStretch()
        brow.addWidget(self.landscape_btn)
        brow.addWidget(self.animate_btn)
        brow.addWidget(self.video_btn)
        self.body_layout.addLayout(brow)

        # Patch 122jj — axis controls; applied to the plot on screen at once.
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Axes:", self))
        self.swap_btn = QPushButton("Swap X ↔ Y", self)
        self.swap_btn.setCheckable(True)
        self.swap_btn.setToolTip("Exchange the horizontal and vertical axes")
        self.swap_btn.toggled.connect(self._on_swap)
        arow.addWidget(self.swap_btn)
        self.xscale_cb, self.yscale_cb = QComboBox(self), QComboBox(self)
        for lab, cb in (("X:", self.xscale_cb), ("Y:", self.yscale_cb)):
            arow.addWidget(QLabel(lab, self))
            cb.addItems(["linear", "log10"])
            cb.setToolTip("Scale of the displayed axis. On a log axis, values ≤ 0 "
                          "are left out (each plot says how many).")
            cb.currentIndexChanged.connect(self._replot)
            arow.addWidget(cb)
        from mufasa.tools.feature_plots import JOINT_STYLES
        arow.addWidget(QLabel("Joint:", self))
        self.joint_style_cb = QComboBox(self)
        self.joint_style_cb.addItems(list(JOINT_STYLES))
        self.joint_style_cb.setToolTip(
            "heatmap: 2-D log density · scatter: a sample of frames as points · "
            "scatter-heatmap: points over the density")
        self.joint_style_cb.currentIndexChanged.connect(
            lambda _i: self._mode == "joint" and self._replot())
        arow.addWidget(self.joint_style_cb)
        arow.addStretch()
        self.body_layout.addLayout(arow)
        self._cache: dict = {}
        # Patch 122jk: open video × feature viewers follow the selection
        from PySide6.QtCore import QTimer as _QT
        self._viewers: list = []
        self._sel_timer = _QT(self)
        self._sel_timer.setSingleShot(True)
        self._sel_timer.setInterval(300)
        self._sel_timer.timeout.connect(self._sync_viewers)

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(420)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        host = QWidget(self)
        hl = QVBoxLayout(host)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(NavigationToolbar2QT(self.canvas, host))
        hl.addWidget(self.canvas)
        self.body_layout.addWidget(host)
        self.plot_info = QLabel("", self)
        self.plot_info.setWordWrap(True)
        self.plot_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body_layout.addWidget(self.plot_info)
        self._mode: str | None = None
        self._time_fps = 30.0
        self.refresh()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        self._cache = {}
        self.video_cb.blockSignals(True)
        cur = self.video_cb.currentText()
        self.video_cb.clear()
        self._videos = {}
        if self.config_path:
            try:
                from mufasa.utils.annotation_status import project_videos
                from mufasa.utils.feature_io import list_video_stems_with_features
                vids = project_videos(self.config_path)
                for stem in list_video_stems_with_features(self.config_path):
                    self._videos[stem] = vids.get(stem)
            except Exception as exc:
                self.sel_label.setText(f"Could not list features: {exc}")
        self.video_cb.addItems(sorted(self._videos))
        if cur in self._videos:
            self.video_cb.setCurrentText(cur)
        self.video_cb.blockSignals(False)
        self.colour_cb.clear()
        self.colour_cb.addItem("(none)")
        if self.config_path:
            from mufasa.ui_qt.forms.analysis import _load_classifier_names
            self.colour_cb.addItems(_load_classifier_names(self.config_path))
        if not self._videos:
            self.sel_label.setText(
                "No computed features found — run Compute Features first."
                if self.config_path else "No project loaded.")
        self._on_video_changed()

    def _on_video_changed(self, *_a) -> None:
        video = self.video_cb.currentText()
        self._catalog = {}
        if video and self.config_path:
            from mufasa.utils.feature_io import feature_catalog
            self._catalog = feature_catalog(video, self.config_path)
        have = {c for cols in self._catalog.values() for c in cols}
        self._checked = [c for c in self._checked if c in have]
        self._build_tree()
        self.video_btn.setEnabled(bool(self._videos.get(video)))
        self.video_btn.setToolTip("" if self._videos.get(video) else
                                  "This video is not in sources/videos")

    def _family_of(self, col: str) -> str:
        for f, cols in self._catalog.items():
            if col in cols:
                return f
        return ""

    @staticmethod
    def family_title(family: str) -> str:
        return family.replace("_", " ")

    # Tree: family → [component group → [sub-group →]] feature. Leaves carry
    # the column in UserRole; families carry ("family", name) and groups
    # ("group", label) in UserRole + 1.  (122jf / 122jg)
    _KIND = Qt.UserRole + 1

    def _build_tree(self) -> None:
        """Families as branches; inside each, features grouped by shared
        name components (patch 122jg, ``compact_family``). Expansion is
        kept per node path across rebuilds."""
        from mufasa.tools.feature_plots import compact_family
        expanded = self._expanded
        self._syncing = True
        self.tree.clear()
        for fam, cols in self._catalog.items():
            template, nodes = compact_family(cols) if self.compact_cb.isChecked() \
                else compact_family(cols, min_family=10 ** 9)
            top = QTreeWidgetItem([self.family_title(fam), template])
            top.setData(0, self._KIND, ("family", fam))
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
            top.setToolTip(0, f"{fam}\nnames: {template or '(various)'}\n"
                              "Tick to select every shown feature of this family")
            top.setForeground(1, self.palette().placeholderText())
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self._add_nodes(top, nodes)
            self.tree.addTopLevelItem(top)
        for it in self._walk():
            it.setExpanded(self._path(it) in expanded)
        self._syncing = False
        self._apply_filter()

    def _on_expanded(self, it: QTreeWidgetItem) -> None:
        if not self._syncing:
            self._expanded.add(self._path(it))

    def _on_collapsed(self, it: QTreeWidgetItem) -> None:
        if not self._syncing:
            self._expanded.discard(self._path(it))

    def _add_nodes(self, parent: QTreeWidgetItem, nodes) -> None:
        for n in nodes:
            it = QTreeWidgetItem([n.label])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            if n.col is not None:
                it.setData(0, Qt.UserRole, n.col)
                it.setToolTip(0, n.col)
                it.setCheckState(0, Qt.Checked if n.col in self._checked else Qt.Unchecked)
            else:
                it.setData(0, self._KIND, ("group", n.label))
                it.setToolTip(0, f"{n.label}: {len(n.leaves())} features — tick to "
                                 "select every shown one")
                self._add_nodes(it, n.children)
            parent.addChild(it)

    def _walk(self, root: QTreeWidgetItem | None = None):
        """Every item, depth first."""
        items = ([self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
                 if root is None else [root.child(i) for i in range(root.childCount())])
        for it in items:
            yield it
            yield from self._walk(it)

    def _path(self, it: QTreeWidgetItem) -> tuple:
        out = []
        while it is not None:
            out.append(it.data(0, self._KIND) or it.data(0, Qt.UserRole))
            it = it.parent()
        return tuple(reversed(out))

    @staticmethod
    def _is_leaf(it: QTreeWidgetItem) -> bool:
        return it.data(0, Qt.UserRole) is not None

    def _children(self, it: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        return [it.child(i) for i in range(it.childCount())]

    def _leaves(self, it: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        return [x for x in self._walk(it) if self._is_leaf(x)]

    def _refresh_node(self, it: QTreeWidgetItem) -> None:
        """Checkbox (none / some / all) and counts of a family or group."""
        leaves = self._leaves(it)
        n_on = sum(x.checkState(0) == Qt.Checked for x in leaves)
        n_vis = sum(not x.isHidden() for x in leaves)
        kind, name = it.data(0, self._KIND)
        label = self.family_title(name) if kind == "family" else name
        shown = f"{n_vis}/{len(leaves)} shown · " if n_vis != len(leaves) else ""
        it.setText(0, f"{label}   ({shown}{n_on}/{len(leaves)} selected)" if n_on
                   else f"{label}   ({shown}{len(leaves)})")
        it.setCheckState(0, Qt.Checked if leaves and n_on == len(leaves)
                         else Qt.PartiallyChecked if n_on else Qt.Unchecked)

    def _refresh_up(self, it: QTreeWidgetItem | None) -> None:
        while it is not None:
            self._refresh_node(it)
            it = it.parent()

    def _refresh_all(self) -> None:
        # deepest first so parents see their children's states
        nodes = [it for it in self._walk() if not self._is_leaf(it)]
        depth = {id(it): len(self._path(it)) for it in nodes}
        for it in sorted(nodes, key=lambda x: -depth[id(x)]):
            self._refresh_node(it)

    def _apply_filter(self, *_a) -> None:
        """Hide features not matching every search word (feature, group and
        family names all count); hide empty groups / families; open the
        branches that hold matches while searching."""
        words = [w.lower() for w in self.search.text().split()]
        self._syncing = True
        shown = total = 0
        for it in self._walk():
            if not self._is_leaf(it):
                continue
            ctx = " ".join(str(p) for p in self._path(it)).lower()
            hit = not words or all(w in ctx for w in words)
            it.setHidden(not hit)
            total += 1
            shown += hit
        nodes = [it for it in self._walk() if not self._is_leaf(it)]
        for it in reversed(nodes):                   # children before parents
            vis = any(not x.isHidden() for x in self._leaves(it))
            it.setHidden(not vis)
            if words and vis:
                it.setExpanded(True)
        self._refresh_all()
        self._syncing = False
        self._update_sel_label(shown, total)

    def _on_item_changed(self, it: QTreeWidgetItem, _col: int = 0) -> None:
        if self._syncing or _col != 0:
            return
        self._syncing = True
        try:
            if self._is_leaf(it):
                self._track(it)
                self._refresh_up(it.parent())
            else:
                # family / group checkbox: (un)select its *shown* features
                on = it.checkState(0) != Qt.Unchecked
                for leaf in self._leaves(it):
                    if self._hidden_in(leaf, it):
                        continue
                    leaf.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
                    self._track(leaf)
                for sub in [x for x in self._walk(it) if not self._is_leaf(x)][::-1]:
                    self._refresh_node(sub)
                self._refresh_up(it)
        finally:
            self._syncing = False
        self._update_sel_label()

    @staticmethod
    def _hidden_in(leaf: QTreeWidgetItem, root: QTreeWidgetItem) -> bool:
        x = leaf
        while x is not None and x is not root:
            if x.isHidden():
                return True
            x = x.parent()
        return False

    def _track(self, ch: QTreeWidgetItem) -> None:
        c = ch.data(0, Qt.UserRole)
        if ch.checkState(0) == Qt.Checked and c not in self._checked:
            self._checked.append(c)
        elif ch.checkState(0) != Qt.Checked and c in self._checked:
            self._checked.remove(c)

    def _clear_selection(self) -> None:
        self._checked = []
        self._build_tree()

    def _update_sel_label(self, shown: int | None = None, total: int | None = None) -> None:
        from mufasa.tools.feature_plots import short_name
        head = (f"{shown:,} of {total:,} features shown · " if shown is not None else "")
        if not self._checked:
            self.sel_label.setText(head + "none selected — tick features above")
        else:
            names = ", ".join(short_name(c) for c in self._checked[:4])
            more = f" … +{len(self._checked) - 4}" if len(self._checked) > 4 else ""
            self.sel_label.setText(head + f"<b>{len(self._checked)} selected</b>: {names}{more}")
        if getattr(self, "_viewers", None):
            self._sel_timer.start()

    # ------------------------------------------------------------------ #
    # Selection API (also used by tests)
    # ------------------------------------------------------------------ #
    def selected_features(self) -> list[str]:
        return list(self._checked)

    def set_selected(self, cols: list[str]) -> None:
        self._checked = list(cols)
        self._build_tree()

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _behaviour(self) -> str | None:
        b = self.colour_cb.currentText()
        return None if not b or b == "(none)" else b

    def _axes(self) -> dict:
        """Patch 122jj: axis options for the plotting functions."""
        return dict(swap=self.swap_btn.isChecked(),
                    xlog=self.xscale_cb.currentText() == "log10",
                    ylog=self.yscale_cb.currentText() == "log10")

    def _on_swap(self, _on: bool) -> None:
        """Swap also exchanges the X / Y scale settings, so each quantity
        keeps its scale when it moves to the other axis."""
        x, y = self.xscale_cb.currentIndex(), self.yscale_cb.currentIndex()
        for cb, i in ((self.xscale_cb, y), (self.yscale_cb, x)):
            cb.blockSignals(True)
            cb.setCurrentIndex(i)
            cb.blockSignals(False)
        self._replot()

    def _replot(self, *_a) -> None:
        """Redraw the plot on screen with the current axis options (data
        come from the cache, so nothing is re-read)."""
        {"dist": self.plot_distributions, "joint": self.plot_joint,
         "time": self.plot_over_time}.get(self._mode, lambda: None)()

    def _load(self, cols: list[str], all_videos: bool):
        key = ("load", tuple(cols), all_videos, self._behaviour(),
               self.video_cb.currentText())
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != "load"}
            self._cache[key] = self._load_uncached(cols, all_videos)
        return self._cache[key]

    def _load_uncached(self, cols: list[str], all_videos: bool):
        """(data {col: array}, behaviour mask or None, n videos)."""
        from mufasa.utils.feature_io import load_feature_columns
        from mufasa.utils.label_io import load_labels_for_video
        beh = self._behaviour()
        videos = sorted(self._videos) if all_videos else [self.video_cb.currentText()]
        parts, masks = [], []
        self._unlabelled: list[str] = []
        for v in videos:
            df = load_feature_columns(v, self.config_path, cols)
            if df.empty:
                continue
            df = df.reindex(columns=cols)
            if beh:
                # Frames of a video nobody labelled are unknown, not
                # "without the behaviour": leave such videos out of a split.
                try:
                    lab = load_labels_for_video(v, self.config_path, targets=[beh])[beh]
                    lab = pd.to_numeric(lab, errors="coerce")
                except Exception:
                    lab = pd.Series(dtype=float)
                if lab.notna().sum() == 0:
                    self._unlabelled.append(v)
                    continue
                m = (lab == 1).to_numpy()
                mm = np.zeros(len(df), bool)
                k = min(len(m), len(df))
                mm[:k] = m[:k]
                masks.append(mm)
            parts.append(df)
        if not parts:
            return {}, None, 0
        X = pd.concat(parts, ignore_index=True)
        data = {c: pd.to_numeric(X[c], errors="coerce").to_numpy(np.float64) for c in cols}
        mask = np.concatenate(masks) if beh else None
        return data, mask, len(parts)

    def _need(self, lo: int, hi: int | None, what: str) -> list[str] | None:
        cols = self.selected_features()
        if not self.config_path or not self._videos:
            QMessageBox.information(self.window(), self.title, "No features to plot.")
            return None
        if len(cols) < lo or (hi is not None and len(cols) > hi):
            want = f"exactly {lo}" if hi == lo else (f"{lo}–{hi}" if hi else f"at least {lo}")
            QMessageBox.information(self.window(), self.title,
                                    f"{what} needs {want} selected feature(s); "
                                    f"{len(cols)} selected.")
            return None
        return cols

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    def plot_distributions(self) -> dict | None:
        cols = self._need(1, MAX_DIST_FEATURES, "Distributions")
        if cols is None:
            return None
        from mufasa.tools.feature_plots import plot_distributions, short_name
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            all_v = self.scope_cb.currentIndex() == 1
            data, mask, nv = self._load(cols, all_v)
            beh = self._behaviour() or ""
            aucs = plot_distributions(self.figure, data, mask=mask, behaviour=beh,
                                      **self._axes())
            self.canvas.draw_idle()
        finally:
            QApplication.restoreOverrideCursor()
        self._mode = "dist"
        n = len(next(iter(data.values()))) if data else 0
        txt = f"{n:,} frames from {nv} video(s)."
        if getattr(self, "_unlabelled", None):
            txt += (f" {len(self._unlabelled)} video(s) without '{beh}' labels left "
                    "out of the split.")
        if not data:
            self.plot_info.setText(f"No labelled frames for '{beh}' in this scope."
                                   if beh else "No feature data.")
            return {}
        if mask is not None:
            txt += f" {int(mask.sum()):,} labelled '{beh}'."
            if aucs:
                best = sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))
                txt += " Best separating: " + ", ".join(
                    f"{short_name(c)} (AUC {a:.2f})" for c, a in best[:3]
                    if np.isfinite(a))
        self.plot_info.setText(txt)
        return aucs

    def plot_joint(self) -> None:
        cols = self._need(2, 2, "Joint")
        if cols is None:
            return
        from mufasa.tools.feature_plots import plot_joint
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data, mask, nv = self._load(cols, self.scope_cb.currentIndex() == 1)
            from mufasa.utils.feature_names import display, load_names
            names = load_names(self.config_path)
            plot_joint(self.figure, data[cols[0]], data[cols[1]],
                       display(cols[0], names), display(cols[1], names),
                       mask=mask, behaviour=self._behaviour() or "",
                       style=self.joint_style_cb.currentText(), **self._axes())
            self.canvas.draw_idle()
        finally:
            QApplication.restoreOverrideCursor()
        self._mode = "joint"
        self.plot_info.setText(f"{nv} video(s).")

    def plot_over_time(self) -> None:
        cols = self._need(1, MAX_TIME_FEATURES, "Over time")
        if cols is None:
            return
        from mufasa.tools.feature_plots import plot_over_time
        from mufasa.utils.feature_io import load_feature_columns
        video = self.video_cb.currentText()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            key = ("time", video, tuple(cols))
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != "time"}
                self._cache[key] = (load_feature_columns(video, self.config_path, cols),
                                    self._video_fps(video), self._labels(video))
            df, fps, labels = self._cache[key]
            plot_over_time(self.figure, {c: df[c].to_numpy(np.float64) for c in df.columns},
                           fps, labels=labels, **self._axes())
            self.canvas.draw_idle()
        finally:
            QApplication.restoreOverrideCursor()
        self._mode = "time"
        self._time_fps = fps
        self.plot_info.setText(f"{video}: {len(df):,} frames at {fps:g} fps. "
                               "Double-click a time to open the video there.")

    def _labels(self, video: str) -> dict[str, np.ndarray]:
        try:
            from mufasa.utils.label_io import load_labels_for_video
            df = load_labels_for_video(video, self.config_path)
        except Exception:
            return {}
        out = {c: pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in df.columns}
        return {k: v for k, v in out.items() if (v == 1).any()}

    def _video_fps(self, video: str) -> float:
        path = self._videos.get(video)
        if path:
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return float(fps)
            except Exception:
                pass
        return 30.0

    def _on_canvas_click(self, ev) -> None:  # noqa: ANN001
        if self._mode == "time" and ev.dblclick and ev.inaxes is not None:
            t = ev.ydata if self.swap_btn.isChecked() else ev.xdata   # minutes
            if t is not None:
                self.open_viewer(start_frame=max(0, int(round(t * 60 * self._time_fps))))

    # ------------------------------------------------------------------ #
    # Patch 122jk — landscape / animation of the selection
    # ------------------------------------------------------------------ #
    def generate_landscape(self, ask: bool = True, options: dict | None = None):
        cols = self._need(2, None, "Landscape")
        if cols is None:
            return None
        from mufasa.ui_qt.landscape_runner import run_landscape
        self._run = run_landscape(self.window(), self.config_path, cols, ask=ask,
                                  options=options)
        return self._run

    def generate_animation(self, ask: bool = True, options: dict | None = None):
        cols = self._need(2, None, "Animate")
        if cols is None:
            return None
        from mufasa.ui_qt.landscape_runner import run_animation
        self._run = run_animation(self.window(), self.config_path, cols, ask=ask,
                                  options=options)
        return self._run

    # ------------------------------------------------------------------ #
    def open_viewer(self, start_frame: int = 0):
        cols = self._need(1, None, "Open with video")
        if cols is None:
            return None
        video = self.video_cb.currentText()
        path = self._videos.get(video)
        if not path:
            QMessageBox.warning(self.window(), self.title,
                                f"'{video}' is not in sources/videos.")
            return None
        from mufasa.ui_qt.feature_viewer import open_feature_viewer
        self._viewer = open_feature_viewer(self.window(), self.config_path, path,
                                           cols[:MAX_VIEWER_FEATURES], start_frame)
        self._viewers.append(self._viewer)
        return self._viewer

    def _sync_viewers(self) -> int:
        """Push the current selection to every open viewer that follows it
        (each keeps its own video); returns how many were updated."""
        from shiboken6 import isValid
        cols = self.selected_features()[:MAX_VIEWER_FEATURES]
        self._viewers = [v for v in self._viewers if isValid(v) and v.isVisible()]
        n = 0
        for v in self._viewers:
            w = getattr(v, "viewer", None)
            if w is not None and isValid(w) and cols:
                try:
                    n += bool(w.follow(cols))
                except Exception as exc:   # a column missing for that video
                    self.plot_info.setText(f"Viewer not updated: {exc}")
        return n


__all__ = ["FeatureVisualizationForm"]
