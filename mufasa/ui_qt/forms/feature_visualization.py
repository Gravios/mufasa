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
* **Joint** — two features as a 2-D density, behaviour frames overlaid.
* **Over time** — the whole session per feature with behaviour bouts
  shaded; double-click a time to open the video there.
* **Open with video** — the video with the selected features as live
  traces under it (click / drag the traces to move, wheel to zoom).

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
        self.search.textChanged.connect(self._apply_filter)
        frow.addWidget(self.search, 3)
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
        self.tree.setColumnCount(1)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(180)
        self.tree.setMaximumHeight(300)
        self.tree.itemChanged.connect(self._on_item_changed)
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
        for b in (self.dist_btn, self.joint_btn, self.time_btn):
            brow.addWidget(b)
        brow.addStretch()
        brow.addWidget(self.video_btn)
        self.body_layout.addLayout(brow)

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

    def _build_tree(self) -> None:
        """One top-level item per family, one checkable child per feature.
        Expansion state is kept per family across rebuilds."""
        from mufasa.tools.feature_plots import short_name
        expanded = {self.tree.topLevelItem(i).data(0, Qt.UserRole)
                    for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).isExpanded()}
        self._syncing = True
        self.tree.clear()
        for fam, cols in self._catalog.items():
            top = QTreeWidgetItem([self.family_title(fam)])
            top.setData(0, Qt.UserRole, fam)
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
            top.setToolTip(0, f"{fam}\nTick to select every (shown) feature of this family")
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for c in cols:
                ch = QTreeWidgetItem([short_name(c)])
                ch.setData(0, Qt.UserRole, c)
                ch.setToolTip(0, c)
                ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
                ch.setCheckState(0, Qt.Checked if c in self._checked else Qt.Unchecked)
                top.addChild(ch)
            self.tree.addTopLevelItem(top)
            top.setExpanded(fam in expanded)
        self._syncing = False
        self._apply_filter()

    def _children(self, top: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        return [top.child(i) for i in range(top.childCount())]

    def _refresh_family(self, top: QTreeWidgetItem) -> None:
        """Family checkbox (checked / partial / empty) and title counts."""
        kids = self._children(top)
        n_on = sum(k.checkState(0) == Qt.Checked for k in kids)
        n_vis = sum(not k.isHidden() for k in kids)
        fam = top.data(0, Qt.UserRole)
        shown = f"{n_vis}/{len(kids)} shown · " if n_vis != len(kids) else ""
        top.setText(0, f"{self.family_title(fam)}   ({shown}{n_on}/{len(kids)} selected)"
                    if n_on else f"{self.family_title(fam)}   ({shown}{len(kids)})")
        top.setCheckState(0, Qt.Checked if kids and n_on == len(kids)
                          else Qt.PartiallyChecked if n_on else Qt.Unchecked)

    def _apply_filter(self, *_a) -> None:
        """Hide features not matching every search word; hide families with
        no match; expand the families that match while searching."""
        words = [w.lower() for w in self.search.text().split()]
        self._syncing = True
        shown = total = 0
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            n = 0
            fam = (top.data(0, Qt.UserRole) or "").lower()
            for ch in self._children(top):
                low = ch.data(0, Qt.UserRole).lower() + " " + fam
                hit = not words or all(w in low for w in words)
                ch.setHidden(not hit)
                n += hit
            total += top.childCount()
            shown += n
            top.setHidden(n == 0)
            if words and n:
                top.setExpanded(True)
            self._refresh_family(top)
        self._syncing = False
        self._update_sel_label(shown, total)

    def _on_item_changed(self, it: QTreeWidgetItem, _col: int = 0) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if it.parent() is None:
                # family checkbox: (un)select its *shown* features
                on = it.checkState(0) != Qt.Unchecked
                for ch in self._children(it):
                    if ch.isHidden():
                        continue
                    ch.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
                    self._track(ch)
                self._refresh_family(it)
            else:
                self._track(it)
                self._refresh_family(it.parent())
        finally:
            self._syncing = False
        self._update_sel_label()

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

    def _load(self, cols: list[str], all_videos: bool):
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
            aucs = plot_distributions(self.figure, data, mask=mask, behaviour=beh)
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
            plot_joint(self.figure, data[cols[0]], data[cols[1]], cols[0], cols[1],
                       mask=mask, behaviour=self._behaviour() or "")
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
            df = load_feature_columns(video, self.config_path, cols)
            fps = self._video_fps(video)
            labels = self._labels(video)
            plot_over_time(self.figure, {c: df[c].to_numpy(np.float64) for c in df.columns},
                           fps, labels=labels)
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
        if self._mode == "time" and ev.dblclick and ev.inaxes is not None \
                and ev.xdata is not None:
            self.open_viewer(start_frame=int(round(ev.xdata * 60 * self._time_fps)))

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
                                           cols[:16], start_frame)
        return self._viewer


__all__ = ["FeatureVisualizationForm"]
