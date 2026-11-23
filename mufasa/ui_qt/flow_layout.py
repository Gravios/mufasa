"""A layout that wraps its widgets onto further rows when the width runs out (122ls).

Qt's own row layouts ask for the sum of their widgets' widths, so a long tool row makes a window
wider than the screen. This one asks only for its widest widget and grows downward instead.
"""
from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget

__all__ = ["FlowLayout"]


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, h_spacing: int = 6, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h, self._v = h_spacing, v_spacing
        self._width = 0                       # the width last laid out in: the rows it needs
        self._pending = False
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout -------------------------------------------------------------------------
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        h = self._do_layout(rect, test=False)
        if rect.width() != self._width:
            old = self.heightForWidth(self._width) if self._width else None
            self._width = rect.width()
            if old != h and not self._pending:     # more or fewer rows: ask the parent for the room
                self._pending = True
                QTimer.singleShot(0, self._relayout)

    def _relayout(self) -> None:
        self._pending = False
        self.invalidate()
        if self.parentWidget() is not None:
            self.parentWidget().updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: N802
        w = h = 0
        for it in self._visible():
            s = it.sizeHint()
            w += s.width() + self._h
            h = max(h, s.height())
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802
        s = QSize()
        for it in self._visible():
            s = s.expandedTo(it.minimumSize())
        m: QMargins = self.contentsMargins()
        s = s + QSize(m.left() + m.right(), m.top() + m.bottom())
        if self._width:                            # as many rows as the current width takes
            s.setHeight(max(s.height(), self.heightForWidth(self._width)))
        return s

    # -- conveniences like QBoxLayout's --------------------------------------------------
    def addWidget(self, w: QWidget, *_stretch) -> None:  # noqa: N802 — a stretch is ignored
        super().addWidget(w)

    def addSpacing(self, width: int) -> None:  # noqa: N802
        spacer = QWidget()
        spacer.setFixedSize(max(width - self._h, 0), 1)
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.addWidget(spacer)

    def addStretch(self, *_a) -> None:  # noqa: N802 — rows wrap instead of stretching
        pass

    # -- the wrapping --------------------------------------------------------------------
    def _visible(self) -> list[QLayoutItem]:
        return [it for it in self._items if not (it.widget() and it.widget().isHidden())]

    def _do_layout(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = r.x(), r.y(), 0
        for it in self._visible():
            s = it.sizeHint()
            if x > r.x() and x + s.width() > r.right() + 1:
                x, y, row_h = r.x(), y + row_h + self._v, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), s))
            x += s.width() + self._h
            row_h = max(row_h, s.height())
        return y + row_h - rect.y() + m.bottom()
