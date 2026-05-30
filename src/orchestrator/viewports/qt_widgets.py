"""
qt_widgets.py
─────────────
Custom Qt widgets dùng chung cho viewport — collapsible section + worker thread
signal bridge.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Section có nút header xổ/gập content. Default expanded=True or False.

    Qt KHÔNG có collapsible widget built-in (QGroupBox.checkable chỉ disable
    children, không hide). Custom: QPushButton header (text-align left,
    ▼/▶ arrow) + QWidget content có thể toggle visible.
    """

    def __init__(self, title: str, expanded: bool = True,
                  parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QPushButton()
        # Custom collapsible header — flat, full-width, accent border-bottom
        # for active visual cue. Uses theme palette colors qua hard-coded
        # values (avoid circular import of qt_theme constants).
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 7px 12px; "
            "  background-color: #2d2d30; color: #cccccc; "
            "  border: 1px solid #3e3e42; border-radius: 4px; "
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background-color: #3a3a3d; border-color: #0078d4;"
            "}"
        )
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_text()
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 6)
        self._content_layout.setSpacing(4)
        self._content.setVisible(expanded)
        outer.addWidget(self._content)

    def _update_text(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._title}")

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._update_text()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, w: QWidget) -> None:
        self._content_layout.addWidget(w)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)


class WorkerSignals(QObject):
    """Bridge worker thread → main thread (PyQt6 thread-safe pattern).

    Worker thread emit các signal này; QObject machinery của Qt tự queue về
    main thread → slot chạy an toàn trên GUI thread.
    """

    joints_update = pyqtSignal(list)        # joints_deg
    status        = pyqtSignal(str, str)    # message, level (info/ok/warn/err)
    gripper       = pyqtSignal(bool)        # close
    program_done  = pyqtSignal()
    camera_result = pyqtSignal(object)      # dict {rgb, depth, objects, fps, source}
    sim_reset     = pyqtSignal()            # SimEvent 'reset_objects' → objects về vị trí ban đầu
    exp_progress  = pyqtSignal(int, object)  # (trial đã xong, stats dict) — Digital Twin experiment
    exp_done      = pyqtSignal(object)       # stats dict — mirror/experiment kết thúc → re-enable UI
