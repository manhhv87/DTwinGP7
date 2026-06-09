"""
qt_widgets.py
─────────────
Custom Qt widgets shared across viewports — collapsible section + worker thread
signal bridge.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Section with a header button that expands/collapses content. Default expanded=True or False.

    Qt has no built-in collapsible widget (QGroupBox.checkable only disables
    children, does not hide them). Custom: QPushButton header (text-align left,
    ▼/▶ arrow) + QWidget content that can be toggled visible.
    """

    def __init__(self, title: str, expanded: bool = True,
                  parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # A QFrame draws a stylesheet border RELIABLY (a plain QWidget subclass
        # does not, even with WA_StyledBackground in some setups). This frame
        # wraps BOTH the header bar and the content body → one box around the
        # whole section, so the content shown to the user is clearly framed.
        self._frame = QFrame()
        self._frame.setObjectName("sectionFrame")
        self._frame.setStyleSheet(
            "QFrame#sectionFrame {"
            "  border: 1px solid #3e3e42; border-radius: 4px;"
            "}"
        )
        fl = QVBoxLayout(self._frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)
        outer.addWidget(self._frame)

        self._toggle_btn = QPushButton()
        # Header = the card's title bar: rounded top corners to match the frame,
        # a bottom separator line dividing it from the content body below.
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 7px 12px; "
            "  background-color: #2d2d30; color: #cccccc; "
            "  border: none; border-bottom: 1px solid #3e3e42; "
            "  border-top-left-radius: 4px; border-top-right-radius: 4px; "
            "  font-weight: 600;"
            "}"
            "QPushButton:hover { background-color: #3a3a3d; }"
        )
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_text()
        self._toggle_btn.clicked.connect(self._toggle)
        fl.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 6)
        self._content_layout.setSpacing(4)
        self._content.setVisible(expanded)
        fl.addWidget(self._content)

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

    Worker thread emits these signals; Qt's QObject machinery automatically
    queues them to the main thread → slot runs safely on the GUI thread.
    """

    joints_update = pyqtSignal(list)        # joints_deg
    status        = pyqtSignal(str, str)    # message, level (info/ok/warn/err)
    gripper       = pyqtSignal(bool)        # close
    program_done  = pyqtSignal()
    prog_step     = pyqtSignal(int)         # highlight currently-executing step (row index)
    prog_show_job = pyqtSignal(str)         # switch program list to show this job (CALL JOB)
    camera_result = pyqtSignal(object)      # dict {rgb, depth, objects, fps, source}
    sim_reset     = pyqtSignal()            # SimEvent 'reset_objects' → objects return to initial positions
    exp_progress  = pyqtSignal(int, object)  # (trials completed, stats dict) — Digital Twin experiment
    exp_done      = pyqtSignal(object)       # stats dict — mirror/experiment finished → re-enable UI
    live_jog_off  = pyqtSignal(str)          # Phase-2 live jog worker exited abnormally → uncheck toggle + show msg
    run_overwrite_blocked = pyqtSignal(str, str)  # (job_name, message) — Run-on-Robot upload blocked (can't overwrite) → offer rename
