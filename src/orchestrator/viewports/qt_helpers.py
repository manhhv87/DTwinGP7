"""
qt_helpers.py
─────────────
Utility functions for Qt/VTK viewport — hand-drawn icons (QPainter) and
numpy ↔ vtkMatrix4x4 converter. Split from gp7_app_qt.py to keep the main
file lean and allow reuse by other Qt modules in the project.
"""
from __future__ import annotations

import numpy as np
import vtk
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def numpy_to_vtk_matrix(T: np.ndarray) -> vtk.vtkMatrix4x4:
    """numpy 4x4 → vtkMatrix4x4 (for actor.SetUserMatrix).

    Fast path: DeepCopy with a flat sequence (1 C-level memcpy) instead of a
    nested 16-call SetElement Python loop. Falls back to unrolled SetElement
    if DeepCopy does not accept the input type.
    """
    m = vtk.vtkMatrix4x4()
    try:
        m.DeepCopy(T.ravel().astype(np.float64, copy=False).tolist())
    except Exception:                                     # noqa: BLE001
        e = m.SetElement
        e(0,0,T[0,0]); e(0,1,T[0,1]); e(0,2,T[0,2]); e(0,3,T[0,3])
        e(1,0,T[1,0]); e(1,1,T[1,1]); e(1,2,T[1,2]); e(1,3,T[1,3])
        e(2,0,T[2,0]); e(2,1,T[2,1]); e(2,2,T[2,2]); e(2,3,T[2,3])
        e(3,0,T[3,0]); e(3,1,T[3,1]); e(3,2,T[3,2]); e(3,3,T[3,3])
    return m


def draw_copy_icon(color: str = "#cccccc", mask: str = "#2d2d30") -> QIcon:
    """Copy icon — 2 overlapping sheets of paper.

    Glyph ⎘ (U+2398) does not render in Segoe UI → draw shape instead.
    """
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.2)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(6, 2, 7, 9, 2, 2)
    p.setBrush(QColor(mask))
    p.drawRoundedRect(3, 5, 7, 9, 2, 2)
    p.end()
    return QIcon(pix)


def draw_paste_icon(color: str = "#cccccc") -> QIcon:
    """Paste icon — clipboard (body + top clip).

    Glyph 📋 (U+1F4CB) renders as emoji/empty box → draw shape instead.
    """
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.2)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(3, 3, 10, 11, 2, 2)
    p.setBrush(QColor(color))
    p.drawRoundedRect(6, 1, 4, 3, 1, 1)
    p.end()
    return QIcon(pix)


def draw_menu_icon(color: str = "#cccccc") -> QIcon:
    """Hamburger menu (≡) — 3 horizontal lines."""
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.4)
    p.setPen(pen)
    for y in (4, 8, 12):
        p.drawLine(3, y, 13, y)
    p.end()
    return QIcon(pix)


def draw_plus_icon(color: str = "#cccccc") -> QIcon:
    """Plus sign (Add) — 2 crossing strokes. Glyph '+' renders fine but drawn
    manually to stay consistent with other job icons (rename/trash don't render glyphs)."""
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(8, 3, 8, 13)
    p.drawLine(3, 8, 13, 8)
    p.end()
    return QIcon(pix)


def draw_rename_icon(color: str = "#cccccc") -> QIcon:
    """Pencil (Rename/Edit) — diagonal parallelogram body + graphite tip.

    Glyph ⟲ (U+27F2) does not render in Segoe UI → draw shape instead.
    """
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.3)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    # Pencil body (parallelogram): eraser end top-right → tip base bottom-left
    p.drawLine(9, 3, 13, 7)     # top edge (eraser cap)
    p.drawLine(9, 3, 3, 9)      # left edge of body
    p.drawLine(13, 7, 7, 13)    # right edge of body
    p.drawLine(3, 9, 7, 13)     # bottom edge (tip base)
    # Graphite tip converging to a single point at bottom-left
    p.drawLine(3, 9, 2, 14)
    p.drawLine(7, 13, 2, 14)
    p.end()
    return QIcon(pix)


def draw_trash_icon(color: str = "#cccccc") -> QIcon:
    """Trash can (Delete) — lid + handle + slightly tapered body + 3 vertical ribs.

    Glyph ✕ (U+2715) does not render in Segoe UI → draw shape instead.
    """
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.3)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    # Lid
    p.drawLine(3, 5, 13, 5)
    # Handle
    p.drawLine(6, 5, 6, 3); p.drawLine(6, 3, 10, 3); p.drawLine(10, 3, 10, 5)
    # Body (slightly tapered inward)
    p.drawLine(4, 5, 5, 13)
    p.drawLine(12, 5, 11, 13)
    p.drawLine(5, 13, 11, 13)
    # 3 vertical ribs
    p.drawLine(6, 7, 6, 11)
    p.drawLine(8, 7, 8, 11)
    p.drawLine(10, 7, 10, 11)
    p.end()
    return QIcon(pix)


def draw_arrow_up_icon(color: str = "#cccccc") -> QIcon:
    """Arrow up (Move up) — chevron + shaft. Hand-drawn for consistency and reliable rendering."""
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(4, 9, 8, 5)      # left chevron
    p.drawLine(8, 5, 12, 9)     # right chevron
    p.drawLine(8, 5, 8, 12)     # shaft
    p.end()
    return QIcon(pix)


def draw_arrow_down_icon(color: str = "#cccccc") -> QIcon:
    """Arrow down (Move down) — chevron + shaft."""
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(4, 7, 8, 11)     # left chevron
    p.drawLine(8, 11, 12, 7)    # right chevron
    p.drawLine(8, 4, 8, 11)     # shaft
    p.end()
    return QIcon(pix)


def draw_x_icon(color: str = "#cccccc") -> QIcon:
    """X mark (Delete/close) — 2 diagonal strokes. Glyph ✕ (U+2715) does not render in Segoe UI."""
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(4, 4, 12, 12)
    p.drawLine(12, 4, 4, 12)
    p.end()
    return QIcon(pix)
