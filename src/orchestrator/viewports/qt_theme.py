"""
qt_theme.py
───────────
Unified dark theme cho PyQt6 app — VSCode/Fluent Design-inspired.

Apply qua `app.setStyleSheet(DARK_QSS)` 1 lần ở entry point. Đè qua per-widget
stylesheet được, nhưng nên giữ tối thiểu (vd colored Play/Stop buttons).

Palette:
  Background      #1e1e1e   (window)
  Surface         #252526   (group boxes, list backgrounds)
  Surface raised  #2d2d30   (hover, dialog)
  Border subtle   #3e3e42
  Text primary    #cccccc
  Text muted      #969696
  Text disabled   #6e6e6e
  Accent          #0078d4   (Fluent blue) — primary actions, focus, selection
  Accent hover    #1c8de8
  Success         #2da44e   (Play, ✓)
  Warning         #fb8500   (Run-on-Robot)
  Danger          #cf222e   (Stop, error)

Typography:
  System sans:    "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif
  Monospace:      "Cascadia Code", "JetBrains Mono", "Consolas", monospace
  Body 10pt, headings 11pt bold
"""
from __future__ import annotations

# Color constants — re-exportable cho widgets nào cần inline color (vd
# label background dùng status level).
BG          = "#1e1e1e"
SURFACE     = "#252526"
SURFACE_RAISED = "#2d2d30"
BORDER      = "#3e3e42"
BORDER_FOCUS = "#0078d4"
TEXT        = "#cccccc"
TEXT_MUTED  = "#969696"
TEXT_DISABLED = "#6e6e6e"
ACCENT      = "#0078d4"
ACCENT_HOVER = "#1c8de8"
ACCENT_PRESSED = "#005a9e"
SUCCESS     = "#2da44e"
SUCCESS_HOVER = "#2c974b"
WARNING     = "#fb8500"
WARNING_HOVER = "#d96e00"
DANGER      = "#cf222e"
DANGER_HOVER = "#a40e26"
INFO        = "#38b6e7"

# Status badge colors (cho status bar)
STATUS_OK    = SUCCESS
STATUS_WARN  = WARNING
STATUS_ERR   = DANGER
STATUS_INFO  = INFO


DARK_QSS = f"""
/* ════════════════════════════════════════════════════════════════════════
   GLOBAL — base background + typography
   ════════════════════════════════════════════════════════════════════════ */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
    selection-background-color: {ACCENT};
    selection-color: white;
}}

QMainWindow {{
    background-color: {BG};
}}

QDialog {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QToolTip {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 8px;
}}

/* ════════════════════════════════════════════════════════════════════════
   MENU BAR + MENU
   ════════════════════════════════════════════════════════════════════════ */
QMenuBar {{
    background-color: {BG};
    color: {TEXT};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 2px;
    spacing: 2px;
}}
QMenuBar::item {{
    padding: 6px 12px;
    border-radius: 4px;
    background-color: transparent;
}}
QMenuBar::item:selected {{
    background-color: {SURFACE_RAISED};
}}
QMenuBar::item:pressed {{
    background-color: {ACCENT};
    color: white;
}}

QMenu {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    /* Native indicator column reserve cùng width cho mọi item (checkable
       hoặc plain) → text căn lề pixel-perfect. Padding-right 24 cho shortcut. */
    padding: 6px 24px 6px 8px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QMenu::item:disabled {{
    color: {TEXT_DISABLED};
}}
QMenu::indicator {{
    /* Fixed width column for ✓ — Qt reserve cùng width cho all items. */
    width: 14px;
    height: 14px;
    background-color: transparent;
    border: none;
    margin-left: 6px;
}}
QMenu::indicator:checked {{
    image: url({{CHECK_ICON_PATH}});
}}
QMenu::indicator:unchecked {{
    /* Empty — transparent. Column vẫn reserve để text căn lề. */
    image: none;
}}
QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 4px 8px;
}}

/* ════════════════════════════════════════════════════════════════════════
   BUTTONS — default + variant via [class] property (objectName="primary",
   "success", "danger", "warning"). Inline setStyleSheet vẫn override được.
   ════════════════════════════════════════════════════════════════════════ */
QPushButton {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 18px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #3a3a3d;
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
    color: white;
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    border-color: {BORDER};
    background-color: {SURFACE};
}}
QPushButton:checked {{
    background-color: {ACCENT};
    color: white;
    border-color: {ACCENT};
}}
QPushButton:focus {{
    border-color: {ACCENT};
}}

/* Primary action variant (set via objectName="primaryBtn") */
QPushButton#primaryBtn {{
    background-color: {ACCENT};
    color: white;
    border-color: {ACCENT};
    font-weight: bold;
}}
QPushButton#primaryBtn:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primaryBtn:pressed {{ background-color: {ACCENT_PRESSED}; }}

/* ════════════════════════════════════════════════════════════════════════
   GROUP BOX — modern card style
   ════════════════════════════════════════════════════════════════════════ */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    color: {ACCENT_HOVER};
    background-color: {BG};
}}

/* ════════════════════════════════════════════════════════════════════════
   TAB WIDGET — modern tabs
   ════════════════════════════════════════════════════════════════════════ */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {SURFACE};
    top: -1px;
}}
QTabBar {{
    background-color: transparent;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_MUTED};
    padding: 6px 16px;
    border: 1px solid transparent;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background-color: {SURFACE_RAISED};
}}
QTabBar::tab:selected {{
    color: white;
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* ════════════════════════════════════════════════════════════════════════
   INPUT FIELDS — LineEdit, SpinBox, DoubleSpinBox, ComboBox
   ════════════════════════════════════════════════════════════════════════ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {TEXT_DISABLED};
    background-color: {SURFACE};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    background-color: {SURFACE_RAISED};
    border: none;
    border-top-right-radius: 4px;
    width: 16px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {SURFACE_RAISED};
    border: none;
    border-bottom-right-radius: 4px;
    width: 16px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {ACCENT};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    /* Custom up/down-button làm Fusion ngừng vẽ arrow → cấp image riêng. */
    image: url({{SPIN_UP_ICON_PATH}});
    width: 8px; height: 8px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({{SPIN_DOWN_ICON_PATH}});
    width: 8px; height: 8px;
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
    background-color: transparent;
}}
QComboBox::down-arrow {{
    /* Styling drop-down sub-control làm Fusion ngừng vẽ arrow mặc định →
       phải cấp arrow image riêng (▼ glyph generated runtime). */
    image: url({{COMBO_ARROW_ICON_PATH}});
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: white;
    border-radius: 4px;
    padding: 2px;
}}

/* ════════════════════════════════════════════════════════════════════════
   LIST + TREE
   ════════════════════════════════════════════════════════════════════════ */
QListWidget, QTreeWidget {{
    background-color: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 2px;
    alternate-background-color: {SURFACE};
}}
QListWidget::item, QTreeWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {SURFACE_RAISED};
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}

/* ════════════════════════════════════════════════════════════════════════
   CHECKBOX + RADIO
   ════════════════════════════════════════════════════════════════════════ */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {TEXT};
    padding: 2px;
    /* Transparent → hòa vào nền container (groupbox SURFACE) thay vì hiện
       ô tối #1e1e1e của QWidget base rule phía sau chữ. */
    background-color: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
}}
QCheckBox::indicator:unchecked {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 3px;
    image: none;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QRadioButton::indicator:unchecked {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border: 4px solid {BG};
    border-radius: 8px;
    width: 8px; height: 8px;
}}

/* ════════════════════════════════════════════════════════════════════════
   DOCK WIDGET — modern title bar
   ════════════════════════════════════════════════════════════════════════ */
QDockWidget {{
    color: {TEXT};
    /* Custom icons cho float (□) + close (✕) — generated runtime, paths
       được substitute trong apply_dark_theme(). */
    titlebar-normal-icon: url({{DOCK_FLOAT_ICON_PATH}});
    titlebar-close-icon:  url({{DOCK_CLOSE_ICON_PATH}});
}}
QDockWidget::title {{
    background-color: {SURFACE};
    color: {TEXT};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    text-align: left;
    font-weight: 600;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background-color: transparent;
    border: none;
    border-radius: 3px;
    padding: 2px;
}}
QDockWidget::close-button:hover {{
    background-color: {DANGER};
}}
QDockWidget::float-button:hover {{
    background-color: {ACCENT};
}}

/* ════════════════════════════════════════════════════════════════════════
   STATUS BAR + LABELS
   ════════════════════════════════════════════════════════════════════════ */
QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{ border: none; }}

QLabel {{ color: {TEXT}; background-color: transparent; }}

/* ════════════════════════════════════════════════════════════════════════
   TOOLBAR
   ════════════════════════════════════════════════════════════════════════ */
QToolBar {{
    background-color: {BG};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px;
}}
QToolBar::separator {{
    background-color: {BORDER};
    width: 1px;
    margin: 4px 6px;
}}

/* ════════════════════════════════════════════════════════════════════════
   SCROLLBAR — thin modern
   ════════════════════════════════════════════════════════════════════════ */
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: #4a4a4a;
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{ background-color: #6a6a6a; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none; height: 0; border: none;
}}
QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background-color: #4a4a4a;
    min-width: 24px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{ background-color: #6a6a6a; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: none; width: 0; border: none;
}}

/* ════════════════════════════════════════════════════════════════════════
   SLIDER (cho joint sliders, gripper, sim speed)
   ════════════════════════════════════════════════════════════════════════ */
QSlider::groove:horizontal {{
    background: {SURFACE_RAISED};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid {BG};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::groove:vertical {{
    background: {SURFACE_RAISED};
    width: 4px;
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background: {ACCENT};
    width: 14px; height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
    border: 2px solid {BG};
}}

/* ════════════════════════════════════════════════════════════════════════
   PROGRESS BAR (cho potential use)
   ════════════════════════════════════════════════════════════════════════ */
QProgressBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ════════════════════════════════════════════════════════════════════════
   QDial — sử dụng cho rotary jog encoder
   ════════════════════════════════════════════════════════════════════════ */
QDial {{
    background-color: {SURFACE_RAISED};
}}
"""


def _paint_glyph_png(glyph: str, color: str = TEXT,
                       size: int = 14, font_size: int = 10,
                       suffix: str = "_glyph.png") -> str:
    """Generic helper — paint a Unicode glyph vào QPixmap + save temp PNG.

    Dùng cho QSS `image: url(...)` references (check mark, dock float/close
    buttons, etc).
    """
    import tempfile
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap

    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(color))
    p.setFont(QFont("Segoe UI Symbol", font_size, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    pix.save(tmp.name)
    return tmp.name.replace("\\", "/")              # QSS url() cần forward slash


def _generate_check_png() -> str:
    """Check mark cho QMenu indicator (toggle items)."""
    return _paint_glyph_png("✓", TEXT, size=14, font_size=10,
                              suffix="_check.png")


def _generate_dock_close_png() -> str:
    """Close (✕) icon cho QDockWidget title bar."""
    return _paint_glyph_png("✕", TEXT_MUTED, size=14, font_size=10,
                              suffix="_dock_close.png")


def _generate_dock_float_png() -> str:
    """Float/dock toggle icon (□) cho QDockWidget title bar."""
    return _paint_glyph_png("□", TEXT_MUTED, size=14, font_size=11,
                              suffix="_dock_float.png")


def _generate_combo_arrow_png() -> str:
    """Down-arrow (▼) icon cho QComboBox — Fusion ngừng vẽ arrow khi
    drop-down sub-control bị custom styling."""
    return _paint_glyph_png("▼", TEXT_MUTED, size=12, font_size=8,
                              suffix="_combo_arrow.png")


def _generate_spin_up_png() -> str:
    """Up-arrow (▲) icon cho QSpinBox/QDoubleSpinBox up-button."""
    return _paint_glyph_png("▲", TEXT, size=10, font_size=7,
                              suffix="_spin_up.png")


def _generate_spin_down_png() -> str:
    """Down-arrow (▼) icon cho QSpinBox/QDoubleSpinBox down-button."""
    return _paint_glyph_png("▼", TEXT, size=10, font_size=7,
                              suffix="_spin_down.png")


def make_check_icon():
    """Tạo QIcon với 2 state pixmaps cho menu toggle action — Word-style:
      - State.On  → check mark glyph (✓) painted với theme TEXT color
      - State.Off → transparent pixmap cùng kích thước (preserve indicator
                    column width nhưng vô hình)

    Cách dùng:
        act = QAction("Fullscreen", self)
        act.setCheckable(True)
        act.setIcon(make_check_icon())
        act.setChecked(initial)
        # Qt auto-switches On/Off pixmap khi setChecked() được gọi → text
        # KHÔNG bị shift, check chỉ hiện ở cột margin trái như Word/VSCode.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

    # Trick: pixmap rộng (matches Fusion default icon column ~16px) NHƯNG
    # paint ✓ ở **mép phải** của pixmap → visually ✓ sát text, dù Qt reserve
    # cả icon column. Fusion không cho phép shrink icon column dễ dàng nên
    # ta render ✓ "đẩy" về bên phải bên trong column.
    size = 16
    # On (checked) — paint ✓ ALIGN RIGHT trong pixmap
    on_pix = QPixmap(size, size)
    on_pix.fill(QColor(0, 0, 0, 0))                       # transparent base
    p = QPainter(on_pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(TEXT))
    f = QFont("Segoe UI Symbol", 11, QFont.Weight.Bold)
    p.setFont(f)
    # AlignRight → ✓ ở mép phải pixmap, giảm gap visual tới text
    p.drawText(on_pix.rect(),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                "✓")
    p.end()

    # Off (unchecked) — fully transparent, cùng size để Qt reserve column
    off_pix = QPixmap(size, size)
    off_pix.fill(QColor(0, 0, 0, 0))

    icon = QIcon()
    icon.addPixmap(on_pix, QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(off_pix, QIcon.Mode.Normal, QIcon.State.Off)
    return icon


class TightMenuStyle:
    """QProxyStyle override — force compact menu icon column.

    Fusion style hard-code `PM_SmallIconSize=16` cho menu item icon column +
    thêm padding mặc định ~8-12px around. Tổng gap icon-to-text = ~25-30px,
    QSS `icon-size` không override được PM_SmallIconSize.

    Override `pixelMetric()` để return giá trị nhỏ cho:
      - PM_SmallIconSize       (icon column width)
      - PM_ButtonIconSize      (button icon area)

    Áp dụng global qua `app.setStyle(TightMenuStyle(app.style()))`.
    """
    # Defined lazy trong apply_dark_theme để tránh import QProxyStyle ở
    # module-level (test env có thể không có Qt installed).


def apply_dark_theme(app) -> None:
    """Apply dark QSS toàn cục cho QApplication.

    Gọi 1 lần ở entry point sau khi tạo QApplication. Per-widget stylesheets
    đặt sau sẽ override (vd Play button color, status label color).
    """
    from PyQt6.QtGui import QPalette, QColor

    app.setStyle("Fusion")              # Base style cho cross-platform consistent
    # Pre-set palette để widgets không-QSS (vd dialogs popup) hợp tone.
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT_HOVER))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(palette)
    # Generate icons PNG runtime để QSS reference qua image: url(...):
    #   - check.png      : QMenu indicator (toggle items)
    #   - dock_close.png : QDockWidget close button (✕)
    #   - dock_float.png : QDockWidget float/dock toggle button (□)
    qss = (DARK_QSS
           .replace("{CHECK_ICON_PATH}",       _generate_check_png())
           .replace("{DOCK_CLOSE_ICON_PATH}",  _generate_dock_close_png())
           .replace("{DOCK_FLOAT_ICON_PATH}",  _generate_dock_float_png())
           .replace("{COMBO_ARROW_ICON_PATH}", _generate_combo_arrow_png())
           .replace("{SPIN_UP_ICON_PATH}",     _generate_spin_up_png())
           .replace("{SPIN_DOWN_ICON_PATH}",   _generate_spin_down_png()))
    app.setStyleSheet(qss)
