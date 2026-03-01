"""
페이지 썸네일 사이드바
"""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize


class ThumbnailPanel(QWidget):
    """페이지 썸네일 목록"""

    page_selected = pyqtSignal(int)  # 페이지 번호 (0-indexed)

    THUMBNAIL_WIDTH = 120
    THUMBNAIL_HEIGHT = 170

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ThumbnailPanel")
        self.setFixedWidth(150)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)

        # 제목
        title = QLabel("페이지")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #333;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 썸네일 목록
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(self.THUMBNAIL_WIDTH, self.THUMBNAIL_HEIGHT))
        self.list_widget.setSpacing(4)
        self.list_widget.setViewMode(QListWidget.ViewMode.ListMode)
        self.list_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.list_widget.currentRowChanged.connect(self._on_row_changed)

        layout.addWidget(self.list_widget)

    def set_pages(self, page_images: list[np.ndarray]):
        """페이지 이미지들로 썸네일 설정"""
        self.list_widget.clear()

        for i, img in enumerate(page_images):
            pixmap = self._array_to_pixmap(img)
            scaled = pixmap.scaled(
                self.THUMBNAIL_WIDTH,
                self.THUMBNAIL_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            item = QListWidgetItem()
            item.setIcon(QIcon(scaled))
            item.setText(f"  {i + 1}페이지")
            item.setSizeHint(QSize(self.THUMBNAIL_WIDTH + 20, self.THUMBNAIL_HEIGHT + 30))
            self.list_widget.addItem(item)

        # 첫 페이지 선택
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _array_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """numpy 배열을 QPixmap으로 변환"""
        h, w = img.shape[:2]

        if len(img.shape) == 3:
            bytes_per_line = 3 * w
            qimage = QImage(
                img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            bytes_per_line = w
            qimage = QImage(
                img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            )

        return QPixmap.fromImage(qimage)

    def _on_row_changed(self, row: int):
        """선택된 행 변경"""
        if row >= 0:
            self.page_selected.emit(row)

    def select_page(self, index: int):
        """프로그래밍적으로 페이지 선택"""
        if 0 <= index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)

    def clear_pages(self):
        """모든 썸네일 제거"""
        self.list_widget.clear()
