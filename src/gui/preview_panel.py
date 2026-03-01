"""
문서 미리보기 패널 - 영역 오버레이 표시
"""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QWidget, QVBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QWheelEvent
from PyQt6.QtCore import Qt, pyqtSignal, QRectF

from .resources import REGION_COLORS, REGION_BORDER_COLORS


class PreviewPanel(QWidget):
    """문서 미리보기 + 영역 오버레이"""

    region_clicked = pyqtSignal(int)  # 클릭된 영역 인덱스

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewPanel")

        self._current_image: Optional[QPixmap] = None
        self._region_items: list[QGraphicsRectItem] = []
        self._regions_data: list[dict] = []
        self._show_overlay = True

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 그래픽스 뷰/씬
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor("#e8e8e8")))

        layout.addWidget(self.view)

    def set_image(self, image: np.ndarray):
        """페이지 이미지 설정"""
        self.scene.clear()
        self._region_items.clear()

        pixmap = self._array_to_pixmap(image)
        self._current_image = pixmap

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        # 뷰에 맞게 표시
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_regions(self, regions: list[dict]):
        """영역 오버레이 설정

        regions: [{"bbox": (x, y, x2, y2), "type": "text"|"math"|...}, ...]
        """
        # 기존 오버레이 제거
        for item in self._region_items:
            self.scene.removeItem(item)
        self._region_items.clear()
        self._regions_data = regions

        if not self._show_overlay:
            return

        for i, region in enumerate(regions):
            bbox = region.get("bbox", (0, 0, 0, 0))
            region_type = region.get("type", "unknown")

            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y

            # 오버레이 사각형
            fill_color = REGION_COLORS.get(region_type, REGION_COLORS["unknown"])
            border_color = REGION_BORDER_COLORS.get(region_type, REGION_BORDER_COLORS["unknown"])

            rect_item = ClickableRectItem(i, QRectF(x, y, w, h))
            rect_item.setBrush(QBrush(QColor(*fill_color)))
            rect_item.setPen(QPen(QColor(*border_color), 2))
            rect_item.clicked.connect(self._on_region_clicked)

            self.scene.addItem(rect_item)
            self._region_items.append(rect_item)

    def toggle_overlay(self, show: bool):
        """오버레이 표시/숨김"""
        self._show_overlay = show
        for item in self._region_items:
            item.setVisible(show)

    def clear_image(self):
        """이미지 초기화"""
        self.scene.clear()
        self._region_items.clear()
        self._regions_data.clear()
        self._current_image = None

    def _array_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """numpy 배열을 QPixmap으로 변환"""
        h, w = img.shape[:2]

        if len(img.shape) == 3:
            bytes_per_line = 3 * w
            qimage = QImage(
                img.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            bytes_per_line = w
            qimage = QImage(
                img.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            )

        return QPixmap.fromImage(qimage)

    def _on_region_clicked(self, index: int):
        """영역 클릭 처리"""
        self.region_clicked.emit(index)

        # 클릭된 영역 하이라이트
        for i, item in enumerate(self._region_items):
            if i == index:
                item.setPen(QPen(QColor(255, 255, 0, 255), 3))
            else:
                region_type = self._regions_data[i].get("type", "unknown")
                border_color = REGION_BORDER_COLORS.get(region_type, REGION_BORDER_COLORS["unknown"])
                item.setPen(QPen(QColor(*border_color), 2))


class ZoomableGraphicsView(QGraphicsView):
    """마우스 휠 줌 지원 그래픽스 뷰"""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Ctrl+마우스 휠로 줌"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                factor = 1.15
            else:
                factor = 1 / 1.15

            self._zoom_factor *= factor
            # 줌 제한
            if 0.1 < self._zoom_factor < 10.0:
                self.scale(factor, factor)
            else:
                self._zoom_factor /= factor
        else:
            super().wheelEvent(event)


class ClickableRectItem(QGraphicsRectItem):
    """클릭 가능한 사각형 아이템"""

    class _SignalHelper:
        """pyqtSignal을 QGraphicsItem에서 사용하기 위한 헬퍼"""
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for cb in self._callbacks:
                cb(*args)

    def __init__(self, index: int, rect: QRectF, parent=None):
        super().__init__(rect, parent)
        self.index = index
        self.clicked = self._SignalHelper()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)
        super().mousePressEvent(event)
