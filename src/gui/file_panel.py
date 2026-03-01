"""
파일 열기 / 드래그앤드롭 패널
"""

from pathlib import Path

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from .resources import SUPPORTED_EXTENSIONS


class DropZone(QWidget):
    """파일 드래그앤드롭 영역 (파일이 없을 때 표시)"""

    file_dropped = pyqtSignal(str)  # 파일 경로

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet("""
            DropZone {
                background-color: #fafafa;
                border: 2px dashed #ccc;
                border-radius: 12px;
                min-height: 300px;
            }
            DropZone:hover {
                border-color: #1a73e8;
                background-color: #f0f7ff;
            }
        """)

        # 아이콘 대용 텍스트
        icon_label = QLabel("📄")
        icon_label.setStyleSheet("font-size: 48px; border: none;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # 안내 텍스트
        text_label = QLabel("파일을 여기에 끌어놓으세요")
        text_label.setStyleSheet(
            "font-size: 16px; color: #666; font-weight: bold; border: none;"
        )
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)

        sub_label = QLabel("PDF, PNG, JPG, BMP, TIFF 지원")
        sub_label.setStyleSheet("font-size: 12px; color: #999; border: none;")
        sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        DropZone {
                            background-color: #e8f0fe;
                            border: 2px dashed #1a73e8;
                            border-radius: 12px;
                            min-height: 300px;
                        }
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._setup_ui()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS:
                self.file_dropped.emit(file_path)
        self._setup_ui()
