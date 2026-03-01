"""
진행 표시 위젯
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel, QPushButton
from PyQt6.QtCore import Qt


class ProgressWidget(QWidget):
    """OCR 진행 상황 표시 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ProgressWidget")
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        # 진행 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)

        # 상태 메시지
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet("color: #666666; font-size: 12px;")

        # 퍼센트 표시
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet("color: #333333; font-weight: bold; font-size: 12px;")
        self.percent_label.setFixedWidth(40)
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # 취소 버튼
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setFixedWidth(60)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.percent_label)
        layout.addWidget(self.cancel_btn)

    def update_progress(self, message: str, progress: float):
        """진행 상황 업데이트"""
        percent = int(progress * 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        self.percent_label.setText(f"{percent}%")

    def start(self):
        """진행 표시 시작"""
        self.progress_bar.setValue(0)
        self.status_label.setText("처리 시작...")
        self.percent_label.setText("0%")
        self.show()

    def stop(self):
        """진행 표시 종료"""
        self.hide()
        self.progress_bar.setValue(0)
