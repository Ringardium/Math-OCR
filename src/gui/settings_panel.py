"""
설정 패널
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QComboBox, QLabel, QFormLayout
)
from PyQt6.QtCore import pyqtSignal

from ..config import AppConfig, OCRConfig, ExportConfig, ExportFormat, TextType


class SettingsPanel(QWidget):
    """OCR 및 내보내기 설정 패널"""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPanel")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # OCR 설정 그룹
        ocr_group = QGroupBox("OCR 설정")
        ocr_layout = QFormLayout()

        # 텍스트 유형
        self.text_type_combo = QComboBox()
        self.text_type_combo.addItem("인쇄체 + 손글씨", TextType.BOTH)
        self.text_type_combo.addItem("인쇄체만", TextType.PRINTED)
        self.text_type_combo.addItem("손글씨만", TextType.HANDWRITTEN)
        self.text_type_combo.currentIndexChanged.connect(self._on_changed)
        ocr_layout.addRow("텍스트 유형:", self.text_type_combo)

        # 언어 설정
        self.lang_ko = QCheckBox("한국어")
        self.lang_ko.setChecked(True)
        self.lang_en = QCheckBox("영어")
        self.lang_en.setChecked(True)
        self.lang_ja = QCheckBox("일본어")

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(self.lang_ko)
        lang_layout.addWidget(self.lang_en)
        lang_layout.addWidget(self.lang_ja)
        lang_layout.addStretch()
        ocr_layout.addRow("언어:", lang_layout)

        # 수식 OCR
        self.math_ocr_check = QCheckBox("수식 인식 (Texify)")
        self.math_ocr_check.setChecked(True)
        self.math_ocr_check.stateChanged.connect(self._on_changed)
        ocr_layout.addRow("", self.math_ocr_check)

        # 표 인식
        self.table_ocr_check = QCheckBox("표 인식")
        self.table_ocr_check.setChecked(True)
        self.table_ocr_check.stateChanged.connect(self._on_changed)
        ocr_layout.addRow("", self.table_ocr_check)

        # GPU
        self.gpu_check = QCheckBox("GPU 사용")
        self.gpu_check.setChecked(True)
        self.gpu_check.stateChanged.connect(self._on_changed)
        ocr_layout.addRow("", self.gpu_check)

        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)

        # 내보내기 설정 그룹
        export_group = QGroupBox("내보내기 설정")
        export_layout = QFormLayout()

        # 출력 형식
        self.format_combo = QComboBox()
        self.format_combo.addItem("Word (.docx)", ExportFormat.DOCX)
        self.format_combo.addItem("HWP (.hwp)", ExportFormat.HWP)
        self.format_combo.currentIndexChanged.connect(self._on_changed)
        export_layout.addRow("출력 형식:", self.format_combo)

        # 레이아웃 적용
        self.layout_check = QCheckBox("감지된 레이아웃 적용")
        self.layout_check.setChecked(True)
        self.layout_check.setToolTip("여백, 단 구성, 문제 번호 등을 자동 적용합니다")
        self.layout_check.stateChanged.connect(self._on_changed)
        export_layout.addRow("", self.layout_check)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()

    def get_config(self) -> AppConfig:
        """현재 설정을 AppConfig로 반환"""
        languages = []
        if self.lang_ko.isChecked():
            languages.append("ko")
        if self.lang_en.isChecked():
            languages.append("en")
        if self.lang_ja.isChecked():
            languages.append("ja")

        if not languages:
            languages = ["ko", "en"]

        ocr_config = OCRConfig(
            text_type=self.text_type_combo.currentData(),
            languages=languages,
            enable_math_ocr=self.math_ocr_check.isChecked(),
            enable_table_ocr=self.table_ocr_check.isChecked(),
            use_gpu=self.gpu_check.isChecked(),
        )

        export_config = ExportConfig(
            output_format=self.format_combo.currentData(),
            apply_detected_layout=self.layout_check.isChecked(),
        )

        return AppConfig(ocr=ocr_config, export=export_config)

    def get_export_format(self) -> ExportFormat:
        """선택된 내보내기 형식 반환"""
        return self.format_combo.currentData()

    def _on_changed(self):
        """설정 변경 시그널"""
        self.settings_changed.emit()
