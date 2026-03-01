"""
OCR 결과 미리보기 패널
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLabel, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..core.builder import DocumentContent, ContentBlock, PageContent


class ResultPanel(QWidget):
    """OCR 결과 표시 및 편집 패널"""

    content_edited = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ResultPanel")
        self._current_page: int = 0
        self._content: Optional[DocumentContent] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 헤더
        header = QWidget()
        header.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("OCR 결과")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # 오버레이 토글 버튼
        self.overlay_btn = QPushButton("영역 표시")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setChecked(True)
        self.overlay_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px; border-radius: 4px;
                border: 1px solid #e0e0e0; font-size: 11px;
            }
            QPushButton:checked {
                background-color: #e8f0fe; border-color: #1a73e8; color: #1a73e8;
            }
        """)
        header_layout.addWidget(self.overlay_btn)

        layout.addWidget(header)

        # 결과 텍스트 편집기
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("맑은 고딕", 11))
        self.text_edit.setReadOnly(False)
        self.text_edit.setPlaceholderText(
            "파일을 열고 변환을 시작하면 OCR 결과가 여기에 표시됩니다."
        )
        self.text_edit.textChanged.connect(self._on_text_changed)

        layout.addWidget(self.text_edit)

        # 통계 바
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(
            "padding: 4px 12px; color: #666; font-size: 11px; "
            "background-color: #f8f9fa; border-top: 1px solid #e0e0e0;"
        )
        layout.addWidget(self.stats_label)

    def set_content(self, content: DocumentContent):
        """OCR 결과 설정"""
        self._content = content
        self._show_page(0)

    def show_page(self, page_index: int):
        """특정 페이지의 OCR 결과 표시"""
        self._show_page(page_index)

    def _show_page(self, page_index: int):
        """페이지 콘텐츠를 텍스트로 변환하여 표시"""
        self._current_page = page_index

        if not self._content or page_index >= len(self._content.pages):
            self.text_edit.clear()
            return

        page = self._content.pages[page_index]
        text_parts = []

        for block in page.blocks:
            if block.block_type == "text":
                problem_num = block.metadata.get("problem_number")
                content = str(block.content) if block.content else ""
                if problem_num:
                    text_parts.append(f"[문제 {problem_num}] {content}")
                else:
                    text_parts.append(content)

            elif block.block_type == "math":
                latex = block.metadata.get("latex") or block.content
                text_parts.append(f"$$ {latex} $$")

            elif block.block_type == "image":
                text_parts.append("[이미지]")

            elif block.block_type == "table":
                text_parts.append("[표]")
                if isinstance(block.content, list):
                    for row in block.content:
                        text_parts.append(
                            " | ".join(str(c) for c in row)
                        )

        # 블록 구분자 없이 직접 연결
        self.text_edit.blockSignals(True)
        self.text_edit.setPlainText("\n\n".join(text_parts))
        self.text_edit.blockSignals(False)

    def set_stats(self, stats: dict):
        """통계 표시"""
        parts = []
        if stats.get("pages"):
            parts.append(f"페이지: {stats['pages']}")
        if stats.get("text_regions"):
            parts.append(f"텍스트: {stats['text_regions']}")
        if stats.get("math_regions"):
            parts.append(f"수식: {stats['math_regions']}")
        if stats.get("table_regions"):
            parts.append(f"표: {stats['table_regions']}")
        if stats.get("image_regions"):
            parts.append(f"이미지: {stats['image_regions']}")

        self.stats_label.setText("  |  ".join(parts))

    def clear(self):
        """초기화"""
        self.text_edit.clear()
        self.stats_label.setText("")
        self._content = None

    def _on_text_changed(self):
        """텍스트 편집 시"""
        self.content_edited.emit()
