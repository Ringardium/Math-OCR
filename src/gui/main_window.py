"""
YM-OCR 메인 윈도우
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QFileDialog, QMessageBox, QSplitter,
    QStatusBar, QApplication, QStackedWidget
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QSettings

from .resources import APP_TITLE, APP_VERSION, SUPPORTED_FILE_FILTERS
from .thumbnail_panel import ThumbnailPanel
from .preview_panel import PreviewPanel
from .result_panel import ResultPanel
from .settings_panel import SettingsPanel
from .progress_widget import ProgressWidget
from .file_panel import DropZone
from .worker import OCRWorker, DocumentLoadWorker

from ..pipeline import PipelineResult
from ..config import AppConfig, ExportFormat


class MainWindow(QMainWindow):
    """YM-OCR 메인 윈도우"""

    def __init__(self):
        super().__init__()

        self._document = None
        self._result: Optional[PipelineResult] = None
        self._ocr_worker: Optional[OCRWorker] = None
        self._load_worker: Optional[DocumentLoadWorker] = None
        self._current_file: Optional[str] = None

        self._setup_window()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._load_stylesheet()

    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1200, 800)

        # 이전 위치/크기 복원
        settings = QSettings("YM-OCR", "MainWindow")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # 화면 중앙에 표시
            screen = QApplication.primaryScreen()
            if screen:
                screen_geo = screen.availableGeometry()
                self.resize(1400, 900)
                self.move(
                    (screen_geo.width() - 1400) // 2,
                    (screen_geo.height() - 900) // 2
                )

    def _setup_toolbar(self):
        """툴바 설정"""
        toolbar = QToolBar("메인 툴바")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        # 파일 열기
        self.open_action = QAction("📂 파일 열기", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._open_file)
        toolbar.addAction(self.open_action)

        toolbar.addSeparator()

        # 변환 시작
        self.convert_action = QAction("▶ 변환 시작", self)
        self.convert_action.setShortcut("Ctrl+R")
        self.convert_action.setEnabled(False)
        self.convert_action.triggered.connect(self._start_conversion)
        toolbar.addAction(self.convert_action)

        toolbar.addSeparator()

        # 내보내기
        self.export_action = QAction("💾 내보내기", self)
        self.export_action.setShortcut("Ctrl+S")
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self._export_result)
        toolbar.addAction(self.export_action)

        toolbar.addSeparator()

        # 설정 토글
        self.settings_action = QAction("⚙ 설정", self)
        self.settings_action.setCheckable(True)
        self.settings_action.triggered.connect(self._toggle_settings)
        toolbar.addAction(self.settings_action)

    def _setup_ui(self):
        """UI 레이아웃 구성"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 콘텐츠 영역
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 썸네일 패널
        self.thumbnail_panel = ThumbnailPanel()
        self.thumbnail_panel.page_selected.connect(self._on_page_selected)
        self.thumbnail_panel.hide()

        # 메인 영역 (드롭존 / 미리보기 전환)
        self.main_stack = QStackedWidget()

        # 드롭존 (파일 없을 때)
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._load_file)
        self.main_stack.addWidget(self.drop_zone)

        # 미리보기 + 결과 스플리터
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.preview_panel = PreviewPanel()
        self.preview_panel.region_clicked.connect(self._on_region_clicked)
        self.content_splitter.addWidget(self.preview_panel)

        self.result_panel = ResultPanel()
        self.result_panel.overlay_btn.toggled.connect(self.preview_panel.toggle_overlay)
        self.content_splitter.addWidget(self.result_panel)

        self.content_splitter.setSizes([600, 400])
        self.main_stack.addWidget(self.content_splitter)

        # 설정 패널 (숨김)
        self.settings_panel = SettingsPanel()
        self.settings_panel.setFixedWidth(280)
        self.settings_panel.hide()

        content_layout.addWidget(self.thumbnail_panel)
        content_layout.addWidget(self.main_stack, 1)
        content_layout.addWidget(self.settings_panel)

        main_layout.addWidget(content_widget, 1)

        # 진행 표시
        self.progress_widget = ProgressWidget()
        self.progress_widget.cancel_btn.clicked.connect(self._cancel_conversion)
        main_layout.addWidget(self.progress_widget)

    def _setup_statusbar(self):
        """상태 바 설정"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"YM-OCR v{APP_VERSION} — 파일을 열어 시작하세요")

    def _load_stylesheet(self):
        """스타일시트 로드"""
        import sys
        # PyInstaller 환경 대응
        if getattr(sys, 'frozen', False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent

        style_path = base / "src" / "gui" / "styles.qss" if getattr(sys, 'frozen', False) else base / "styles.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # === 파일 열기 ===

    def _open_file(self):
        """파일 대화상자로 파일 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "파일 열기",
            "",
            SUPPORTED_FILE_FILTERS
        )
        if file_path:
            self._load_file(file_path)

    def _load_file(self, file_path: str):
        """파일 로드"""
        self._current_file = file_path
        self.status_bar.showMessage(f"로딩 중: {Path(file_path).name}")

        # 이전 결과 초기화
        self._result = None
        self.result_panel.clear()
        self.export_action.setEnabled(False)

        # 백그라운드에서 문서 로드
        self._load_worker = DocumentLoadWorker(file_path)
        self._load_worker.finished.connect(self._on_document_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_document_loaded(self, document):
        """문서 로드 완료"""
        self._document = document

        # 썸네일 설정
        page_images = [page.image for page in document.pages]
        self.thumbnail_panel.set_pages(page_images)
        self.thumbnail_panel.show()

        # 첫 페이지 미리보기
        if document.pages:
            self.preview_panel.set_image(document.pages[0].image)

        # UI 전환
        self.main_stack.setCurrentIndex(1)
        self.convert_action.setEnabled(True)

        self.status_bar.showMessage(
            f"로드 완료: {Path(self._current_file).name} "
            f"({document.page_count}페이지)"
        )

    def _on_load_error(self, error_msg: str):
        """문서 로드 오류"""
        QMessageBox.critical(self, "로드 오류", f"파일을 열 수 없습니다:\n{error_msg}")
        self.status_bar.showMessage("로드 실패")

    # === 페이지 탐색 ===

    def _on_page_selected(self, page_index: int):
        """썸네일에서 페이지 선택"""
        if self._document and page_index < len(self._document.pages):
            page = self._document.pages[page_index]
            self.preview_panel.set_image(page.image)

            # 영역 오버레이 표시
            if page.regions:
                regions = [
                    {
                        "bbox": r.bbox.to_tuple(),
                        "type": r.region_type.value,
                    }
                    for r in page.regions
                ]
                self.preview_panel.set_regions(regions)

            # 결과 페이지도 전환
            if self._result and self._result.content:
                self.result_panel.show_page(page_index)

    def _on_region_clicked(self, region_index: int):
        """미리보기에서 영역 클릭"""
        self.status_bar.showMessage(f"영역 #{region_index + 1} 선택됨")

    # === OCR 변환 ===

    def _start_conversion(self):
        """OCR 변환 시작"""
        if not self._current_file:
            return

        # 설정 가져오기
        config = self.settings_panel.get_config()

        # UI 상태 변경
        self.convert_action.setEnabled(False)
        self.progress_widget.start()

        # 워커 시작
        self._ocr_worker = OCRWorker(
            input_path=self._current_file,
            config=config
        )
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.error.connect(self._on_ocr_error)
        self._ocr_worker.start()

    def _on_ocr_progress(self, message: str, progress: float):
        """OCR 진행 상황 업데이트"""
        self.progress_widget.update_progress(message, progress)

    def _on_ocr_finished(self, result: PipelineResult):
        """OCR 완료"""
        self._result = result
        self.progress_widget.stop()
        self.convert_action.setEnabled(True)

        if result.success:
            # 결과 표시
            if result.content:
                self.result_panel.set_content(result.content)

            # 통계 표시
            self.result_panel.set_stats(result.stats)

            # 영역 오버레이 업데이트
            if result.document:
                self._document = result.document
                current_page = self.thumbnail_panel.list_widget.currentRow()
                self._on_page_selected(max(current_page, 0))

            self.export_action.setEnabled(True)
            self.status_bar.showMessage(
                f"변환 완료 — "
                f"텍스트: {result.stats.get('text_regions', 0)}, "
                f"수식: {result.stats.get('math_regions', 0)}, "
                f"표: {result.stats.get('table_regions', 0)}"
            )
        else:
            errors = "\n".join(result.errors)
            QMessageBox.warning(self, "변환 오류", f"일부 오류가 발생했습니다:\n{errors}")
            self.status_bar.showMessage("변환 완료 (오류 있음)")

    def _on_ocr_error(self, error_msg: str):
        """OCR 오류"""
        self.progress_widget.stop()
        self.convert_action.setEnabled(True)
        QMessageBox.critical(self, "변환 오류", f"변환 중 오류 발생:\n{error_msg}")
        self.status_bar.showMessage("변환 실패")

    def _cancel_conversion(self):
        """변환 취소"""
        if self._ocr_worker and self._ocr_worker.isRunning():
            self._ocr_worker.cancel()
            self.progress_widget.stop()
            self.convert_action.setEnabled(True)
            self.status_bar.showMessage("변환 취소됨")

    # === 내보내기 ===

    def _export_result(self):
        """결과 내보내기"""
        if not self._result or not self._result.content:
            return

        export_format = self.settings_panel.get_export_format()

        if export_format == ExportFormat.HWP:
            ext = "HWP 파일 (*.hwp)"
            default_ext = ".hwp"
        else:
            ext = "Word 파일 (*.docx)"
            default_ext = ".docx"

        # 기본 파일명
        default_name = ""
        if self._current_file:
            default_name = str(Path(self._current_file).with_suffix(default_ext))

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "내보내기",
            default_name,
            ext
        )

        if not file_path:
            return

        try:
            from ..exporters.docx_exporter import DocxExporter

            if export_format == ExportFormat.HWP:
                from ..exporters.hwp_exporter import HwpExporter
                exporter = HwpExporter()
            else:
                exporter = DocxExporter()

            output_path = exporter.export(self._result.content, Path(file_path))
            self.status_bar.showMessage(f"저장 완료: {output_path}")

            QMessageBox.information(
                self,
                "내보내기 완료",
                f"파일이 저장되었습니다:\n{output_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "내보내기 오류",
                f"파일 저장 중 오류 발생:\n{str(e)}"
            )

    # === 설정 ===

    def _toggle_settings(self, checked: bool):
        """설정 패널 토글"""
        self.settings_panel.setVisible(checked)

    # === 윈도우 이벤트 ===

    def closeEvent(self, event):
        """윈도우 닫기 시 설정 저장"""
        settings = QSettings("YM-OCR", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())

        # 진행 중인 작업 취소
        if self._ocr_worker and self._ocr_worker.isRunning():
            self._ocr_worker.cancel()
            self._ocr_worker.wait(3000)

        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait(3000)

        event.accept()

    def dragEnterEvent(self, event):
        """메인 윈도우 드래그 진입"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """메인 윈도우 드롭"""
        urls = event.mimeData().urls()
        if urls:
            from .resources import SUPPORTED_EXTENSIONS
            file_path = urls[0].toLocalFile()
            if Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS:
                self._load_file(file_path)
