"""
OCR 처리 워커 스레드
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..pipeline import OCRPipeline, PipelineResult
from ..config import AppConfig


class OCRWorker(QThread):
    """백그라운드 OCR 처리 워커"""

    # 시그널
    progress = pyqtSignal(str, float)  # (메시지, 진행률 0.0~1.0)
    finished = pyqtSignal(PipelineResult)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        config: Optional[AppConfig] = None,
        parent=None
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.config = config or AppConfig()
        self._cancelled = False

    def run(self):
        """OCR 파이프라인 실행"""
        try:
            pipeline = OCRPipeline(
                config=self.config,
                progress_callback=self._on_progress
            )

            result = pipeline.process(
                self.input_path,
                self.output_path
            )

            if not self._cancelled:
                self.finished.emit(result)

        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def _on_progress(self, message: str, progress: float):
        """진행률 콜백"""
        if not self._cancelled:
            self.progress.emit(message, progress)

    def cancel(self):
        """처리 취소"""
        self._cancelled = True


class DocumentLoadWorker(QThread):
    """문서 로딩 워커 (미리보기용)"""

    finished = pyqtSignal(object)  # Document 객체
    error = pyqtSignal(str)

    def __init__(self, input_path: str, parent=None):
        super().__init__(parent)
        self.input_path = input_path

    def run(self):
        """문서 로드"""
        try:
            from ..core.loader import AutoLoader
            loader = AutoLoader(dpi=150)  # 미리보기용 낮은 DPI
            document = loader.load(Path(self.input_path))
            self.finished.emit(document)
        except Exception as e:
            self.error.emit(str(e))
