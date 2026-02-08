"""
OCR 파이프라인 - 전체 변환 프로세스 조율
"""

from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

from tqdm import tqdm

from .config import AppConfig, OCRConfig, ExportConfig, TextType, RegionType
from .core.document import Document, Page, Region
from .core.loader import AutoLoader
from .core.detector import RegionDetector
from .core.builder import DocumentBuilder, DocumentContent
from .ocr.base import BaseOCR
from .ocr.text_ocr import TextOCR
from .ocr.math_ocr import MathOCR
from .ocr.table_ocr import TableOCR
from .classifiers.text_classifier import TextClassifier, HandwritingClassifier, TextClass
from .exporters.base import BaseExporter
from .exporters.docx_exporter import DocxExporter


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    success: bool
    output_path: Optional[Path] = None
    document: Optional[Document] = None
    content: Optional[DocumentContent] = None
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class OCRPipeline:
    """
    OCR 변환 파이프라인

    사용법:
        pipeline = OCRPipeline()
        result = pipeline.process("input.pdf", "output.docx")
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        text_ocr: Optional[BaseOCR] = None,
        math_ocr: Optional[BaseOCR] = None,
        table_ocr: Optional[BaseOCR] = None,
        classifier: Optional[TextClassifier] = None,
        exporter: Optional[BaseExporter] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.config = config or AppConfig()

        # OCR 엔진들 (지연 초기화)
        self._text_ocr = text_ocr
        self._math_ocr = math_ocr
        self._table_ocr = table_ocr

        # 분류기
        self._classifier = classifier

        # 내보내기
        self._exporter = exporter

        # 진행 콜백
        self.progress_callback = progress_callback

        # 컴포넌트들
        self._loader = AutoLoader(dpi=200)
        self._detector = RegionDetector()
        self._builder = DocumentBuilder()

    @property
    def text_ocr(self) -> BaseOCR:
        if self._text_ocr is None:
            self._text_ocr = TextOCR(
                languages=self.config.ocr.languages,
                use_gpu=self.config.ocr.use_gpu
            )
        return self._text_ocr

    @property
    def math_ocr(self) -> BaseOCR:
        if self._math_ocr is None:
            self._math_ocr = MathOCR(use_gpu=self.config.ocr.use_gpu)
        return self._math_ocr

    @property
    def table_ocr(self) -> BaseOCR:
        if self._table_ocr is None:
            self._table_ocr = TableOCR(use_gpu=self.config.ocr.use_gpu)
        return self._table_ocr

    @property
    def classifier(self) -> TextClassifier:
        if self._classifier is None:
            self._classifier = HandwritingClassifier()
        return self._classifier

    @property
    def exporter(self) -> BaseExporter:
        if self._exporter is None:
            self._exporter = DocxExporter()
        return self._exporter

    def _report_progress(self, message: str, progress: float) -> None:
        """진행 상황 보고"""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def process(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> PipelineResult:
        """
        문서 OCR 변환 수행

        Args:
            input_path: 입력 파일 경로 (PDF 또는 이미지)
            output_path: 출력 파일 경로 (기본: 입력파일명.docx)

        Returns:
            PipelineResult: 변환 결과
        """
        input_path = Path(input_path)
        errors = []
        stats = {
            "pages": 0,
            "regions": 0,
            "text_regions": 0,
            "math_regions": 0,
            "image_regions": 0,
            "table_regions": 0
        }

        try:
            # 1. 문서 로드
            self._report_progress("문서 로딩 중...", 0.1)
            document = self._loader.load(input_path)
            stats["pages"] = document.page_count

            # 2. 영역 감지
            self._report_progress("영역 감지 중...", 0.2)
            for page in tqdm(document.pages, desc="영역 감지"):
                self._detector.process_page(page)
                stats["regions"] += len(page.regions)

            # 3. 텍스트 분류 및 필터링
            self._report_progress("텍스트 분류 중...", 0.3)
            self._classify_and_filter(document)

            # 4. OCR 수행
            self._report_progress("OCR 처리 중...", 0.5)
            self._perform_ocr(document, stats)

            # 5. 문서 콘텐츠 생성
            self._report_progress("문서 생성 중...", 0.8)
            content = self._builder.build(document)

            # 6. 내보내기
            if output_path is None:
                output_path = input_path.with_suffix(self.exporter.file_extension)
            else:
                output_path = Path(output_path)

            self._report_progress("파일 저장 중...", 0.9)
            final_path = self.exporter.export(content, output_path)

            self._report_progress("완료!", 1.0)

            return PipelineResult(
                success=True,
                output_path=final_path,
                document=document,
                content=content,
                stats=stats
            )

        except Exception as e:
            errors.append(str(e))
            return PipelineResult(
                success=False,
                errors=errors,
                stats=stats
            )

    def _classify_and_filter(self, document: Document) -> None:
        """텍스트 분류 및 필터링"""
        text_type = self.config.ocr.text_type

        for page in document.pages:
            regions_to_keep = []

            for region in page.regions:
                if region.region_type != RegionType.TEXT:
                    regions_to_keep.append(region)
                    continue

                # 손글씨/인쇄체 분류
                if region.image is not None:
                    result = self.classifier.classify(region.image)
                    region.is_handwritten = (result.text_class == TextClass.HANDWRITTEN)
                    region.metadata["classification_confidence"] = result.confidence

                # 필터링
                if text_type == TextType.BOTH:
                    regions_to_keep.append(region)
                elif text_type == TextType.PRINTED and not region.is_handwritten:
                    regions_to_keep.append(region)
                elif text_type == TextType.HANDWRITTEN and region.is_handwritten:
                    regions_to_keep.append(region)

            page.regions = regions_to_keep

    def _perform_ocr(self, document: Document, stats: dict) -> None:
        """OCR 수행"""
        for page in tqdm(document.pages, desc="OCR 처리"):
            for region in page.regions:
                if region.image is None:
                    continue

                try:
                    if region.region_type == RegionType.TEXT:
                        result = self.text_ocr(region.image)
                        region.content = result.text
                        region.confidence = result.confidence
                        stats["text_regions"] += 1

                    elif region.region_type == RegionType.MATH and self.config.ocr.enable_math_ocr:
                        result = self.math_ocr(region.image)
                        region.content = result.text
                        region.confidence = result.confidence
                        region.metadata["latex"] = result.latex
                        stats["math_regions"] += 1

                    elif region.region_type == RegionType.TABLE and self.config.ocr.enable_table_ocr:
                        result = self.table_ocr(region.image)
                        region.content = result.text
                        region.confidence = result.confidence
                        region.metadata["table_data"] = result.table_data
                        stats["table_regions"] += 1

                    elif region.region_type == RegionType.IMAGE:
                        # 이미지는 OCR 없이 그대로 유지
                        stats["image_regions"] += 1

                except Exception as e:
                    region.metadata["ocr_error"] = str(e)


class SimplePipeline:
    """
    간단한 사용을 위한 래퍼

    사용법:
        from src.pipeline import SimplePipeline

        # 기본 사용
        SimplePipeline.convert("input.pdf", "output.docx")

        # 손글씨만
        SimplePipeline.convert("input.pdf", "output.docx", text_type="handwritten")

        # 인쇄체만
        SimplePipeline.convert("input.pdf", "output.docx", text_type="printed")
    """

    @staticmethod
    def convert(
        input_path: str,
        output_path: Optional[str] = None,
        text_type: str = "both",
        languages: list[str] = None,
        enable_math: bool = True,
        enable_table: bool = True,
        use_gpu: bool = True
    ) -> PipelineResult:
        """
        간단한 변환 함수

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로
            text_type: "both", "printed", "handwritten"
            languages: OCR 언어 목록
            enable_math: 수식 OCR 활성화
            enable_table: 표 OCR 활성화
            use_gpu: GPU 사용

        Returns:
            PipelineResult
        """
        # 설정 생성
        text_type_enum = {
            "both": TextType.BOTH,
            "printed": TextType.PRINTED,
            "handwritten": TextType.HANDWRITTEN
        }.get(text_type, TextType.BOTH)

        config = AppConfig(
            ocr=OCRConfig(
                text_type=text_type_enum,
                languages=languages or ["ko", "en"],
                enable_math_ocr=enable_math,
                enable_table_ocr=enable_table,
                use_gpu=use_gpu
            )
        )

        pipeline = OCRPipeline(config=config)
        return pipeline.process(input_path, output_path)
