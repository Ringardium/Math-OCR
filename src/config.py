"""
설정 관리 모듈
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TextType(Enum):
    """텍스트 유형"""
    PRINTED = "printed"      # 인쇄체
    HANDWRITTEN = "handwritten"  # 손글씨
    BOTH = "both"            # 둘 다


class RegionType(Enum):
    """영역 유형"""
    TEXT = "text"
    MATH = "math"
    IMAGE = "image"
    TABLE = "table"
    UNKNOWN = "unknown"


class ExportFormat(Enum):
    """출력 형식"""
    DOCX = "docx"
    HWP = "hwp"


@dataclass
class OCRConfig:
    """OCR 설정"""
    # 텍스트 유형 선택
    text_type: TextType = TextType.BOTH

    # 언어 설정
    languages: list[str] = field(default_factory=lambda: ["ko", "en"])

    # 수식 OCR 활성화
    enable_math_ocr: bool = True

    # 표 인식 활성화
    enable_table_ocr: bool = True

    # 이미지 추출 활성화
    extract_images: bool = True

    # 신뢰도 임계값
    confidence_threshold: float = 0.5

    # GPU 사용
    use_gpu: bool = True


@dataclass
class ExportConfig:
    """내보내기 설정"""
    # 출력 형식
    output_format: ExportFormat = ExportFormat.DOCX

    # 수식을 이미지로 내보내기 (False면 LaTeX 텍스트)
    math_as_image: bool = True

    # 원본 레이아웃 유지 시도
    preserve_layout: bool = True

    # 감지된 레이아웃 적용 여부
    apply_detected_layout: bool = True

    # 출력 경로
    output_path: Optional[Path] = None


@dataclass
class AppConfig:
    """전체 앱 설정"""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # 모델 경로
    models_dir: Path = field(default_factory=lambda: Path("models"))

    # 임시 파일 경로
    temp_dir: Path = field(default_factory=lambda: Path("temp"))

    # 디버그 모드
    debug: bool = False
