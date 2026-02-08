"""
OCR 기본 클래스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class OCRResult:
    """OCR 결과"""
    text: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # 수식인 경우 LaTeX
    latex: Optional[str] = None

    # 표인 경우 구조화된 데이터
    table_data: Optional[list[list[str]]] = None


class BaseOCR(ABC):
    """OCR 엔진 기본 클래스"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """OCR 엔진 초기화 (지연 로딩)"""
        pass

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """이미지에서 텍스트/수식 인식"""
        pass

    def ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    def __call__(self, image: np.ndarray) -> OCRResult:
        """호출 가능한 인터페이스"""
        self.ensure_initialized()
        return self.recognize(image)
