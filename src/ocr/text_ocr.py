"""
텍스트 OCR (EasyOCR 기반)
"""

from typing import Optional
import numpy as np

from .base import BaseOCR, OCRResult


class TextOCR(BaseOCR):
    """일반 텍스트 OCR"""

    def __init__(
        self,
        languages: list[str] = None,
        use_gpu: bool = True
    ):
        super().__init__(use_gpu)
        self.languages = languages or ["ko", "en"]
        self._reader = None

    def initialize(self) -> None:
        """EasyOCR 초기화"""
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu
            )
        except ImportError:
            raise ImportError("EasyOCR이 필요합니다: pip install easyocr")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """텍스트 인식"""
        self.ensure_initialized()

        results = self._reader.readtext(image)

        if not results:
            return OCRResult(text="", confidence=0.0)

        # 결과 합치기
        texts = []
        confidences = []

        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            metadata={
                "raw_results": results,
                "word_count": len(texts)
            }
        )

    def recognize_with_positions(self, image: np.ndarray) -> list[dict]:
        """위치 정보와 함께 텍스트 인식"""
        self.ensure_initialized()

        results = self._reader.readtext(image)

        parsed = []
        for bbox, text, conf in results:
            parsed.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox,
                "x": int(min(p[0] for p in bbox)),
                "y": int(min(p[1] for p in bbox)),
            })

        return parsed


class PaddleTextOCR(BaseOCR):
    """PaddleOCR 기반 텍스트 OCR (대안)"""

    def __init__(
        self,
        lang: str = "korean",
        use_gpu: bool = True
    ):
        super().__init__(use_gpu)
        self.lang = lang
        self._ocr = None

    def initialize(self) -> None:
        """PaddleOCR 초기화"""
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu
            )
        except ImportError:
            raise ImportError("PaddleOCR이 필요합니다: pip install paddleocr")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """텍스트 인식"""
        self.ensure_initialized()

        result = self._ocr.ocr(image, cls=True)

        if not result or not result[0]:
            return OCRResult(text="", confidence=0.0)

        texts = []
        confidences = []

        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            texts.append(text)
            confidences.append(conf)

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence
        )
