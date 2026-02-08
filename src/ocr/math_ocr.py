"""
수학 수식 OCR (pix2tex 기반)
"""

from typing import Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .base import BaseOCR, OCRResult


class MathOCR(BaseOCR):
    """수학 수식 OCR (LaTeX-OCR / pix2tex)"""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self._model = None

    def initialize(self) -> None:
        """pix2tex 모델 초기화"""
        try:
            from pix2tex.cli import LatexOCR
            self._model = LatexOCR()
        except ImportError:
            raise ImportError("pix2tex가 필요합니다: pip install pix2tex")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """수식 인식하여 LaTeX 반환"""
        self.ensure_initialized()

        # numpy array를 PIL Image로 변환
        if Image is None:
            raise ImportError("Pillow가 필요합니다: pip install Pillow")

        pil_image = Image.fromarray(image)

        # LaTeX 변환
        try:
            latex = self._model(pil_image)
            confidence = 0.9  # pix2tex는 신뢰도를 직접 제공하지 않음
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                metadata={"error": str(e)}
            )

        return OCRResult(
            text=latex,
            confidence=confidence,
            latex=latex,
            metadata={"format": "latex"}
        )


class NougatMathOCR(BaseOCR):
    """Nougat 기반 수식 OCR (학술 문서 특화)"""

    def __init__(self, use_gpu: bool = True, model_tag: str = "0.1.0-small"):
        super().__init__(use_gpu)
        self.model_tag = model_tag
        self._model = None
        self._processor = None

    def initialize(self) -> None:
        """Nougat 모델 초기화"""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            import torch

            self._processor = NougatProcessor.from_pretrained(
                f"facebook/nougat-{self.model_tag}"
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                f"facebook/nougat-{self.model_tag}"
            )

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

        except ImportError:
            raise ImportError(
                "transformers가 필요합니다: pip install transformers torch"
            )

    def recognize(self, image: np.ndarray) -> OCRResult:
        """수식 인식"""
        self.ensure_initialized()

        if Image is None:
            raise ImportError("Pillow가 필요합니다")

        import torch

        pil_image = Image.fromarray(image)

        # 전처리
        pixel_values = self._processor(
            pil_image,
            return_tensors="pt"
        ).pixel_values

        if self.use_gpu and torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        # 추론
        outputs = self._model.generate(
            pixel_values,
            max_length=512,
            bad_words_ids=[[self._processor.tokenizer.unk_token_id]]
        )

        # 디코딩
        text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return OCRResult(
            text=text,
            confidence=0.85,
            latex=text,
            metadata={"model": "nougat"}
        )
