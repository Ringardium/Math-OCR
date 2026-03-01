"""
수학 수식 OCR (Texify 기반)
"""

from typing import Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .base import BaseOCR, OCRResult


class MathOCR(BaseOCR):
    """수학 수식 OCR (Texify)"""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self._model = None
        self._processor = None

    def initialize(self) -> None:
        """Texify 모델 초기화"""
        try:
            from texify.model.model import load_model
            from texify.model.processor import load_processor

            self._model = load_model()
            self._processor = load_processor()

            if not self.use_gpu:
                self._model = self._model.cpu()

        except ImportError:
            raise ImportError("texify가 필요합니다: pip install texify")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """수식 인식하여 LaTeX 반환"""
        self.ensure_initialized()

        if Image is None:
            raise ImportError("Pillow가 필요합니다: pip install Pillow")

        pil_image = Image.fromarray(image)

        try:
            from texify.inference import batch_inference

            results = batch_inference(
                [pil_image],
                self._model,
                self._processor
            )

            if results and len(results) > 0:
                latex = results[0]
                confidence = 0.92
            else:
                latex = ""
                confidence = 0.0

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
            metadata={"format": "latex", "engine": "texify"}
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
