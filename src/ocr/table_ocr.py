"""
표 OCR 모듈
"""

from typing import Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .base import BaseOCR, OCRResult


class TableOCR(BaseOCR):
    """표 인식 OCR"""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self._extractor = None

    def initialize(self) -> None:
        """img2table 초기화"""
        try:
            from img2table.document import Image as Img2TableImage
            from img2table.ocr import TesseractOCR
            self._ocr = TesseractOCR(lang="kor+eng")
            self._initialized = True
        except ImportError:
            # fallback: 기본 OCR 사용
            self._ocr = None
            self._initialized = True

    def recognize(self, image: np.ndarray) -> OCRResult:
        """표 인식"""
        self.ensure_initialized()

        try:
            from img2table.document import Image as Img2TableImage

            # numpy to PIL
            if Image is None:
                raise ImportError("Pillow가 필요합니다")

            pil_image = Image.fromarray(image)

            # 임시 파일로 저장 (img2table은 파일 경로 필요)
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            try:
                # 표 추출
                doc = Img2TableImage(src=tmp_path)
                tables = doc.extract_tables(
                    ocr=self._ocr,
                    implicit_rows=True,
                    borderless_tables=True
                )

                if not tables:
                    return OCRResult(
                        text="",
                        confidence=0.0,
                        table_data=[]
                    )

                # 첫 번째 표 데이터 추출
                table = tables[0]
                table_data = []

                for row in table.content.values():
                    row_data = []
                    for cell in row:
                        row_data.append(cell.value if cell.value else "")
                    table_data.append(row_data)

                # 텍스트 형태로도 변환
                text_lines = []
                for row in table_data:
                    text_lines.append(" | ".join(row))
                text = "\n".join(text_lines)

                return OCRResult(
                    text=text,
                    confidence=0.8,
                    table_data=table_data,
                    metadata={"table_count": len(tables)}
                )

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                metadata={"error": str(e)}
            )


class SimpleTableOCR(BaseOCR):
    """간단한 표 인식 (OpenCV 기반)"""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)

    def initialize(self) -> None:
        """초기화"""
        try:
            import cv2
            self._initialized = True
        except ImportError:
            raise ImportError("OpenCV가 필요합니다: pip install opencv-python")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """표 구조 인식"""
        self.ensure_initialized()
        import cv2

        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 수평/수직 선 감지
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # 표 구조 찾기
        table_mask = cv2.add(horizontal_lines, vertical_lines)

        # 셀 영역 찾기
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # 최소 크기
                cells.append({"x": x, "y": y, "w": w, "h": h})

        return OCRResult(
            text="",
            confidence=0.7,
            metadata={
                "cells": cells,
                "cell_count": len(cells)
            }
        )
