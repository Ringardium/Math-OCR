"""
문서 로더 모듈
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .document import Document, Page


class DocumentLoader(ABC):
    """문서 로더 기본 클래스"""

    @abstractmethod
    def load(self, path: Path) -> Document:
        """문서를 로드하여 Document 객체 반환"""
        pass

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """해당 파일 형식 지원 여부"""
        pass


class PDFLoader(DocumentLoader):
    """PDF 문서 로더"""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, dpi: int = 200):
        self.dpi = dpi
        if fitz is None:
            raise ImportError("PyMuPDF가 필요합니다: pip install PyMuPDF")

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> Document:
        doc = Document(source_path=path)

        pdf_doc = fitz.open(str(path))
        doc.metadata["pdf_metadata"] = pdf_doc.metadata

        for page_num in range(len(pdf_doc)):
            pdf_page = pdf_doc[page_num]

            # 페이지를 이미지로 변환
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = pdf_page.get_pixmap(matrix=mat)

            # numpy 배열로 변환
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, pix.n)

            # RGB로 변환 (RGBA인 경우)
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            page = Page(
                page_number=page_num + 1,
                image=img_array,
                width=pix.width,
                height=pix.height
            )

            # PDF에서 직접 이미지 추출 (메타데이터로 저장)
            page.metadata["embedded_images"] = self._extract_images(pdf_page)

            doc.add_page(page)

        pdf_doc.close()
        return doc

    def _extract_images(self, pdf_page) -> list[dict]:
        """PDF 페이지에서 임베디드 이미지 추출"""
        images = []
        image_list = pdf_page.get_images()

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            images.append({
                "xref": xref,
                "index": img_index,
            })

        return images


class ImageLoader(DocumentLoader):
    """이미지 파일 로더"""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self):
        if Image is None:
            raise ImportError("Pillow가 필요합니다: pip install Pillow")

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> Document:
        doc = Document(source_path=path)

        # 이미지 로드
        img = Image.open(path)

        # RGB로 변환
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img)

        page = Page(
            page_number=1,
            image=img_array,
            width=img.width,
            height=img.height
        )

        doc.add_page(page)

        return doc


class AutoLoader:
    """파일 확장자에 따라 적절한 로더 자동 선택"""

    def __init__(self, dpi: int = 200):
        self.loaders: list[DocumentLoader] = [
            PDFLoader(dpi=dpi),
            ImageLoader(),
        ]

    def load(self, path: Path) -> Document:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        for loader in self.loaders:
            if loader.supports(path):
                return loader.load(path)

        supported = set()
        for loader in self.loaders:
            if hasattr(loader, "SUPPORTED_EXTENSIONS"):
                supported.update(loader.SUPPORTED_EXTENSIONS)

        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {path.suffix}\n"
            f"지원 형식: {', '.join(sorted(supported))}"
        )
