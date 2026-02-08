"""
문서 빌더 모듈 - OCR 결과를 조합하여 최종 문서 구성
"""

from dataclasses import dataclass, field
from typing import Any

from .document import Document, Page, Region
from ..config import RegionType


@dataclass
class ContentBlock:
    """출력 문서의 콘텐츠 블록"""
    block_type: str  # "text", "math", "image", "table"
    content: Any     # 텍스트, LaTeX, 이미지 바이트, 표 데이터
    metadata: dict = field(default_factory=dict)


@dataclass
class PageContent:
    """페이지 콘텐츠"""
    page_number: int
    blocks: list[ContentBlock] = field(default_factory=list)

    def add_block(self, block: ContentBlock) -> None:
        self.blocks.append(block)


@dataclass
class DocumentContent:
    """전체 문서 콘텐츠"""
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_page(self, page: PageContent) -> None:
        self.pages.append(page)


class DocumentBuilder:
    """OCR 결과로부터 문서 콘텐츠 생성"""

    def build(self, document: Document) -> DocumentContent:
        """Document에서 DocumentContent 생성"""
        doc_content = DocumentContent(
            metadata=document.metadata.copy()
        )

        for page in document.pages:
            page_content = self._build_page(page)
            doc_content.add_page(page_content)

        return doc_content

    def _build_page(self, page: Page) -> PageContent:
        """페이지 콘텐츠 생성"""
        page_content = PageContent(page_number=page.page_number)

        for region in page.regions:
            block = self._region_to_block(region)
            if block:
                page_content.add_block(block)

        return page_content

    def _region_to_block(self, region: Region) -> ContentBlock | None:
        """Region을 ContentBlock으로 변환"""
        if region.region_type == RegionType.TEXT:
            return ContentBlock(
                block_type="text",
                content=region.content or "",
                metadata={
                    "is_handwritten": region.is_handwritten,
                    "confidence": region.confidence,
                    "bbox": region.bbox.to_tuple()
                }
            )

        elif region.region_type == RegionType.MATH:
            return ContentBlock(
                block_type="math",
                content=region.content or "",
                metadata={
                    "latex": region.metadata.get("latex"),
                    "confidence": region.confidence,
                    "bbox": region.bbox.to_tuple()
                }
            )

        elif region.region_type == RegionType.IMAGE:
            return ContentBlock(
                block_type="image",
                content=region.image,
                metadata={
                    "bbox": region.bbox.to_tuple()
                }
            )

        elif region.region_type == RegionType.TABLE:
            return ContentBlock(
                block_type="table",
                content=region.metadata.get("table_data", region.content),
                metadata={
                    "confidence": region.confidence,
                    "bbox": region.bbox.to_tuple()
                }
            )

        return None
