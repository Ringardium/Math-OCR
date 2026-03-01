"""
HWP 문서 내보내기 (pyhwpx 기반)
"""

from pathlib import Path
from typing import Optional
import io

try:
    import pyhwpx
    HAS_PYHWPX = True
except ImportError:
    HAS_PYHWPX = False

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None

from .base import BaseExporter
from ..core.builder import DocumentContent, ContentBlock, PageContent
from ..core.layout import Margins


class HwpExporter(BaseExporter):
    """HWP 문서로 내보내기"""

    def __init__(
        self,
        font_name: str = "맑은 고딕",
        font_size: int = 11
    ):
        if not HAS_PYHWPX:
            raise ImportError("pyhwpx가 필요합니다: pip install pyhwpx")

        self.font_name = font_name
        self.font_size = font_size

    @property
    def format_name(self) -> str:
        return "HWP"

    @property
    def file_extension(self) -> str:
        return ".hwp"

    def export(self, content: DocumentContent, output_path: Path) -> Path:
        """DocumentContent를 HWP 문서로 내보내기"""
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(self.file_extension)

        hwp = pyhwpx.Hwp(visible=False)

        try:
            # 각 페이지 처리
            for idx, page_content in enumerate(content.pages):
                if idx > 0:
                    # 페이지 나누기
                    hwp.insert_ctrl_character("page")

                # 레이아웃 적용
                self._apply_layout(hwp, page_content)

                # 콘텐츠 추가
                self._add_page_content(hwp, page_content)

            # 저장
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hwp.save_as(str(output_path))

        finally:
            hwp.quit()

        return output_path

    def _apply_layout(self, hwp, page: PageContent) -> None:
        """페이지 레이아웃 설정"""
        margins = page.metadata.get("margins")
        columns = page.metadata.get("columns", 1)

        if margins and isinstance(margins, Margins):
            # 여백을 mm 단위로 변환 (pyhwpx는 mm 사용)
            margins_cm = margins.to_cm(dpi=200)
            try:
                hwp.set_page_margin(
                    top=margins_cm.top * 10,
                    bottom=margins_cm.bottom * 10,
                    left=margins_cm.left * 10,
                    right=margins_cm.right * 10,
                    header=0,
                    footer=0,
                )
            except Exception:
                pass  # 기본 여백 유지

        # 다단 설정
        if columns == 2:
            try:
                column_gap = page.metadata.get("column_gap", 10)
                hwp.set_multi_column(2, gap=column_gap)
            except Exception:
                pass

    def _add_page_content(self, hwp, page: PageContent) -> None:
        """페이지 콘텐츠 추가"""
        for block in page.blocks:
            self._add_block(hwp, block)

    def _add_block(self, hwp, block: ContentBlock) -> None:
        """콘텐츠 블록 추가"""
        if block.block_type == "text":
            self._add_text_block(hwp, block)
        elif block.block_type == "math":
            self._add_math_block(hwp, block)
        elif block.block_type == "image":
            self._add_image_block(hwp, block)
        elif block.block_type == "table":
            self._add_table_block(hwp, block)

    def _add_text_block(self, hwp, block: ContentBlock) -> None:
        """텍스트 블록 추가"""
        if not block.content:
            return

        text = str(block.content)

        # 문제 번호가 있으면 볼드 처리
        problem_number = block.metadata.get("problem_number")
        indent_level = block.metadata.get("indent_level", 0)

        # 들여쓰기 적용
        if indent_level > 0:
            text = "    " * indent_level + text

        hwp.insert_text(text)
        hwp.insert_ctrl_character("paragraph")

    def _add_math_block(self, hwp, block: ContentBlock) -> None:
        """수식 블록 추가"""
        latex = block.metadata.get("latex") or block.content

        if not latex:
            return

        # HWP 수식 객체로 삽입 시도
        try:
            hwp.insert_equation(latex)
        except Exception:
            # 실패 시 텍스트로 삽입
            hwp.insert_text(f"$${latex}$$")

        hwp.insert_ctrl_character("paragraph")

    def _add_image_block(self, hwp, block: ContentBlock) -> None:
        """이미지 블록 추가"""
        if block.content is None or Image is None:
            return

        try:
            if isinstance(block.content, np.ndarray):
                pil_image = Image.fromarray(block.content)
            else:
                return

            # 임시 파일에 저장 후 삽입
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp, format="PNG")
                tmp_path = tmp.name

            hwp.insert_picture(tmp_path)

            # 임시 파일 삭제
            import os
            os.unlink(tmp_path)

        except Exception:
            hwp.insert_text("[이미지]")

        hwp.insert_ctrl_character("paragraph")

    def _add_table_block(self, hwp, block: ContentBlock) -> None:
        """표 블록 추가"""
        table_data = block.content

        if not table_data or not isinstance(table_data, list):
            if block.metadata.get("text"):
                hwp.insert_text(str(block.metadata.get("text")))
                hwp.insert_ctrl_character("paragraph")
            return

        if not table_data[0]:
            return

        rows = len(table_data)
        cols = len(table_data[0])

        try:
            hwp.insert_table(rows, cols)

            for i, row_data in enumerate(table_data):
                for j, cell_value in enumerate(row_data):
                    hwp.put_field_text(
                        f"cell_{i}_{j}",
                        str(cell_value) if cell_value else ""
                    )
        except Exception:
            # 표 삽입 실패 시 텍스트로 표시
            for row in table_data:
                hwp.insert_text(" | ".join(str(c) for c in row))
                hwp.insert_ctrl_character("paragraph")
