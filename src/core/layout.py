"""
레이아웃 데이터 모델
"""

from dataclasses import dataclass, field
from typing import Optional

from .document import BoundingBox
from ..config import RegionType


@dataclass
class Margins:
    """페이지 여백 (픽셀 단위)"""
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def to_cm(self, dpi: int = 200) -> "Margins":
        """픽셀을 cm로 변환한 새 Margins 반환"""
        px_per_cm = dpi / 2.54
        return Margins(
            top=round(self.top / px_per_cm * 100) / 100,
            bottom=round(self.bottom / px_per_cm * 100) / 100,
            left=round(self.left / px_per_cm * 100) / 100,
            right=round(self.right / px_per_cm * 100) / 100,
        )


@dataclass
class LayoutRegion:
    """레이아웃 분석이 적용된 영역"""
    bbox: BoundingBox
    region_type: RegionType
    column: int = 0                        # 0=전체폭, 1=왼쪽단, 2=오른쪽단
    problem_number: Optional[str] = None   # "1", "2", "3" 등
    is_choice_group: bool = False          # ①②③④ 보기 그룹 여부
    indent_level: int = 0                  # 들여쓰기 수준


@dataclass
class PageLayout:
    """페이지 레이아웃 정보"""
    margins: Margins = field(default_factory=Margins)
    columns: int = 1                       # 1단 또는 2단
    column_gap: float = 0.0               # 단 간격 (px)
    line_spacing: float = 1.0             # 줄 간격 비율
    content_regions: list[LayoutRegion] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
