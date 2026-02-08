"""
문서 데이터 모델
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import numpy as np

from ..config import RegionType


@dataclass
class BoundingBox:
    """영역 경계 박스"""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        return cls(x=x1, y=y1, width=x2-x1, height=y2-y1)


@dataclass
class Region:
    """문서 내 영역"""
    bbox: BoundingBox
    region_type: RegionType

    # 영역 이미지 (numpy array)
    image: Optional[np.ndarray] = None

    # OCR 결과
    content: Optional[str] = None

    # 손글씨 여부 (None이면 미분류)
    is_handwritten: Optional[bool] = None

    # 신뢰도 점수
    confidence: float = 0.0

    # 추가 메타데이터
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Page:
    """문서 페이지"""
    page_number: int
    image: np.ndarray
    width: int
    height: int

    # 감지된 영역들
    regions: list[Region] = field(default_factory=list)

    # 추가 메타데이터
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_region(self, region: Region) -> None:
        self.regions.append(region)

    def get_regions_by_type(self, region_type: RegionType) -> list[Region]:
        return [r for r in self.regions if r.region_type == region_type]

    def sort_regions_by_position(self) -> None:
        """영역을 읽는 순서대로 정렬 (위→아래, 왼쪽→오른쪽)"""
        self.regions.sort(key=lambda r: (r.bbox.y, r.bbox.x))


@dataclass
class Document:
    """전체 문서"""
    source_path: Path
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def add_page(self, page: Page) -> None:
        self.pages.append(page)

    def get_all_regions(self) -> list[Region]:
        """모든 페이지의 영역 반환"""
        regions = []
        for page in self.pages:
            regions.extend(page.regions)
        return regions
