"""
레이아웃 분석기 - 페이지 여백, 단 구성, 문제 번호, 보기 그룹 감지
"""

import re
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .document import Page, Region, BoundingBox
from .layout import PageLayout, Margins, LayoutRegion
from ..config import RegionType


class LayoutAnalyzer:
    """페이지 레이아웃 분석"""

    # 문제 번호 패턴
    PROBLEM_NUMBER_PATTERNS = [
        re.compile(r'^(\d{1,3})\s*[.\)]\s'),       # "1. " "2) "
        re.compile(r'^\[(\d{1,3})\]\s'),            # "[1] "
        re.compile(r'^(\d{1,3})\s*번\s'),            # "1번 "
        re.compile(r'^문\s*(\d{1,3})\s*[.\)]\s'),    # "문1. "
    ]

    # 보기 패턴
    CHOICE_PATTERNS = [
        re.compile(r'[①②③④⑤]'),                    # 원문자
        re.compile(r'[ㄱㄴㄷㄹㅁ]\.?\s'),             # ㄱ. ㄴ. ㄷ. ㄹ.
        re.compile(r'\([1-5]\)'),                    # (1) (2) (3)
    ]

    def __init__(self, min_margin_px: int = 20):
        self.min_margin_px = min_margin_px

    def analyze(self, page: Page) -> PageLayout:
        """페이지 레이아웃 분석"""
        layout = PageLayout(
            page_width=page.width,
            page_height=page.height
        )

        # 1. 여백 감지
        layout.margins = self._detect_margins(page.image)

        # 2. 단 구성 감지
        columns, column_gap = self._detect_columns(page.image, layout.margins)
        layout.columns = columns
        layout.column_gap = column_gap

        # 3. 줄 간격 추정
        layout.line_spacing = self._estimate_line_spacing(page.regions)

        # 4. 영역별 레이아웃 정보 분석
        for region in page.regions:
            layout_region = self._analyze_region(
                region, layout.columns, layout.margins, page.width
            )
            layout.content_regions.append(layout_region)

        return layout

    def _detect_margins(self, image: np.ndarray) -> Margins:
        """이미지 프로젝션으로 여백 감지"""
        if cv2 is None:
            return Margins()

        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 이진화 (흰 배경, 검은 텍스트)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape

        # 수평 프로젝션 (각 행의 검은 픽셀 수)
        h_proj = np.sum(binary > 0, axis=1)
        # 수직 프로젝션 (각 열의 검은 픽셀 수)
        v_proj = np.sum(binary > 0, axis=0)

        threshold = w * 0.01  # 1% 이상의 픽셀이 있으면 콘텐츠

        # 상단 여백
        top = 0
        for i in range(h):
            if h_proj[i] > threshold:
                top = i
                break

        # 하단 여백
        bottom = 0
        for i in range(h - 1, -1, -1):
            if h_proj[i] > threshold:
                bottom = h - i - 1
                break

        threshold_v = h * 0.01

        # 좌측 여백
        left = 0
        for i in range(w):
            if v_proj[i] > threshold_v:
                left = i
                break

        # 우측 여백
        right = 0
        for i in range(w - 1, -1, -1):
            if v_proj[i] > threshold_v:
                right = w - i - 1
                break

        return Margins(
            top=max(top, self.min_margin_px),
            bottom=max(bottom, self.min_margin_px),
            left=max(left, self.min_margin_px),
            right=max(right, self.min_margin_px)
        )

    def _detect_columns(
        self, image: np.ndarray, margins: Margins
    ) -> tuple[int, float]:
        """수직 프로젝션으로 단 구성 감지"""
        if cv2 is None:
            return 1, 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape

        # 콘텐츠 영역만 분석
        content_left = margins.left
        content_right = w - margins.right
        content_top = margins.top
        content_bottom = h - margins.bottom

        if content_right <= content_left or content_bottom <= content_top:
            return 1, 0.0

        content = binary[content_top:content_bottom, content_left:content_right]
        content_w = content.shape[1]

        # 수직 프로젝션
        v_proj = np.sum(content > 0, axis=0)

        # 중앙 30% 영역에서 빈 공간 찾기
        center_start = int(content_w * 0.35)
        center_end = int(content_w * 0.65)
        center_proj = v_proj[center_start:center_end]

        if len(center_proj) == 0:
            return 1, 0.0

        # 빈 공간 임계값 (전체 높이의 5% 이하면 빈 공간)
        content_h = content.shape[0]
        gap_threshold = content_h * 0.05

        # 연속된 빈 공간 찾기
        is_gap = center_proj < gap_threshold
        gap_runs = self._find_runs(is_gap, True)

        # 20px 이상의 빈 공간이 있으면 2단
        min_gap_width = 20
        for start, length in gap_runs:
            if length >= min_gap_width:
                actual_gap_center = center_start + start + length // 2
                return 2, float(length)

        return 1, 0.0

    def _find_runs(self, arr: np.ndarray, value: bool) -> list[tuple[int, int]]:
        """연속된 값의 시작 위치와 길이 반환"""
        runs = []
        in_run = False
        start = 0

        for i, v in enumerate(arr):
            if v == value and not in_run:
                in_run = True
                start = i
            elif v != value and in_run:
                in_run = False
                runs.append((start, i - start))

        if in_run:
            runs.append((start, len(arr) - start))

        return runs

    def _estimate_line_spacing(self, regions: list[Region]) -> float:
        """텍스트 영역 간 줄 간격 추정"""
        if len(regions) < 2:
            return 1.0

        text_regions = [
            r for r in regions if r.region_type == RegionType.TEXT
        ]

        if len(text_regions) < 2:
            return 1.0

        # 인접 텍스트 영역 간 간격 계산
        spacings = []
        sorted_regions = sorted(text_regions, key=lambda r: r.bbox.y)

        for i in range(len(sorted_regions) - 1):
            curr = sorted_regions[i]
            next_r = sorted_regions[i + 1]

            gap = next_r.bbox.y - curr.bbox.y2
            height = curr.bbox.height

            if height > 0 and gap > 0:
                ratio = gap / height
                if 0.1 < ratio < 5.0:  # 합리적 범위
                    spacings.append(ratio)

        if spacings:
            return round(sum(spacings) / len(spacings), 2)

        return 1.0

    def _analyze_region(
        self,
        region: Region,
        num_columns: int,
        margins: Margins,
        page_width: int
    ) -> LayoutRegion:
        """개별 영역의 레이아웃 정보 분석"""
        layout_region = LayoutRegion(
            bbox=region.bbox,
            region_type=region.region_type
        )

        # 단 배정
        if num_columns == 2:
            content_width = page_width - margins.left - margins.right
            mid_x = margins.left + content_width // 2
            region_center_x = region.bbox.center[0]

            if region.bbox.width > content_width * 0.7:
                layout_region.column = 0  # 전체폭
            elif region_center_x < mid_x:
                layout_region.column = 1  # 왼쪽단
            else:
                layout_region.column = 2  # 오른쪽단
        else:
            layout_region.column = 0

        # 문제 번호 인식 (텍스트가 있는 경우)
        if region.content and region.region_type == RegionType.TEXT:
            layout_region.problem_number = self._detect_problem_number(region.content)
            layout_region.is_choice_group = self._detect_choice_group(region.content)

        # 들여쓰기 수준 계산
        indent_px = region.bbox.x - margins.left
        if indent_px > 30:
            layout_region.indent_level = min(indent_px // 30, 3)

        return layout_region

    def _detect_problem_number(self, text: str) -> Optional[str]:
        """텍스트에서 문제 번호 감지"""
        text = text.strip()
        for pattern in self.PROBLEM_NUMBER_PATTERNS:
            match = pattern.match(text)
            if match:
                return match.group(1)
        return None

    def _detect_choice_group(self, text: str) -> bool:
        """텍스트가 보기 그룹인지 감지"""
        for pattern in self.CHOICE_PATTERNS:
            matches = pattern.findall(text)
            if len(matches) >= 2:  # 2개 이상 보기 패턴이 있으면
                return True
        return False
