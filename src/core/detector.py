"""
영역 감지 모듈
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .document import Page, Region, BoundingBox
from ..config import RegionType


@dataclass
class DetectionResult:
    """감지 결과"""
    bbox: BoundingBox
    region_type: RegionType
    confidence: float
    metadata: dict = None


class BaseDetector(ABC):
    """영역 감지 기본 클래스"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """이미지에서 영역 감지"""
        pass


class ContourBasedDetector(BaseDetector):
    """컨투어 기반 영역 감지 (기본 구현)"""

    def __init__(
        self,
        min_area: int = 100,
        merge_distance: int = 20
    ):
        if cv2 is None:
            raise ImportError("OpenCV가 필요합니다: pip install opencv-python")

        self.min_area = min_area
        self.merge_distance = merge_distance

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 모폴로지 연산으로 텍스트 영역 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # 컨투어 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 최소 크기 필터
            if w * h < self.min_area:
                continue

            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            # 영역 유형 추정 (간단한 휴리스틱)
            region_type = self._classify_region(image, bbox)

            results.append(DetectionResult(
                bbox=bbox,
                region_type=region_type,
                confidence=0.8
            ))

        # 영역 병합
        results = self._merge_overlapping(results)

        return results

    def _classify_region(self, image: np.ndarray, bbox: BoundingBox) -> RegionType:
        """영역 유형 분류 (기본 휴리스틱)"""
        roi = image[bbox.y:bbox.y2, bbox.x:bbox.x2]

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi

        # 텍스트 밀도 계산
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        density = np.sum(binary > 0) / binary.size

        # 가로세로 비율
        aspect_ratio = bbox.width / max(bbox.height, 1)

        # 간단한 분류 규칙
        if density < 0.05:
            return RegionType.IMAGE
        elif aspect_ratio > 3 and density > 0.3:
            return RegionType.TABLE
        else:
            return RegionType.TEXT

    def _merge_overlapping(self, results: list[DetectionResult]) -> list[DetectionResult]:
        """겹치는 영역 병합"""
        if not results:
            return results

        # 간단한 병합 (추후 개선 가능)
        merged = []
        used = set()

        for i, r1 in enumerate(results):
            if i in used:
                continue

            current_bbox = r1.bbox
            current_type = r1.region_type

            for j, r2 in enumerate(results[i+1:], i+1):
                if j in used:
                    continue

                # 겹침 확인
                if self._boxes_overlap(current_bbox, r2.bbox):
                    # 병합
                    current_bbox = self._merge_boxes(current_bbox, r2.bbox)
                    used.add(j)

            merged.append(DetectionResult(
                bbox=current_bbox,
                region_type=current_type,
                confidence=r1.confidence
            ))

        return merged

    def _boxes_overlap(self, b1: BoundingBox, b2: BoundingBox) -> bool:
        """두 박스가 겹치는지 확인"""
        return not (
            b1.x2 + self.merge_distance < b2.x or
            b2.x2 + self.merge_distance < b1.x or
            b1.y2 + self.merge_distance < b2.y or
            b2.y2 + self.merge_distance < b1.y
        )

    def _merge_boxes(self, b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
        """두 박스 병합"""
        x = min(b1.x, b2.x)
        y = min(b1.y, b2.y)
        x2 = max(b1.x2, b2.x2)
        y2 = max(b1.y2, b2.y2)
        return BoundingBox.from_xyxy(x, y, x2, y2)


class RegionDetector:
    """통합 영역 감지기"""

    def __init__(self, detector: Optional[BaseDetector] = None):
        self.detector = detector or ContourBasedDetector()

    def detect_regions(self, page: Page) -> list[Region]:
        """페이지에서 영역 감지"""
        detection_results = self.detector.detect(page.image)

        regions = []
        for result in detection_results:
            # 영역 이미지 추출
            roi = page.image[
                result.bbox.y:result.bbox.y2,
                result.bbox.x:result.bbox.x2
            ].copy()

            region = Region(
                bbox=result.bbox,
                region_type=result.region_type,
                image=roi,
                confidence=result.confidence,
                metadata=result.metadata or {}
            )
            regions.append(region)

        return regions

    def process_page(self, page: Page) -> None:
        """페이지의 영역을 감지하고 추가"""
        regions = self.detect_regions(page)
        for region in regions:
            page.add_region(region)
        page.sort_regions_by_position()
