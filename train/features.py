"""
특징 추출 모듈
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.feature import hog, local_binary_pattern
    from skimage.filters import gabor
except ImportError:
    hog = None
    local_binary_pattern = None
    gabor = None


class BaseFeatureExtractor(ABC):
    """특징 추출 기본 클래스"""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 특징 벡터 추출"""
        pass

    @property
    @abstractmethod
    def feature_size(self) -> int:
        """특징 벡터 크기"""
        pass


class HOGExtractor(BaseFeatureExtractor):
    """
    HOG (Histogram of Oriented Gradients) 특징 추출

    손글씨/인쇄체 분류에 효과적
    - 엣지 방향 분포를 캡처
    - 획의 방향성 차이 감지
    """

    def __init__(
        self,
        image_size: tuple = (64, 64),
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2)
    ):
        if hog is None:
            raise ImportError("scikit-image가 필요합니다: pip install scikit-image")

        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # 특징 크기 계산
        self._feature_size = self._calculate_feature_size()

    def _calculate_feature_size(self) -> int:
        """HOG 특징 벡터 크기 계산"""
        cells_x = self.image_size[1] // self.pixels_per_cell[1]
        cells_y = self.image_size[0] // self.pixels_per_cell[0]
        blocks_x = cells_x - self.cells_per_block[1] + 1
        blocks_y = cells_y - self.cells_per_block[0] + 1
        return (blocks_x * blocks_y *
                self.cells_per_block[0] * self.cells_per_block[1] *
                self.orientations)

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        """HOG 특징 추출"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 리사이즈
        image = cv2.resize(image, self.image_size)

        # HOG 추출
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True
        )

        return features.astype(np.float32)


class LBPExtractor(BaseFeatureExtractor):
    """
    LBP (Local Binary Pattern) 특징 추출

    텍스처 분석에 효과적
    - 로컬 텍스처 패턴 캡처
    - 손글씨의 불규칙한 텍스처 감지
    """

    def __init__(
        self,
        image_size: tuple = (64, 64),
        radius: int = 3,
        n_points: int = 24,
        n_bins: int = 64
    ):
        if local_binary_pattern is None:
            raise ImportError("scikit-image가 필요합니다: pip install scikit-image")

        self.image_size = image_size
        self.radius = radius
        self.n_points = n_points
        self.n_bins = n_bins

    @property
    def feature_size(self) -> int:
        return self.n_bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        """LBP 특징 추출"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 리사이즈
        image = cv2.resize(image, self.image_size)

        # LBP 계산
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')

        # 히스토그램
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.n_bins,
            range=(0, self.n_bins)
        )

        # 정규화
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)

        return hist


class StatisticalExtractor(BaseFeatureExtractor):
    """
    통계적 특징 추출

    간단하지만 효과적
    - 픽셀 분포 통계
    - 획 특성 분석
    """

    def __init__(self, image_size: tuple = (64, 64)):
        if cv2 is None:
            raise ImportError("OpenCV가 필요합니다: pip install opencv-python")

        self.image_size = image_size

    @property
    def feature_size(self) -> int:
        return 32  # 고정된 특징 수

    def extract(self, image: np.ndarray) -> np.ndarray:
        """통계적 특징 추출"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 리사이즈
        image = cv2.resize(image, self.image_size)

        features = []

        # 1. 기본 통계
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.median(image))
        features.append(np.min(image))
        features.append(np.max(image))

        # 2. 이진화 후 통계
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        features.append(np.sum(binary > 0) / binary.size)  # 텍스트 밀도

        # 3. 엣지 특징
        edges = cv2.Canny(image, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)  # 엣지 밀도

        # 4. 수평/수직 프로젝션
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        features.append(np.std(h_proj))  # 수평 변동
        features.append(np.std(v_proj))  # 수직 변동
        features.append(np.mean(h_proj))
        features.append(np.mean(v_proj))

        # 5. 획 두께 (거리 변환)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if dist.max() > 0:
            features.append(np.mean(dist[dist > 0]))
            features.append(np.std(dist[dist > 0]))
            features.append(np.max(dist))
        else:
            features.extend([0, 0, 0])

        # 6. 컨투어 특징
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours))  # 컨투어 수

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            features.append(np.mean(areas))
            features.append(np.std(areas))
            features.append(np.mean(perimeters))
            features.append(np.std(perimeters))

            # 복잡도
            complexities = []
            for area, peri in zip(areas, perimeters):
                if peri > 0:
                    complexities.append(4 * np.pi * area / (peri * peri))
            if complexities:
                features.append(np.mean(complexities))
                features.append(np.std(complexities))
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        # 7. 모멘트 특징
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments[:7])

        # 크기 맞추기
        features = np.array(features[:self.feature_size], dtype=np.float32)
        if len(features) < self.feature_size:
            features = np.pad(features, (0, self.feature_size - len(features)))

        return features


class CombinedExtractor(BaseFeatureExtractor):
    """
    여러 특징 추출기 조합

    HOG + LBP + 통계 특징을 결합
    """

    def __init__(self, image_size: tuple = (64, 64)):
        self.extractors = [
            HOGExtractor(image_size=image_size),
            LBPExtractor(image_size=image_size),
            StatisticalExtractor(image_size=image_size)
        ]

    @property
    def feature_size(self) -> int:
        return sum(e.feature_size for e in self.extractors)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """모든 특징 결합"""
        features = []
        for extractor in self.extractors:
            feat = extractor.extract(image)
            features.append(feat)
        return np.concatenate(features)


def get_extractor(name: str, image_size: tuple = (64, 64)) -> BaseFeatureExtractor:
    """이름으로 특징 추출기 가져오기"""
    extractors = {
        "hog": HOGExtractor,
        "lbp": LBPExtractor,
        "statistical": StatisticalExtractor,
        "combined": CombinedExtractor
    }

    if name not in extractors:
        raise ValueError(f"알 수 없는 특징 추출기: {name}. 가능: {list(extractors.keys())}")

    return extractors[name](image_size=image_size)
