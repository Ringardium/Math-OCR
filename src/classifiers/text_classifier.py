"""
텍스트 분류기 - 손글씨/인쇄체 분류
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class TextClass(Enum):
    """텍스트 분류"""
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """분류 결과"""
    text_class: TextClass
    confidence: float
    probabilities: dict[str, float] = None


class TextClassifier(ABC):
    """텍스트 분류기 기본 클래스"""

    @abstractmethod
    def classify(self, image: np.ndarray) -> ClassificationResult:
        """이미지가 손글씨인지 인쇄체인지 분류"""
        pass


class HandwritingClassifier(TextClassifier):
    """손글씨/인쇄체 분류기 (특징 기반)"""

    def __init__(self):
        if cv2 is None:
            raise ImportError("OpenCV가 필요합니다: pip install opencv-python")

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        이미지 특징을 분석하여 손글씨/인쇄체 분류

        분류 기준:
        - 선의 균일성
        - 문자 간격의 일관성
        - 획의 두께 변화
        - 기울기 변화
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 특징 추출
        features = self._extract_features(binary)

        # 분류 (규칙 기반)
        score = self._compute_handwriting_score(features)

        if score > 0.6:
            text_class = TextClass.HANDWRITTEN
            confidence = score
        elif score < 0.4:
            text_class = TextClass.PRINTED
            confidence = 1 - score
        else:
            text_class = TextClass.UNKNOWN
            confidence = 0.5

        return ClassificationResult(
            text_class=text_class,
            confidence=confidence,
            probabilities={
                "printed": 1 - score,
                "handwritten": score
            }
        )

    def _extract_features(self, binary: np.ndarray) -> dict:
        """특징 추출"""
        features = {}

        # 1. 선 두께 변화
        features["stroke_variation"] = self._compute_stroke_variation(binary)

        # 2. 기울기 변화
        features["slant_variation"] = self._compute_slant_variation(binary)

        # 3. 컨투어 복잡도
        features["contour_complexity"] = self._compute_contour_complexity(binary)

        # 4. 간격 균일성
        features["spacing_uniformity"] = self._compute_spacing_uniformity(binary)

        return features

    def _compute_stroke_variation(self, binary: np.ndarray) -> float:
        """획 두께 변화 계산"""
        # 거리 변환으로 획 두께 추정
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return 0.0

        # 두께 변화의 표준편차
        stroke_widths = dist[dist > 0]
        if len(stroke_widths) == 0:
            return 0.0

        variation = np.std(stroke_widths) / (np.mean(stroke_widths) + 1e-6)
        return min(variation, 1.0)

    def _compute_slant_variation(self, binary: np.ndarray) -> float:
        """기울기 변화 계산"""
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return 0.0

        angles = []
        for contour in contours:
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    angles.append(angle)
                except:
                    pass

        if len(angles) < 2:
            return 0.0

        # 각도 변화의 표준편차
        angle_std = np.std(angles)
        return min(angle_std / 45.0, 1.0)  # 정규화

    def _compute_contour_complexity(self, binary: np.ndarray) -> float:
        """컨투어 복잡도 계산"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        complexities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # 원형도 (circularity)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                complexities.append(1 - circularity)

        if not complexities:
            return 0.0

        return np.mean(complexities)

    def _compute_spacing_uniformity(self, binary: np.ndarray) -> float:
        """간격 균일성 계산"""
        # 수직 프로젝션
        projection = np.sum(binary, axis=0)

        # 공백 영역 찾기
        gaps = []
        in_gap = False
        gap_start = 0

        threshold = np.max(projection) * 0.1

        for i, val in enumerate(projection):
            if val < threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gaps.append(i - gap_start)
                    in_gap = False

        if len(gaps) < 2:
            return 1.0  # 균일하다고 가정

        # 간격의 변동계수 (CV)
        cv = np.std(gaps) / (np.mean(gaps) + 1e-6)
        uniformity = 1 - min(cv, 1.0)

        return uniformity

    def _compute_handwriting_score(self, features: dict) -> float:
        """손글씨 점수 계산"""
        # 가중치
        weights = {
            "stroke_variation": 0.3,
            "slant_variation": 0.25,
            "contour_complexity": 0.2,
            "spacing_uniformity": -0.25  # 균일할수록 인쇄체
        }

        score = 0.5  # 기본값

        for feature_name, weight in weights.items():
            if feature_name in features:
                if weight > 0:
                    score += weight * features[feature_name]
                else:
                    score += weight * (1 - features[feature_name])

        return max(0.0, min(1.0, score))


class CNNHandwritingClassifier(TextClassifier):
    """CNN 기반 손글씨/인쇄체 분류기"""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """모델 로드"""
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms

            # 간단한 CNN 모델 정의
            class SimpleClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, 2)
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            self._model = SimpleClassifier()

            if self.model_path:
                self._model.load_state_dict(torch.load(self.model_path))

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            self._initialized = True

        except ImportError:
            raise ImportError("PyTorch가 필요합니다: pip install torch torchvision")

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """CNN으로 분류"""
        if not self._initialized:
            self.initialize()

        import torch

        # 전처리
        input_tensor = self._transform(image).unsqueeze(0)

        if self.use_gpu and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # 추론
        with torch.no_grad():
            output = self._model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        printed_prob = probs[0].item()
        handwritten_prob = probs[1].item()

        if handwritten_prob > printed_prob:
            text_class = TextClass.HANDWRITTEN
            confidence = handwritten_prob
        else:
            text_class = TextClass.PRINTED
            confidence = printed_prob

        return ClassificationResult(
            text_class=text_class,
            confidence=confidence,
            probabilities={
                "printed": printed_prob,
                "handwritten": handwritten_prob
            }
        )
