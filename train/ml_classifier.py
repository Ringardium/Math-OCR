"""
ML 기반 손글씨/인쇄체 분류기 (추론용)

src/classifiers/text_classifier.py의 ML 버전
"""

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .features import get_extractor, BaseFeatureExtractor
from .ml_model import BaseMLClassifier


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
    probabilities: dict


class MLHandwritingClassifier:
    """
    ML 기반 손글씨/인쇄체 분류기

    사용법:
        classifier = MLHandwritingClassifier(
            model_path="models/handwriting_ml_svm.pkl",
            feature_extractor="hog"
        )
        result = classifier.classify(image)
        print(result.text_class)  # TextClass.PRINTED 또는 TextClass.HANDWRITTEN
    """

    def __init__(
        self,
        model_path: str,
        feature_extractor: str = "hog",
        image_size: int = 64,
        config_path: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size

        # 설정 파일이 있으면 로드
        if config_path:
            self._load_config(config_path)
        else:
            # 자동으로 설정 파일 찾기
            config_path = self.model_path.with_suffix('.pkl').parent / (
                self.model_path.stem + "_config.json"
            )
            if config_path.exists():
                self._load_config(config_path)
            else:
                self.feature_extractor_name = feature_extractor

        # 특징 추출기 로드
        self.extractor = get_extractor(
            self.feature_extractor_name,
            image_size=(self.image_size, self.image_size)
        )

        # 모델 로드
        self.model = self._load_model()

    def _load_config(self, config_path: str):
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.feature_extractor_name = config.get("feature_extractor", "hog")
        self.image_size = config.get("image_size", 64)

    def _load_model(self) -> BaseMLClassifier:
        """모델 로드"""
        import pickle

        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)

        # BaseMLClassifier 형태로 래핑
        class LoadedClassifier:
            def __init__(self, data):
                self.model = data["model"]
                self.scaler = data["scaler"]

            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)

            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)

        return LoadedClassifier(data)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        import cv2

        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        이미지 분류

        Args:
            image: 입력 이미지 (numpy array)

        Returns:
            ClassificationResult
        """
        # 전처리
        image = self.preprocess(image)

        # 특징 추출
        features = self.extractor.extract(image)
        features = features.reshape(1, -1)

        # 예측
        probs = self.model.predict_proba(features)[0]
        pred_class = self.model.predict(features)[0]

        # 결과 생성
        if pred_class == 1:
            text_class = TextClass.HANDWRITTEN
            confidence = probs[1]
        else:
            text_class = TextClass.PRINTED
            confidence = probs[0]

        return ClassificationResult(
            text_class=text_class,
            confidence=float(confidence),
            probabilities={
                "printed": float(probs[0]),
                "handwritten": float(probs[1])
            }
        )

    def classify_file(self, image_path: str) -> ClassificationResult:
        """파일에서 이미지 로드 후 분류"""
        if Image is None:
            raise ImportError("Pillow가 필요합니다")

        img = Image.open(image_path).convert('L')
        image = np.array(img)

        return self.classify(image)

    def classify_batch(self, images: list) -> list:
        """여러 이미지 일괄 분류"""
        results = []
        for image in images:
            result = self.classify(image)
            results.append(result)
        return results
