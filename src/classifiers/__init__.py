"""
분류기 모듈
"""

from .text_classifier import (
    HandwritingClassifier,
    TextClassifier,
    CNNHandwritingClassifier,
    TextClass,
    ClassificationResult
)

__all__ = [
    "HandwritingClassifier",
    "TextClassifier",
    "CNNHandwritingClassifier",
    "TextClass",
    "ClassificationResult",
]

# ML 분류기 (train 모듈에서 가져오기)
def get_ml_classifier(model_path: str, feature_extractor: str = "hog"):
    """ML 기반 분류기 가져오기"""
    from train.ml_classifier import MLHandwritingClassifier
    return MLHandwritingClassifier(model_path, feature_extractor)
