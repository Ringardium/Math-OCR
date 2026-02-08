"""
파이프라인 테스트
"""

import pytest
from pathlib import Path
import numpy as np

# 테스트용 더미 이미지 생성
def create_test_image(width=200, height=100):
    """테스트용 이미지 생성"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 간단한 텍스트 패턴 추가
    image[20:30, 20:180] = 0
    image[40:50, 20:150] = 0
    image[60:70, 20:170] = 0
    return image


class TestBoundingBox:
    """BoundingBox 테스트"""

    def test_creation(self):
        from src.core.document import BoundingBox

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_properties(self):
        from src.core.document import BoundingBox

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70
        assert bbox.area == 5000
        assert bbox.center == (60, 45)

    def test_from_xyxy(self):
        from src.core.document import BoundingBox

        bbox = BoundingBox.from_xyxy(10, 20, 110, 70)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50


class TestConfig:
    """설정 테스트"""

    def test_default_config(self):
        from src.config import AppConfig, TextType

        config = AppConfig()
        assert config.ocr.text_type == TextType.BOTH
        assert "ko" in config.ocr.languages
        assert config.ocr.enable_math_ocr is True

    def test_custom_config(self):
        from src.config import AppConfig, OCRConfig, TextType

        config = AppConfig(
            ocr=OCRConfig(
                text_type=TextType.PRINTED,
                languages=["en"],
                enable_math_ocr=False
            )
        )
        assert config.ocr.text_type == TextType.PRINTED
        assert config.ocr.languages == ["en"]
        assert config.ocr.enable_math_ocr is False


class TestHandwritingClassifier:
    """손글씨 분류기 테스트"""

    def test_classifier_creation(self):
        from src.classifiers.text_classifier import HandwritingClassifier

        classifier = HandwritingClassifier()
        assert classifier is not None

    def test_classification(self):
        from src.classifiers.text_classifier import HandwritingClassifier, TextClass

        classifier = HandwritingClassifier()
        image = create_test_image()

        result = classifier.classify(image)
        assert result.text_class in [TextClass.PRINTED, TextClass.HANDWRITTEN, TextClass.UNKNOWN]
        assert 0 <= result.confidence <= 1


class TestDocument:
    """문서 모델 테스트"""

    def test_document_creation(self):
        from src.core.document import Document, Page
        import numpy as np

        doc = Document(source_path=Path("test.pdf"))
        page = Page(
            page_number=1,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            width=100,
            height=100
        )
        doc.add_page(page)

        assert doc.page_count == 1
        assert doc.pages[0].page_number == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
