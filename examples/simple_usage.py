"""
간단한 사용 예시
"""

from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic():
    """기본 사용법"""
    from src.pipeline import SimplePipeline

    # PDF를 Word로 변환
    result = SimplePipeline.convert(
        input_path="document.pdf",
        output_path="output.docx"
    )

    if result.success:
        print(f"변환 완료: {result.output_path}")
    else:
        print(f"변환 실패: {result.errors}")


def example_printed_only():
    """인쇄체만 인식"""
    from src.pipeline import SimplePipeline

    result = SimplePipeline.convert(
        input_path="document.pdf",
        text_type="printed"  # 인쇄체만
    )

    print(f"인쇄체 텍스트 영역: {result.stats.get('text_regions', 0)}")


def example_handwritten_only():
    """손글씨만 인식"""
    from src.pipeline import SimplePipeline

    result = SimplePipeline.convert(
        input_path="document.pdf",
        text_type="handwritten"  # 손글씨만
    )

    print(f"손글씨 텍스트 영역: {result.stats.get('text_regions', 0)}")


def example_custom_pipeline():
    """커스텀 파이프라인"""
    from src.pipeline import OCRPipeline
    from src.config import AppConfig, OCRConfig, TextType
    from src.ocr.text_ocr import TextOCR
    from src.exporters.docx_exporter import DocxExporter

    # 커스텀 설정
    config = AppConfig(
        ocr=OCRConfig(
            text_type=TextType.BOTH,
            languages=["ko", "en", "ja"],  # 일본어 추가
            enable_math_ocr=True,
            enable_table_ocr=True,
            confidence_threshold=0.7
        )
    )

    # 커스텀 OCR 엔진
    text_ocr = TextOCR(languages=["ko", "en", "ja"])

    # 커스텀 내보내기
    exporter = DocxExporter(
        font_name="나눔고딕",
        font_size=12
    )

    # 파이프라인 생성
    pipeline = OCRPipeline(
        config=config,
        text_ocr=text_ocr,
        exporter=exporter
    )

    # 변환 실행
    result = pipeline.process("document.pdf", "output.docx")

    return result


def example_with_progress():
    """진행 상황 콜백"""
    from src.pipeline import OCRPipeline
    from src.config import AppConfig

    def on_progress(message: str, progress: float):
        bar = "=" * int(progress * 20)
        print(f"\r[{bar:<20}] {progress*100:.0f}% {message}", end="")
        if progress >= 1.0:
            print()

    pipeline = OCRPipeline(
        config=AppConfig(),
        progress_callback=on_progress
    )

    result = pipeline.process("document.pdf")
    return result


def example_batch_processing():
    """배치 처리"""
    from src.pipeline import SimplePipeline
    from pathlib import Path

    input_folder = Path("input_documents")
    output_folder = Path("output_documents")
    output_folder.mkdir(exist_ok=True)

    for pdf_file in input_folder.glob("*.pdf"):
        output_file = output_folder / pdf_file.with_suffix(".docx").name

        print(f"처리 중: {pdf_file.name}")

        result = SimplePipeline.convert(
            input_path=str(pdf_file),
            output_path=str(output_file)
        )

        if result.success:
            print(f"  -> 완료: {output_file}")
        else:
            print(f"  -> 실패: {result.errors}")


if __name__ == "__main__":
    print("사용 예시 파일입니다.")
    print("실제 파일로 테스트하려면 파일 경로를 수정하세요.")
