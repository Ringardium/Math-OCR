"""
YM-OCR: 수학 문서 OCR 변환 프로그램

사용법:
    python main.py                                # GUI 실행
    python main.py --gui                          # GUI 실행
    python main.py input.pdf                      # CLI 변환
    python main.py input.pdf -o output.docx       # 출력 파일 지정
    python main.py input.pdf --format hwp         # HWP 출력
    python main.py input.pdf --text-type printed  # 인쇄체만
"""

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="수학 문서 OCR 변환 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                              # GUI 실행
    python main.py document.pdf                 # CLI 변환
    python main.py document.pdf -o result.docx  # 출력 파일 지정
    python main.py document.pdf --format hwp    # HWP 변환
    python main.py document.png --text-type printed
    python main.py document.pdf --no-math --no-table
        """
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="입력 파일 경로 (PDF 또는 이미지). 생략 시 GUI 실행"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="GUI 모드로 실행"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (기본: 입력파일명.docx)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["docx", "hwp"],
        default="docx",
        help="출력 형식 (기본: docx)"
    )

    parser.add_argument(
        "--text-type",
        type=str,
        choices=["both", "printed", "handwritten"],
        default="both",
        help="인식할 텍스트 유형 (기본: both)"
    )

    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["ko", "en"],
        help="OCR 언어 (기본: ko en)"
    )

    parser.add_argument(
        "--no-math",
        action="store_true",
        help="수식 OCR 비활성화"
    )

    parser.add_argument(
        "--no-table",
        action="store_true",
        help="표 OCR 비활성화"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="GPU 사용 안 함"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드"
    )

    return parser


def run_gui():
    """GUI 모드 실행"""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("YM-OCR")
        app.setOrganizationName("YM-OCR")

        window = MainWindow()
        window.show()

        sys.exit(app.exec())

    except ImportError as e:
        print(f"GUI 실행에 필요한 패키지가 없습니다.")
        print(f"설치: pip install PyQt6")
        print(f"\n상세 오류: {e}")
        sys.exit(1)


def run_cli(args):
    """CLI 모드 실행"""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    print(f"YM-OCR v0.2.0")
    print(f"=" * 50)
    print(f"입력 파일: {input_path}")
    print(f"출력 형식: {args.format.upper()}")
    print(f"텍스트 유형: {args.text_type}")
    print(f"언어: {', '.join(args.languages)}")
    print(f"수식 OCR: {'비활성화' if args.no_math else '활성화 (Texify)'}")
    print(f"표 OCR: {'비활성화' if args.no_table else '활성화'}")
    print(f"GPU: {'비활성화' if args.no_gpu else '활성화'}")
    print(f"=" * 50)

    try:
        from src.pipeline import OCRPipeline
        from src.config import AppConfig, OCRConfig, ExportConfig, ExportFormat, TextType

        def progress_callback(message: str, progress: float):
            bar_length = 30
            filled = int(bar_length * progress)
            bar = "\u2588" * filled + "\u2591" * (bar_length - filled)
            print(f"\r[{bar}] {progress*100:.0f}% - {message}", end="", flush=True)
            if progress >= 1.0:
                print()

        text_type_enum = {
            "both": TextType.BOTH,
            "printed": TextType.PRINTED,
            "handwritten": TextType.HANDWRITTEN
        }[args.text_type]

        export_format = {
            "docx": ExportFormat.DOCX,
            "hwp": ExportFormat.HWP,
        }[args.format]

        config = AppConfig(
            ocr=OCRConfig(
                text_type=text_type_enum,
                languages=args.languages,
                enable_math_ocr=not args.no_math,
                enable_table_ocr=not args.no_table,
                use_gpu=not args.no_gpu
            ),
            export=ExportConfig(
                output_format=export_format,
            ),
            debug=args.debug
        )

        pipeline = OCRPipeline(config=config, progress_callback=progress_callback)
        result = pipeline.process(args.input, args.output)

        if result.success:
            print(f"\n변환 완료!")
            print(f"출력 파일: {result.output_path}")
            print(f"\n통계:")
            print(f"  - 페이지 수: {result.stats.get('pages', 0)}")
            print(f"  - 감지된 영역: {result.stats.get('regions', 0)}")
            print(f"  - 텍스트 영역: {result.stats.get('text_regions', 0)}")
            print(f"  - 수식 영역: {result.stats.get('math_regions', 0)}")
            print(f"  - 이미지 영역: {result.stats.get('image_regions', 0)}")
            print(f"  - 표 영역: {result.stats.get('table_regions', 0)}")
        else:
            print(f"\n변환 실패!")
            for error in result.errors:
                print(f"  오류: {error}")
            sys.exit(1)

    except ImportError as e:
        print(f"\n오류: 필요한 패키지가 설치되지 않았습니다.")
        print(f"다음 명령으로 설치해주세요:")
        print(f"  pip install -r requirements.txt")
        print(f"\n상세 오류: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # GUI 모드: --gui 플래그 또는 입력 파일 없을 때
    if args.gui or args.input is None:
        run_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
