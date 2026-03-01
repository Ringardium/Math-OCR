"""
YM-OCR 빌드 스크립트

사용법:
    python build.py          # 폴더 모드 빌드 (권장, 빠름)
    python build.py --onefile # 단일 exe 빌드 (느림, 용량 큼)
    python build.py --clean   # 빌드 캐시 정리 후 빌드
"""

import subprocess
import sys
import shutil
from pathlib import Path


def clean_build():
    """빌드 캐시 정리"""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for d in dirs_to_clean:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)
            print(f"  삭제: {d}/")

    # .pyc 파일 정리
    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink()


def build_folder():
    """폴더 모드 빌드 (권장)"""
    print("=" * 50)
    print("YM-OCR 빌드 (폴더 모드)")
    print("=" * 50)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "YM-OCR.spec",
        "--noconfirm",
    ]

    print(f"\n실행: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("빌드 완료!")
        print(f"실행 파일: dist/YM-OCR/YM-OCR.exe")
        print("=" * 50)
    else:
        print("\n빌드 실패!")
        sys.exit(1)


def build_onefile():
    """단일 exe 빌드"""
    print("=" * 50)
    print("YM-OCR 빌드 (단일 exe 모드)")
    print("=" * 50)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "YM-OCR",
        "--add-data", "src/gui/styles.qss;src/gui",
        "--add-data", "models;models",
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.sip",
        "--hidden-import", "easyocr",
        "--hidden-import", "texify",
        "--hidden-import", "texify.inference",
        "--hidden-import", "texify.model.model",
        "--hidden-import", "texify.model.processor",
        "--hidden-import", "fitz",
        "--hidden-import", "docx",
        "--hidden-import", "pyhwpx",
        "--hidden-import", "latex2mathml",
        "--hidden-import", "torch",
        "--hidden-import", "cv2",
        "--hidden-import", "img2table",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "main.py",
    ]

    print(f"\n실행 중... (시간이 오래 걸릴 수 있습니다)\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("빌드 완료!")
        print(f"실행 파일: dist/YM-OCR.exe")
        print("=" * 50)
    else:
        print("\n빌드 실패!")
        sys.exit(1)


def main():
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller가 설치되지 않았습니다.")
        print("설치: pip install pyinstaller")
        sys.exit(1)

    args = sys.argv[1:]

    if "--clean" in args:
        print("빌드 캐시 정리 중...")
        clean_build()
        print()

    if "--onefile" in args:
        build_onefile()
    else:
        build_folder()


if __name__ == "__main__":
    main()
