"""
인쇄체 데이터 생성기 (폰트 기반)
"""

import random
import string
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Pillow가 필요합니다: pip install Pillow")


# 한글 샘플 텍스트
KOREAN_TEXTS = [
    "가나다라마바사아자차카타파하",
    "수학의 정석",
    "미적분학 기초",
    "함수의 극한",
    "삼각함수의 미분",
    "정적분의 활용",
    "확률과 통계",
    "벡터의 연산",
    "행렬과 행렬식",
    "공간도형과 좌표",
    "f(x) = 2x + 1",
    "lim x→0",
    "∫ sin(x) dx",
    "Σ n = n(n+1)/2",
    "√(a² + b²)",
    "log₂ 8 = 3",
    "π ≈ 3.14159",
    "θ = 45°",
    "Δx → 0",
    "∞",
]

# 영어/숫자 샘플 텍스트
ENGLISH_TEXTS = [
    "Mathematics",
    "Calculus",
    "Integral",
    "Derivative",
    "Function",
    "Equation",
    "Solution",
    "Variable",
    "Constant",
    "Limit",
    "f(x) = ax + b",
    "y = mx + c",
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "e^x",
    "ln(x)",
    "∫f(x)dx",
    "d/dx",
    "∑",
]


class PrintedTextGenerator:
    """인쇄체 텍스트 이미지 생성기"""

    def __init__(
        self,
        output_dir: Path,
        image_size: int = 128,
        fonts: Optional[List[str]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

        # 사용할 폰트 목록
        self.fonts = fonts or self._get_default_fonts()
        self.available_fonts = self._load_fonts()

        if not self.available_fonts:
            print("경고: 사용 가능한 폰트가 없습니다. 기본 폰트를 사용합니다.")

    def _get_default_fonts(self) -> List[str]:
        """기본 폰트 경로 목록"""
        # Windows 폰트 경로
        windows_fonts = [
            "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
            "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 Bold
            "C:/Windows/Fonts/batang.ttc",      # 바탕
            "C:/Windows/Fonts/gulim.ttc",       # 굴림
            "C:/Windows/Fonts/arial.ttf",       # Arial
            "C:/Windows/Fonts/times.ttf",       # Times New Roman
            "C:/Windows/Fonts/cour.ttf",        # Courier New
            "C:/Windows/Fonts/verdana.ttf",     # Verdana
            "C:/Windows/Fonts/tahoma.ttf",      # Tahoma
            "C:/Windows/Fonts/calibri.ttf",     # Calibri
        ]

        # Linux 폰트 경로
        linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]

        # macOS 폰트 경로
        mac_fonts = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/Arial.ttf",
        ]

        return windows_fonts + linux_fonts + mac_fonts

    def _load_fonts(self) -> List[tuple]:
        """사용 가능한 폰트 로드"""
        available = []

        for font_path in self.fonts:
            if Path(font_path).exists():
                try:
                    # 다양한 크기로 로드
                    for size in [16, 20, 24, 28, 32, 36, 40]:
                        font = ImageFont.truetype(font_path, size)
                        available.append((font_path, size, font))
                except Exception as e:
                    print(f"폰트 로드 실패: {font_path}, 오류: {e}")

        return available

    def generate_image(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        add_noise: bool = True
    ) -> Image.Image:
        """텍스트 이미지 생성"""
        # 흰 배경 이미지 생성
        img = Image.new('L', (self.image_size, self.image_size), color=255)
        draw = ImageDraw.Draw(img)

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 텍스트가 이미지보다 크면 잘라내기
        if text_width > self.image_size * 0.9:
            text = text[:len(text)//2]
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # 중앙에 배치 (약간의 랜덤 오프셋)
        x = (self.image_size - text_width) // 2 + random.randint(-5, 5)
        y = (self.image_size - text_height) // 2 + random.randint(-5, 5)

        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=0)

        # 노이즈 추가
        if add_noise:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """이미지에 노이즈 추가"""
        img_array = np.array(img, dtype=np.float32)

        # 가우시안 노이즈
        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(2, 8), img_array.shape)
            img_array = img_array + noise
            img_array = np.clip(img_array, 0, 255)

        # 밝기 변화
        if random.random() < 0.3:
            brightness = random.uniform(-20, 20)
            img_array = img_array + brightness
            img_array = np.clip(img_array, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def generate_dataset(
        self,
        num_samples: int = 10000,
        texts: Optional[List[str]] = None
    ) -> int:
        """데이터셋 생성"""
        if not self.available_fonts:
            # 기본 폰트 사용
            font = ImageFont.load_default()
            self.available_fonts = [("default", 12, font)]

        if texts is None:
            texts = KOREAN_TEXTS + ENGLISH_TEXTS

        print(f"인쇄체 데이터 생성 시작...")
        print(f"  - 출력 경로: {self.output_dir}")
        print(f"  - 생성 개수: {num_samples}")
        print(f"  - 사용 폰트: {len(self.available_fonts)}개")

        generated = 0

        for i in range(num_samples):
            # 랜덤 텍스트 선택
            text = random.choice(texts)

            # 랜덤 폰트 선택
            font_path, font_size, font = random.choice(self.available_fonts)

            # 이미지 생성
            try:
                img = self.generate_image(text, font)

                # 저장
                filename = f"printed_{i:06d}.png"
                img.save(self.output_dir / filename)
                generated += 1

                if (i + 1) % 1000 == 0:
                    print(f"  진행: {i+1}/{num_samples}")

            except Exception as e:
                print(f"이미지 생성 실패: {e}")

        print(f"생성 완료: {generated}개")
        return generated


def generate_printed_data(
    output_dir: str = "data/printed",
    num_samples: int = 10000,
    image_size: int = 128
):
    """인쇄체 데이터 생성 함수"""
    generator = PrintedTextGenerator(
        output_dir=Path(output_dir),
        image_size=image_size
    )

    return generator.generate_dataset(num_samples=num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="인쇄체 데이터 생성")
    parser.add_argument("--output", type=str, default="data/printed", help="출력 경로")
    parser.add_argument("--num-samples", type=int, default=10000, help="생성 개수")
    parser.add_argument("--image-size", type=int, default=128, help="이미지 크기")

    args = parser.parse_args()

    generate_printed_data(
        output_dir=args.output,
        num_samples=args.num_samples,
        image_size=args.image_size
    )
