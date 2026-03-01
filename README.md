# YM-OCR

수학 문서 OCR 변환 시스템

PDF/이미지에서 수학 수식, 텍스트, 표, 이미지를 인식하여 Word 문서로 변환합니다.

## 주요 기능

- **텍스트 OCR**: EasyOCR 기반 다국어 텍스트 인식
- **수식 OCR**: pix2tex 기반 LaTeX 변환
- **표 인식**: 표 구조 감지 및 Word 표 변환
- **이미지 추출**: 원본 이미지 그대로 유지
- **손글씨/인쇄체 분류**: 선택적 OCR 지원

## 설치

```bash
# 기본 패키지
pip install -r requirements.txt

# 학습용 패키지 (선택)
pip install -r train/requirements.txt
```

## 빠른 시작

### CLI 사용

```bash
# 기본 변환
python main.py document.pdf

# 출력 파일 지정
python main.py document.pdf -o output.docx

# 인쇄체만 인식
python main.py document.pdf --text-type printed

# 손글씨만 인식
python main.py document.pdf --text-type handwritten
```

### Python 코드

```python
from src.pipeline import SimplePipeline

# 기본 변환
result = SimplePipeline.convert("input.pdf", "output.docx")

# 옵션 지정
result = SimplePipeline.convert(
    "input.pdf",
    text_type="printed",      # "both", "printed", "handwritten"
    languages=["ko", "en"],
    enable_math=True,
    enable_table=True
)

if result.success:
    print(f"변환 완료: {result.output_path}")
```

## 프로젝트 구조

```
YM-OCR/
├── main.py                 # CLI 엔트리포인트
├── requirements.txt        # 의존성
├── src/
│   ├── config.py          # 설정 관리
│   ├── pipeline.py        # 전체 파이프라인
│   ├── core/
│   │   ├── document.py    # 문서 데이터 모델
│   │   ├── loader.py      # PDF/이미지 로더
│   │   ├── detector.py    # 영역 감지
│   │   └── builder.py     # 문서 빌더
│   ├── ocr/
│   │   ├── text_ocr.py    # 텍스트 OCR
│   │   ├── math_ocr.py    # 수식 OCR
│   │   └── table_ocr.py   # 표 OCR
│   ├── classifiers/
│   │   └── text_classifier.py  # 손글씨/인쇄체 분류
│   └── exporters/
│       └── docx_exporter.py    # Word 내보내기
├── train/                  # 학습 모듈
│   ├── train.py           # CNN 학습
│   ├── train_ml.py        # ML 학습
│   └── ...
├── examples/              # 사용 예시
└── tests/                 # 테스트
```

## CLI 옵션

```
python main.py [옵션] <입력파일>

필수:
  입력파일              PDF 또는 이미지 파일

옵션:
  -o, --output         출력 파일 경로
  --text-type          인식할 텍스트 유형 (both/printed/handwritten)
  --languages          OCR 언어 (기본: ko en)
  --no-math            수식 OCR 비활성화
  --no-table           표 OCR 비활성화
  --no-gpu             GPU 사용 안 함
  --debug              디버그 모드
```

## 손글씨/인쇄체 분류기 학습

### 데이터 준비

```
data/
├── handwritten/    # 손글씨 이미지
│   ├── img001.png
│   └── ...
└── printed/        # 인쇄체 이미지
    ├── img001.png
    └── ...
```

### 인쇄체 데이터 자동 생성

```bash
python -m train.generate_printed --output data/printed --num-samples 10000
```

### CNN 학습

```bash
# 기본 학습
python -m train.train --data-dir data --epochs 50

# 큰 모델
python -m train.train --data-dir data --model large

# ResNet 스타일
python -m train.train --data-dir data --model resnet
```

### ML 학습 (적은 데이터용)

```bash
# HOG + SVM (추천)
python -m train.train_ml --data-dir data

# 특징 선택
python -m train.train_ml --data-dir data --features hog      # HOG
python -m train.train_ml --data-dir data --features lbp      # LBP
python -m train.train_ml --data-dir data --features combined # 모두

# 분류기 선택
python -m train.train_ml --data-dir data --classifier svm    # SVM
python -m train.train_ml --data-dir data --classifier rf     # Random Forest

# 여러 분류기 비교
python -m train.train_ml --data-dir data --compare

# 하이퍼파라미터 튜닝
python -m train.train_ml --data-dir data --classifier svm --tune
```

### 학습된 모델 사용

```python
# CNN 모델
from src.classifiers import CNNHandwritingClassifier
classifier = CNNHandwritingClassifier(model_path="models/handwriting_classifier.pth")

# ML 모델
from train.ml_classifier import MLHandwritingClassifier
classifier = MLHandwritingClassifier(model_path="models/handwriting_ml_svm.pkl")

# 분류
result = classifier.classify(image)
print(result.text_class)   # PRINTED 또는 HANDWRITTEN
print(result.confidence)   # 신뢰도
```

## CNN vs ML 비교

| | CNN | ML (HOG+SVM) |
|---|---|---|
| 필요 데이터 | 10,000개+ | 1,000~5,000개 |
| 학습 시간 | 수 시간 | 수 분 |
| GPU | 필요 | 불필요 |
| 정확도 | 95%+ | 85~92% |

## 사용 모델

| 기능 | 라이브러리 | 모델 |
|------|-----------|------|
| 텍스트 OCR | EasyOCR | CRAFT + CRNN |
| 수식 OCR | pix2tex | ViT + Transformer |
| 표 인식 | img2table | OpenCV + Tesseract |
| 손글씨 분류 (ML) | scikit-learn | HOG + SVM |
| 손글씨 분류 (CNN) | PyTorch | Custom CNN |

## 데이터셋 추천

### 손글씨 데이터
- **AI Hub 한글 손글씨**: aihub.or.kr (한국어, 무료)
- **IAM Handwriting**: fki.inf.unibe.ch (영어)
- **EMNIST**: nist.gov (영어/숫자)

### 인쇄체 데이터
- 폰트 기반 자동 생성 (train/generate_printed.py)
- SynthText: github.com/ankush-me/SynthText

## 요구사항

- Python 3.9+
- PyTorch 1.9+ (CNN 학습 시)
- CUDA (GPU 사용 시)

## 라이선스

MIT License
