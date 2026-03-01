"""
GUI 리소스 - 색상, 아이콘, 상수
"""

# 영역 유형별 오버레이 색상 (R, G, B, Alpha)
REGION_COLORS = {
    "text": (66, 133, 244, 80),     # 파란색
    "math": (234, 67, 53, 80),      # 빨간색
    "table": (52, 168, 83, 80),     # 초록색
    "image": (251, 188, 4, 80),     # 노란색
    "unknown": (128, 128, 128, 80), # 회색
}

# 영역 유형별 테두리 색상
REGION_BORDER_COLORS = {
    "text": (66, 133, 244, 200),
    "math": (234, 67, 53, 200),
    "table": (52, 168, 83, 200),
    "image": (251, 188, 4, 200),
    "unknown": (128, 128, 128, 200),
}

# 지원 파일 형식
SUPPORTED_FILE_FILTERS = (
    "지원 파일 (*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
    "PDF 파일 (*.pdf);;"
    "이미지 파일 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
    "모든 파일 (*)"
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
}

# 앱 정보
APP_NAME = "YM-OCR"
APP_TITLE = "YM-OCR - 수학 문서 OCR 변환기"
APP_VERSION = "0.2.0"
