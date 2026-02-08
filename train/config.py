"""
학습 설정
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """학습 설정"""

    # 데이터 경로
    data_dir: Path = field(default_factory=lambda: Path("data"))
    handwritten_dir: Optional[Path] = None  # 손글씨 데이터 폴더
    printed_dir: Optional[Path] = None       # 인쇄체 데이터 폴더

    # 모델 저장 경로
    output_dir: Path = field(default_factory=lambda: Path("models"))
    model_name: str = "handwriting_classifier"

    # 이미지 설정
    image_size: int = 128
    grayscale: bool = True

    # 학습 하이퍼파라미터
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 데이터 분할
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 데이터 증강
    use_augmentation: bool = True

    # 기타
    num_workers: int = 4
    seed: int = 42
    use_gpu: bool = True

    # 조기 종료
    early_stopping_patience: int = 10

    # 로깅
    log_interval: int = 10
    save_interval: int = 5

    def __post_init__(self):
        if self.handwritten_dir is None:
            self.handwritten_dir = self.data_dir / "handwritten"
        if self.printed_dir is None:
            self.printed_dir = self.data_dir / "printed"

        # 경로를 Path 객체로 변환
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.handwritten_dir = Path(self.handwritten_dir)
        self.printed_dir = Path(self.printed_dir)
