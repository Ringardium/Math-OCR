"""
학습 실행 스크립트

사용법:
    python -m train.train --data-dir data --epochs 50
    python -m train.train --data-dir data --model large --batch-size 64
"""

import argparse
import sys
from pathlib import Path

import torch

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.config import TrainConfig
from train.dataset import create_dataloaders
from train.model import HandwritingClassifier, HandwritingClassifierLarge, HandwritingResNet
from train.augmentation import get_train_transforms, get_val_transforms
from train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="손글씨/인쇄체 분류기 학습")

    # 데이터 설정
    parser.add_argument("--data-dir", type=str, default="data",
                        help="데이터 디렉토리 (handwritten, printed 하위 폴더 필요)")
    parser.add_argument("--handwritten-dir", type=str, default=None,
                        help="손글씨 데이터 경로 (기본: data-dir/handwritten)")
    parser.add_argument("--printed-dir", type=str, default=None,
                        help="인쇄체 데이터 경로 (기본: data-dir/printed)")

    # 모델 설정
    parser.add_argument("--model", type=str, default="base",
                        choices=["base", "large", "resnet"],
                        help="모델 종류")
    parser.add_argument("--image-size", type=int, default=128,
                        help="입력 이미지 크기")

    # 학습 설정
    parser.add_argument("--epochs", type=int, default=50, help="에포크 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="가중치 감쇠")

    # 기타
    parser.add_argument("--output-dir", type=str, default="models",
                        help="모델 저장 경로")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="데이터 로더 워커 수")
    parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 안 함")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--no-augment", action="store_true",
                        help="데이터 증강 비활성화")

    return parser.parse_args()


def main():
    args = parse_args()

    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 설정 생성
    config = TrainConfig(
        data_dir=Path(args.data_dir),
        handwritten_dir=Path(args.handwritten_dir) if args.handwritten_dir else None,
        printed_dir=Path(args.printed_dir) if args.printed_dir else None,
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_gpu=not args.no_gpu,
        seed=args.seed,
        use_augmentation=not args.no_augment
    )

    print(f"\n{'='*60}")
    print(f"손글씨/인쇄체 분류기 학습")
    print(f"{'='*60}")
    print(f"데이터 경로:")
    print(f"  - 손글씨: {config.handwritten_dir}")
    print(f"  - 인쇄체: {config.printed_dir}")
    print(f"모델: {args.model}")
    print(f"이미지 크기: {config.image_size}x{config.image_size}")
    print(f"{'='*60}\n")

    # 데이터 경로 확인
    if not config.handwritten_dir.exists():
        print(f"오류: 손글씨 데이터 폴더가 없습니다: {config.handwritten_dir}")
        print(f"다음 구조로 데이터를 준비하세요:")
        print(f"  {config.data_dir}/")
        print(f"  ├── handwritten/")
        print(f"  │   ├── image1.png")
        print(f"  │   └── ...")
        print(f"  └── printed/")
        print(f"      ├── image1.png")
        print(f"      └── ...")
        sys.exit(1)

    if not config.printed_dir.exists():
        print(f"오류: 인쇄체 데이터 폴더가 없습니다: {config.printed_dir}")
        print(f"인쇄체 데이터를 생성하려면:")
        print(f"  python -m train.generate_printed --output {config.printed_dir}")
        sys.exit(1)

    # 변환 설정
    train_transform = get_train_transforms() if config.use_augmentation else get_val_transforms()
    val_transform = get_val_transforms()

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        handwritten_dir=config.handwritten_dir,
        printed_dir=config.printed_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        num_workers=config.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=config.seed
    )

    # 모델 생성
    if args.model == "base":
        model = HandwritingClassifier(num_classes=2)
    elif args.model == "large":
        model = HandwritingClassifierLarge(num_classes=2)
    elif args.model == "resnet":
        model = HandwritingResNet(num_classes=2)
    else:
        raise ValueError(f"알 수 없는 모델: {args.model}")

    # 모델 파라미터 수 출력
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params:,}")

    # 트레이너 생성 및 학습
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # 학습 실행
    history = trainer.train()

    # 최종 모델 내보내기
    export_path = trainer.export_model(f"{config.model_name}.pth")

    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"{'='*60}")
    print(f"최종 모델: {export_path}")
    print(f"\n사용 방법:")
    print(f"  from src.classifiers.text_classifier import CNNHandwritingClassifier")
    print(f"  classifier = CNNHandwritingClassifier(model_path='{export_path}')")
    print(f"  result = classifier.classify(image)")


if __name__ == "__main__":
    main()
