"""
데이터셋 로더
"""

import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class HandwritingDataset(Dataset):
    """
    손글씨/인쇄체 분류 데이터셋

    폴더 구조:
        data/
        ├── handwritten/
        │   ├── image1.png
        │   ├── image2.jpg
        │   └── ...
        └── printed/
            ├── image1.png
            ├── image2.jpg
            └── ...
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    # 레이블: 0 = printed, 1 = handwritten
    LABEL_PRINTED = 0
    LABEL_HANDWRITTEN = 1

    def __init__(
        self,
        handwritten_dir: Path,
        printed_dir: Path,
        image_size: int = 128,
        grayscale: bool = True,
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None
    ):
        self.image_size = image_size
        self.grayscale = grayscale
        self.transform = transform

        # 이미지 경로 수집
        self.samples: List[Tuple[Path, int]] = []

        # 손글씨 이미지
        handwritten_images = self._collect_images(handwritten_dir)
        if max_samples_per_class:
            handwritten_images = handwritten_images[:max_samples_per_class]
        for img_path in handwritten_images:
            self.samples.append((img_path, self.LABEL_HANDWRITTEN))

        # 인쇄체 이미지
        printed_images = self._collect_images(printed_dir)
        if max_samples_per_class:
            printed_images = printed_images[:max_samples_per_class]
        for img_path in printed_images:
            self.samples.append((img_path, self.LABEL_PRINTED))

        # 섞기
        random.shuffle(self.samples)

        print(f"데이터셋 로드 완료:")
        print(f"  - 손글씨: {len(handwritten_images)}개")
        print(f"  - 인쇄체: {len(printed_images)}개")
        print(f"  - 총: {len(self.samples)}개")

    def _collect_images(self, directory: Path) -> List[Path]:
        """디렉토리에서 이미지 파일 수집"""
        directory = Path(directory)
        if not directory.exists():
            print(f"경고: 디렉토리가 존재하지 않음: {directory}")
            return []

        images = []
        for ext in self.SUPPORTED_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
            # 하위 폴더도 검색
            images.extend(directory.glob(f"**/*{ext}"))

        return sorted(set(images))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # 이미지 로드
        try:
            image = Image.open(img_path)

            # 그레이스케일 변환
            if self.grayscale:
                image = image.convert('L')
            else:
                image = image.convert('RGB')

            # 리사이즈
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

            # numpy 배열로 변환
            image = np.array(image, dtype=np.float32)

            # 정규화 (0-1)
            image = image / 255.0

            # 채널 차원 추가
            if self.grayscale:
                image = image[np.newaxis, ...]  # (1, H, W)
            else:
                image = image.transpose(2, 0, 1)  # (C, H, W)

            # 텐서로 변환
            image = torch.from_numpy(image)

            # 변환 적용
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"이미지 로드 실패: {img_path}, 오류: {e}")
            # 빈 이미지 반환
            if self.grayscale:
                image = torch.zeros(1, self.image_size, self.image_size)
            else:
                image = torch.zeros(3, self.image_size, self.image_size)
            return image, label

    @staticmethod
    def get_label_name(label: int) -> str:
        """레이블을 이름으로 변환"""
        return "handwritten" if label == 1 else "printed"


def create_dataloaders(
    handwritten_dir: Path,
    printed_dir: Path,
    image_size: int = 128,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    학습/검증/테스트 DataLoader 생성

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 전체 데이터셋 (변환 없이)
    full_dataset = HandwritingDataset(
        handwritten_dir=handwritten_dir,
        printed_dir=printed_dir,
        image_size=image_size,
        grayscale=True,
        transform=None
    )

    # 분할 크기 계산
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 데이터셋 분할
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 변환 래퍼 데이터셋
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    # 변환 적용
    if train_transform:
        train_dataset = TransformDataset(train_dataset, train_transform)
    if val_transform:
        val_dataset = TransformDataset(val_dataset, val_transform)
        test_dataset = TransformDataset(test_dataset, val_transform)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoader 생성:")
    print(f"  - 학습: {len(train_dataset)}개 ({len(train_loader)} 배치)")
    print(f"  - 검증: {len(val_dataset)}개 ({len(val_loader)} 배치)")
    print(f"  - 테스트: {len(test_dataset)}개 ({len(test_loader)} 배치)")

    return train_loader, val_loader, test_loader
