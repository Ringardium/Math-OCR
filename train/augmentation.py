"""
데이터 증강
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class Compose:
    """여러 변환을 순차적으로 적용"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip:
    """랜덤 수평 뒤집기"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(x, dims=[-1])
        return x


class RandomRotation:
    """랜덤 회전"""

    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.degrees, self.degrees)
        angle_rad = angle * np.pi / 180

        # 회전 행렬
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Affine 변환 (배치 차원 추가)
        x = x.unsqueeze(0)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype).unsqueeze(0)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='border')

        return x.squeeze(0)


class RandomAffine:
    """랜덤 어파인 변환 (이동, 스케일, 기울임)"""

    def __init__(
        self,
        translate: tuple = (0.1, 0.1),
        scale: tuple = (0.9, 1.1),
        shear: float = 10
    ):
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 랜덤 파라미터
        tx = random.uniform(-self.translate[0], self.translate[0])
        ty = random.uniform(-self.translate[1], self.translate[1])
        s = random.uniform(self.scale[0], self.scale[1])
        shear_rad = random.uniform(-self.shear, self.shear) * np.pi / 180

        # 변환 행렬
        cos_sh = np.cos(shear_rad)
        sin_sh = np.sin(shear_rad)

        x = x.unsqueeze(0)
        theta = torch.tensor([
            [s * cos_sh, -sin_sh, tx],
            [s * sin_sh, cos_sh, ty]
        ], dtype=x.dtype).unsqueeze(0)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='border')

        return x.squeeze(0)


class RandomNoise:
    """랜덤 가우시안 노이즈"""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            noise = torch.randn_like(x) * self.std
            x = x + noise
            x = torch.clamp(x, 0, 1)
        return x


class RandomBrightness:
    """랜덤 밝기 조절"""

    def __init__(self, factor: float = 0.2):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            delta = random.uniform(-self.factor, self.factor)
            x = x + delta
            x = torch.clamp(x, 0, 1)
        return x


class RandomContrast:
    """랜덤 대비 조절"""

    def __init__(self, factor: tuple = (0.8, 1.2)):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            f = random.uniform(self.factor[0], self.factor[1])
            mean = x.mean()
            x = (x - mean) * f + mean
            x = torch.clamp(x, 0, 1)
        return x


class RandomErasing:
    """랜덤 영역 지우기"""

    def __init__(self, p: float = 0.3, scale: tuple = (0.02, 0.1)):
        self.p = p
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            _, h, w = x.shape
            area = h * w
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(0.5, 2.0)

            eh = int(np.sqrt(erase_area * aspect_ratio))
            ew = int(np.sqrt(erase_area / aspect_ratio))

            if eh < h and ew < w:
                y = random.randint(0, h - eh)
                x_pos = random.randint(0, w - ew)
                x[:, y:y+eh, x_pos:x_pos+ew] = random.random()

        return x


class Normalize:
    """정규화"""

    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Denormalize:
    """역정규화"""

    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def get_train_transforms() -> Compose:
    """학습용 변환"""
    return Compose([
        RandomHorizontalFlip(p=0.3),
        RandomRotation(degrees=10),
        RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        RandomNoise(std=0.03),
        RandomBrightness(factor=0.15),
        RandomContrast(factor=(0.9, 1.1)),
        RandomErasing(p=0.2),
        Normalize(mean=0.5, std=0.5),
    ])


def get_val_transforms() -> Compose:
    """검증/테스트용 변환"""
    return Compose([
        Normalize(mean=0.5, std=0.5),
    ])
