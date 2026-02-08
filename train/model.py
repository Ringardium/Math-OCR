"""
CNN 분류 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU + MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HandwritingClassifier(nn.Module):
    """
    손글씨/인쇄체 분류 CNN (경량 버전)

    입력: (N, 1, 128, 128) 그레이스케일 이미지
    출력: (N, 2) 클래스 확률 [printed, handwritten]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),      # 128 -> 64
            ConvBlock(32, 64),     # 64 -> 32
            ConvBlock(64, 128),    # 32 -> 16
            ConvBlock(128, 256),   # 16 -> 8
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        """확률 예측"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x):
        """클래스 예측"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class HandwritingClassifierLarge(nn.Module):
    """
    손글씨/인쇄체 분류 CNN (대형 버전)

    더 깊은 네트워크, 더 높은 정확도
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            ConvBlock(1, 64, pool=False),
            ConvBlock(64, 64, pool=True),      # 128 -> 64

            # Block 2
            ConvBlock(64, 128, pool=False),
            ConvBlock(128, 128, pool=True),    # 64 -> 32

            # Block 3
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True),    # 32 -> 16

            # Block 4
            ConvBlock(256, 512, pool=False),
            ConvBlock(512, 512, pool=True),    # 16 -> 8

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block for ResNet-style classifier"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HandwritingResNet(nn.Module):
    """ResNet 스타일 분류기"""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
