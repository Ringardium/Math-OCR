"""
학습 트레이너
"""

import time
from pathlib import Path
from typing import Optional, Dict, List
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig


class Trainer:
    """모델 학습 트레이너"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 디바이스 설정
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"사용 디바이스: {self.device}")
        self.model = self.model.to(self.device)

        # 손실 함수 & 옵티마이저
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 학습 기록
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

        # 최고 성능
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # 출력 디렉토리
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> tuple:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="학습", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # 통계
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 진행률 표시
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"학습 시작")
        print(f"{'='*60}")
        print(f"에포크: {self.config.num_epochs}")
        print(f"배치 크기: {self.config.batch_size}")
        print(f"학습률: {self.config.learning_rate}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n에포크 [{epoch}/{self.config.num_epochs}]")

            # 학습
            train_loss, train_acc = self.train_epoch()

            # 검증
            val_loss, val_acc = self.validate(self.val_loader)

            # 학습률 스케줄러 업데이트
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 기록
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # 출력
            print(f"  학습 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  검증 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  학습률: {current_lr:.6f}")

            # 최고 성능 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth", epoch, val_acc)
                print(f"  ★ 최고 성능 갱신!")
            else:
                self.patience_counter += 1

            # 주기적 저장
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", epoch, val_acc)

            # 조기 종료
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n조기 종료: {self.config.early_stopping_patience} 에포크 동안 개선 없음")
                break

        # 학습 완료
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"학습 완료!")
        print(f"{'='*60}")
        print(f"총 소요 시간: {elapsed_time/60:.1f}분")
        print(f"최고 검증 정확도: {self.best_val_acc:.2f}% (에포크 {self.best_epoch})")

        # 테스트
        if self.test_loader:
            self.load_checkpoint("best_model.pth")
            test_loss, test_acc = self.validate(self.test_loader)
            print(f"테스트 정확도: {test_acc:.2f}%")
            self.history["test_acc"] = test_acc

        # 학습 기록 저장
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """체크포인트 저장"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_acc": val_acc,
            "config": {
                "image_size": self.config.image_size,
                "learning_rate": self.config.learning_rate,
            }
        }

        save_path = self.config.output_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, filename: str):
        """체크포인트 로드"""
        load_path = self.config.output_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_history(self):
        """학습 기록 저장"""
        history_path = self.config.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def export_model(self, filename: str = "handwriting_classifier.pth"):
        """최종 모델 내보내기 (추론용)"""
        self.load_checkpoint("best_model.pth")

        export_path = self.config.output_dir / filename
        torch.save(self.model.state_dict(), export_path)
        print(f"모델 저장됨: {export_path}")

        return export_path
