"""
모델 평가 스크립트
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.model import HandwritingClassifier, HandwritingClassifierLarge, HandwritingResNet
from train.augmentation import get_val_transforms


class ModelEvaluator:
    """학습된 모델 평가"""

    LABEL_NAMES = {0: "printed", 1: "handwritten"}

    def __init__(
        self,
        model_path: str,
        model_type: str = "base",
        image_size: int = 128,
        use_gpu: bool = True
    ):
        self.image_size = image_size
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # 모델 로드
        if model_type == "base":
            self.model = HandwritingClassifier(num_classes=2)
        elif model_type == "large":
            self.model = HandwritingClassifierLarge(num_classes=2)
        elif model_type == "resnet":
            self.model = HandwritingResNet(num_classes=2)
        else:
            raise ValueError(f"알 수 없는 모델: {model_type}")

        # 가중치 로드
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 변환
        self.transform = get_val_transforms()

        print(f"모델 로드됨: {model_path}")
        print(f"디바이스: {self.device}")

    def preprocess(self, image_path: str) -> torch.Tensor:
        """이미지 전처리"""
        img = Image.open(image_path).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor.unsqueeze(0)  # (1, 1, H, W)

    @torch.no_grad()
    def predict(self, image_path: str) -> Tuple[str, float, dict]:
        """
        단일 이미지 예측

        Returns:
            (예측 레이블, 신뢰도, 확률 딕셔너리)
        """
        input_tensor = self.preprocess(image_path).to(self.device)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()

        return (
            self.LABEL_NAMES[predicted_class],
            confidence,
            {
                "printed": probs[0].item(),
                "handwritten": probs[1].item()
            }
        )

    def evaluate_folder(self, folder_path: str, expected_label: str) -> dict:
        """폴더 내 이미지 평가"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"폴더가 존재하지 않습니다: {folder}")
            return {}

        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if not images:
            print(f"이미지가 없습니다: {folder}")
            return {}

        correct = 0
        total = 0
        confidences = []

        for img_path in images:
            try:
                label, conf, _ = self.predict(str(img_path))
                total += 1

                if label == expected_label:
                    correct += 1
                confidences.append(conf)

            except Exception as e:
                print(f"오류: {img_path}, {e}")

        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy * 100,
            "avg_confidence": avg_confidence * 100
        }

    def evaluate_dataset(self, handwritten_dir: str, printed_dir: str) -> dict:
        """전체 데이터셋 평가"""
        print("\n손글씨 데이터 평가...")
        hw_results = self.evaluate_folder(handwritten_dir, "handwritten")

        print("\n인쇄체 데이터 평가...")
        pr_results = self.evaluate_folder(printed_dir, "printed")

        # 전체 통계
        total = hw_results.get("total", 0) + pr_results.get("total", 0)
        correct = hw_results.get("correct", 0) + pr_results.get("correct", 0)
        overall_accuracy = correct / total * 100 if total > 0 else 0

        results = {
            "handwritten": hw_results,
            "printed": pr_results,
            "overall": {
                "total": total,
                "correct": correct,
                "accuracy": overall_accuracy
            }
        }

        # 결과 출력
        print(f"\n{'='*50}")
        print("평가 결과")
        print(f"{'='*50}")
        print(f"손글씨: {hw_results.get('accuracy', 0):.2f}% "
              f"({hw_results.get('correct', 0)}/{hw_results.get('total', 0)})")
        print(f"인쇄체: {pr_results.get('accuracy', 0):.2f}% "
              f"({pr_results.get('correct', 0)}/{pr_results.get('total', 0)})")
        print(f"{'='*50}")
        print(f"전체 정확도: {overall_accuracy:.2f}% ({correct}/{total})")
        print(f"{'='*50}")

        return results


def main():
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--model-path", type=str, required=True,
                        help="모델 파일 경로")
    parser.add_argument("--model-type", type=str, default="base",
                        choices=["base", "large", "resnet"])
    parser.add_argument("--image", type=str, default=None,
                        help="단일 이미지 예측")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="데이터셋 평가 (handwritten, printed 하위 폴더 필요)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--no-gpu", action="store_true")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        image_size=args.image_size,
        use_gpu=not args.no_gpu
    )

    if args.image:
        # 단일 이미지 예측
        label, confidence, probs = evaluator.predict(args.image)
        print(f"\n이미지: {args.image}")
        print(f"예측: {label}")
        print(f"신뢰도: {confidence*100:.2f}%")
        print(f"확률: printed={probs['printed']*100:.2f}%, "
              f"handwritten={probs['handwritten']*100:.2f}%")

    elif args.data_dir:
        # 데이터셋 평가
        data_dir = Path(args.data_dir)
        evaluator.evaluate_dataset(
            handwritten_dir=str(data_dir / "handwritten"),
            printed_dir=str(data_dir / "printed")
        )

    else:
        print("--image 또는 --data-dir 옵션을 지정하세요.")


if __name__ == "__main__":
    main()
