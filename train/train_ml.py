"""
ML 모델 학습 스크립트

사용법:
    python -m train.train_ml --data-dir data
    python -m train.train_ml --data-dir data --classifier svm --features hog
    python -m train.train_ml --data-dir data --classifier rf --tune
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.features import get_extractor, BaseFeatureExtractor
from train.ml_model import (
    get_classifier,
    compare_classifiers,
    BaseMLClassifier
)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    raise ImportError("scikit-learn이 필요합니다: pip install scikit-learn")


def load_images_from_folder(
    folder: Path,
    label: int,
    max_samples: int = None
) -> Tuple[List[np.ndarray], List[int]]:
    """폴더에서 이미지 로드"""
    folder = Path(folder)
    if not folder.exists():
        print(f"경고: 폴더가 없습니다: {folder}")
        return [], []

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    images = []
    labels = []

    all_files = []
    for ext in extensions:
        all_files.extend(folder.glob(ext))
        all_files.extend(folder.glob(f"**/{ext}"))

    all_files = list(set(all_files))

    if max_samples:
        all_files = all_files[:max_samples]

    for img_path in all_files:
        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"이미지 로드 실패: {img_path}, {e}")

    return images, labels


def extract_features(
    images: List[np.ndarray],
    extractor: BaseFeatureExtractor,
    desc: str = "특징 추출"
) -> np.ndarray:
    """이미지에서 특징 추출"""
    features = []

    for img in tqdm(images, desc=desc):
        feat = extractor.extract(img)
        features.append(feat)

    return np.array(features)


def main():
    parser = argparse.ArgumentParser(description="ML 기반 손글씨/인쇄체 분류기 학습")

    # 데이터 설정
    parser.add_argument("--data-dir", type=str, default="data",
                        help="데이터 디렉토리")
    parser.add_argument("--handwritten-dir", type=str, default=None)
    parser.add_argument("--printed-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="클래스당 최대 샘플 수")

    # 특징 설정
    parser.add_argument("--features", type=str, default="hog",
                        choices=["hog", "lbp", "statistical", "combined"],
                        help="특징 추출 방법")
    parser.add_argument("--image-size", type=int, default=64,
                        help="특징 추출용 이미지 크기")

    # 분류기 설정
    parser.add_argument("--classifier", type=str, default="svm",
                        choices=["svm", "rf", "random_forest", "xgboost", "xgb", "lightgbm", "lgbm"],
                        help="분류기 종류")
    parser.add_argument("--compare", action="store_true",
                        help="여러 분류기 비교")
    parser.add_argument("--tune", action="store_true",
                        help="하이퍼파라미터 튜닝 (SVM)")

    # 출력 설정
    parser.add_argument("--output-dir", type=str, default="models",
                        help="모델 저장 경로")
    parser.add_argument("--model-name", type=str, default="handwriting_ml",
                        help="모델 파일명")

    # 기타
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="테스트 데이터 비율")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir)
    handwritten_dir = Path(args.handwritten_dir) if args.handwritten_dir else data_dir / "handwritten"
    printed_dir = Path(args.printed_dir) if args.printed_dir else data_dir / "printed"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ML 기반 손글씨/인쇄체 분류기 학습")
    print(f"{'='*60}")
    print(f"손글씨 데이터: {handwritten_dir}")
    print(f"인쇄체 데이터: {printed_dir}")
    print(f"특징 추출: {args.features}")
    print(f"분류기: {args.classifier}")
    print(f"{'='*60}\n")

    # 데이터 로드
    print("데이터 로딩 중...")
    hw_images, hw_labels = load_images_from_folder(
        handwritten_dir, label=1, max_samples=args.max_samples
    )
    pr_images, pr_labels = load_images_from_folder(
        printed_dir, label=0, max_samples=args.max_samples
    )

    if not hw_images or not pr_images:
        print("오류: 데이터가 부족합니다.")
        sys.exit(1)

    all_images = hw_images + pr_images
    all_labels = np.array(hw_labels + pr_labels)

    print(f"\n데이터 로드 완료:")
    print(f"  - 손글씨: {len(hw_images)}개")
    print(f"  - 인쇄체: {len(pr_images)}개")
    print(f"  - 총: {len(all_images)}개")

    # 특징 추출
    print(f"\n{args.features.upper()} 특징 추출 중...")
    extractor = get_extractor(args.features, image_size=(args.image_size, args.image_size))
    X = extract_features(all_images, extractor)
    y = all_labels

    print(f"특징 벡터 크기: {X.shape[1]}")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=y
    )

    print(f"\n데이터 분할:")
    print(f"  - 학습: {len(X_train)}개")
    print(f"  - 테스트: {len(X_test)}개")

    # 분류기 학습
    start_time = time.time()

    if args.compare:
        # 여러 분류기 비교
        print("\n여러 분류기 비교...")
        results = compare_classifiers(X_train, y_train, X_test, y_test)

        # 최고 성능 분류기 선택
        best_name = max(results.keys(), key=lambda k: results[k].get("test_acc", 0))
        best_clf = results[best_name]["classifier"]
        best_acc = results[best_name]["test_acc"]

        print(f"\n최고 성능: {best_name} (정확도: {best_acc*100:.2f}%)")

    else:
        # 단일 분류기 학습
        print(f"\n{args.classifier} 학습 중...")
        clf = get_classifier(args.classifier)

        if args.tune and args.classifier == "svm":
            print("하이퍼파라미터 튜닝 중...")
            clf.tune_hyperparameters(X_train, y_train)
        else:
            clf.fit(X_train, y_train)

        best_clf = clf
        best_name = args.classifier
        best_acc = clf.score(X_test, y_test)

    elapsed_time = time.time() - start_time

    # 평가
    print(f"\n{'='*60}")
    print("평가 결과")
    print(f"{'='*60}")

    y_pred = best_clf.predict(X_test)

    print(f"\n학습 시간: {elapsed_time:.2f}초")
    print(f"테스트 정확도: {best_acc*100:.2f}%")

    print("\n분류 리포트:")
    print(classification_report(
        y_test, y_pred,
        target_names=["printed", "handwritten"]
    ))

    print("혼동 행렬:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Printed  Handwritten")
    print(f"Actual Printed    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"   Handwritten    {cm[1,0]:5d}  {cm[1,1]:5d}")

    # 모델 저장
    model_path = output_dir / f"{args.model_name}_{best_name}.pkl"
    best_clf.save(model_path)

    # 특징 추출기 정보 저장
    import json
    config_path = output_dir / f"{args.model_name}_{best_name}_config.json"
    config = {
        "feature_extractor": args.features,
        "image_size": args.image_size,
        "classifier": best_name,
        "accuracy": best_acc,
        "feature_size": X.shape[1]
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("학습 완료!")
    print(f"{'='*60}")
    print(f"모델 저장됨: {model_path}")
    print(f"설정 저장됨: {config_path}")

    print(f"\n사용 방법:")
    print(f"  from train.ml_classifier import MLHandwritingClassifier")
    print(f"  classifier = MLHandwritingClassifier(")
    print(f"      model_path='{model_path}',")
    print(f"      feature_extractor='{args.features}'")
    print(f"  )")
    print(f"  result = classifier.classify(image)")


if __name__ == "__main__":
    main()
