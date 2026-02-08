"""
ML 기반 분류 모델
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List
import pickle
import numpy as np

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    raise ImportError("scikit-learn이 필요합니다: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class BaseMLClassifier(ABC):
    """ML 분류기 기본 클래스"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    @abstractmethod
    def _create_model(self):
        """모델 생성"""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLClassifier":
        """학습"""
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 모델 생성 및 학습
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """정확도"""
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

    def save(self, path: Path) -> None:
        """모델 저장"""
        path = Path(path)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> "BaseMLClassifier":
        """모델 로드"""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]

        return self


class SVMClassifier(BaseMLClassifier):
    """
    SVM 분류기

    HOG 특징과 함께 사용하면 효과적
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale"
    ):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def _create_model(self):
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=42
        )

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> dict:
        """하이퍼파라미터 튜닝"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }

        X_scaled = self.scaler.fit_transform(X)

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_scaled, y)

        print(f"최적 파라미터: {grid_search.best_params_}")
        print(f"최적 점수: {grid_search.best_score_:.4f}")

        # 최적 파라미터로 업데이트
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        self.kernel = grid_search.best_params_['kernel']
        self.model = grid_search.best_estimator_
        self.is_fitted = True

        return grid_search.best_params_


class RandomForestMLClassifier(BaseMLClassifier):
    """
    Random Forest 분류기

    해석 가능하고 과적합에 강함
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
            n_jobs=-1
        )

    def get_feature_importance(self) -> np.ndarray:
        """특징 중요도 반환"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        return self.model.feature_importances_


class XGBoostClassifier(BaseMLClassifier):
    """
    XGBoost 분류기

    빠르고 정확함
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost가 필요합니다: pip install xgboost")

        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_model(self):
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )


class LightGBMClassifier(BaseMLClassifier):
    """
    LightGBM 분류기

    가장 빠름
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm이 필요합니다: pip install lightgbm")

        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_model(self):
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=-1
        )


def get_classifier(name: str, **kwargs) -> BaseMLClassifier:
    """이름으로 분류기 가져오기"""
    classifiers = {
        "svm": SVMClassifier,
        "random_forest": RandomForestMLClassifier,
        "rf": RandomForestMLClassifier,
        "xgboost": XGBoostClassifier,
        "xgb": XGBoostClassifier,
        "lightgbm": LightGBMClassifier,
        "lgbm": LightGBMClassifier,
    }

    if name not in classifiers:
        available = list(classifiers.keys())
        raise ValueError(f"알 수 없는 분류기: {name}. 가능: {available}")

    return classifiers[name](**kwargs)


def compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifiers: List[str] = None
) -> dict:
    """여러 분류기 비교"""
    if classifiers is None:
        classifiers = ["svm", "random_forest"]
        if HAS_XGBOOST:
            classifiers.append("xgboost")
        if HAS_LIGHTGBM:
            classifiers.append("lightgbm")

    results = {}

    for name in classifiers:
        print(f"\n{name} 학습 중...")
        try:
            clf = get_classifier(name)
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)

            results[name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "classifier": clf
            }

            print(f"  학습 정확도: {train_acc*100:.2f}%")
            print(f"  테스트 정확도: {test_acc*100:.2f}%")

        except Exception as e:
            print(f"  오류: {e}")
            results[name] = {"error": str(e)}

    return results
