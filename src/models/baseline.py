from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MeanImageBaseline:
    def __init__(self, image_size: int = 128):
        self.image_size=image_size
        self.mean_image: np.ndarray | None = None  # (H, W, 3) float32 in [0, 1]

    def _load_rgb01(self, path: str|Path)->np.ndarray:
        img=cv2.imread(str(path))
        img=cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)/255.0

    def fit(self, normal_image_paths: list[str | Path])->"MeanImageBaseline":
        if len(normal_image_paths)==0:
            raise ValueError("MeanImageBaseline.fit requires at least one image path.")
        accumulator=np.zeros((self.image_size, self.image_size, 3), dtype=np.float64)
        for path in normal_image_paths:
            accumulator+=self._load_rgb01(path)
        self.mean_image=(accumulator/len(normal_image_paths)).astype(np.float32)
        return self

    def score(self, image_path: str|Path)->float:
        if self.mean_image is None:
            raise RuntimeError("Call fit() before score().")
        image=self._load_rgb01(image_path)
        return float(np.mean((image-self.mean_image)**2))

    def score_batch(self, image_paths: list[str|Path])->np.ndarray:
        return np.array([self.score(p) for p in image_paths])


def _extract_classical_features(image_path: str | Path, image_size: int=128)->np.ndarray:
    img_bgr=cv2.imread(str(image_path))
    img_bgr=cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hog_features=hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    color_hist=[]
    for channel in range(3):
        hist=cv2.calcHist([img_bgr], [channel], None, [32], [0, 256])
        color_hist.append(hist.flatten())
    color_hist=np.concatenate(color_hist)
    color_hist=color_hist/(color_hist.sum()+1e-8)  # normalize to a distribution
    return np.concatenate([hog_features, color_hist]).astype(np.float32)


class ClassicalFeatureClassifier:
    def __init__(self, image_size: int=128, class_weight: str|dict|None="balanced"):
        self.image_size=image_size
        self.scaler=StandardScaler()
        self.clf=LogisticRegression(max_iter=2000, class_weight=class_weight)
        self._fitted=False

    def _featurize_batch(self, image_paths: list[str|Path])->np.ndarray:
        return np.stack([_extract_classical_features(p, self.image_size) for p in image_paths])

    def fit(self, image_paths: list[str|Path], labels: list[int])->"ClassicalFeatureClassifier":
        features=self._featurize_batch(image_paths)
        features=self.scaler.fit_transform(features)
        self.clf.fit(features, labels)
        self._fitted=True
        return self

    def predict_proba(self, image_paths: list[str|Path])->np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        features=self._featurize_batch(image_paths)
        features=self.scaler.transform(features)
        return self.clf.predict_proba(features)[:, 1]
