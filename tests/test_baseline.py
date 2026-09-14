from __future__ import annotations
import numpy as np
import pytest
from PIL import Image
from src.models.baseline import ClassicalFeatureClassifier, MeanImageBaseline, _extract_classical_features


def _save_random_image(path, size=(64, 64), seed=None):
    rng=np.random.default_rng(seed)
    arr=(rng.random((*size, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestMeanImageBaseline:
    def test_fit_requires_at_least_one_image(self):
        model=MeanImageBaseline(image_size=32)
        with pytest.raises(ValueError):
            model.fit([])

    def test_score_before_fit_raises(self, tmp_path):
        model=MeanImageBaseline(image_size=32)
        img_path=tmp_path / "a.png"
        _save_random_image(img_path, size=(32, 32))
        with pytest.raises(RuntimeError):
            model.score(img_path)

    def test_mean_image_has_correct_shape(self, tmp_path):
        paths=[]
        for i in range(5):
            p=tmp_path/f"{i}.png"
            _save_random_image(p, size=(32, 32), seed=i)
            paths.append(p)
        model=MeanImageBaseline(image_size=32)
        model.fit(paths)
        assert model.mean_image.shape==(32, 32, 3)
        assert model.mean_image.min()>=0.0 and model.mean_image.max() <= 1.0

    def test_score_of_exact_mean_image_is_near_zero(self, tmp_path):
        #a solid-color image repeated identically: the "mean" is that same
        #image, so its own reconstruction error should be exactly zero.
        p= tmp_path / "solid.png"
        arr=np.full((32, 32, 3), 128, dtype=np.uint8)
        Image.fromarray(arr).save(p)
        model=MeanImageBaseline(image_size=32)
        model.fit([p, p, p])
        score=model.score(p)
        assert score<1e-6

    def test_score_batch_returns_array_of_correct_length(self, tmp_path):
        train_paths=[]
        for i in range(3):
            p= tmp_path / f"train_{i}.png"
            _save_random_image(p, size=(32, 32), seed=i)
            train_paths.append(p)
        model=MeanImageBaseline(image_size=32).fit(train_paths)

        test_paths=[]
        for i in range(4):
            p= tmp_path / f"test_{i}.png"
            _save_random_image(p, size=(32, 32), seed=100 + i)
            test_paths.append(p)
        scores=model.score_batch(test_paths)
        assert scores.shape==(4,)
        assert np.all(scores>=0)


class TestClassicalFeatureExtraction:
    def test_feature_vector_is_1d_and_finite(self, tmp_path):
        p= tmp_path / "a.png"
        _save_random_image(p, size=(64, 64), seed=1)
        features=_extract_classical_features(p, image_size=64)
        assert features.ndim==1
        assert np.all(np.isfinite(features))

    def test_feature_vector_length_is_consistent_across_images(self, tmp_path):
        p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
        _save_random_image(p1, size=(64, 64), seed=1)
        _save_random_image(p2, size=(64, 64), seed=2)
        f1=_extract_classical_features(p1, image_size=64)
        f2=_extract_classical_features(p2, image_size=64)
        assert f1.shape==f2.shape


class TestClassicalFeatureClassifier:
    def test_predict_before_fit_raises(self, tmp_path):
        clf=ClassicalFeatureClassifier(image_size=32)
        p= tmp_path / "a.png"
        _save_random_image(p, size=(32, 32))
        with pytest.raises(RuntimeError):
            clf.predict_proba([p])

    def test_fit_and_predict_proba_shape_and_range(self, tmp_path):
        paths, labels = [], []
        for i in range(10):
            p = tmp_path / f"{i}.png"
            _save_random_image(p, size=(48, 48), seed=i)
            paths.append(p)
            labels.append(i % 2)

        clf=ClassicalFeatureClassifier(image_size=48)
        clf.fit(paths, labels)

        test_paths=paths[:3]
        probs=clf.predict_proba(test_paths)
        assert probs.shape==(3,)
        assert np.all((probs>=0) & (probs<=1))
