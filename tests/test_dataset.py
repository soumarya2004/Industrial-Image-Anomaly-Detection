from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.data.dataset import MVTecDataset, build_samples, build_transforms
from src.data.preprocessing import ImageValidationError, validate_image_file


def _make_fake_mvtec(root: Path, category: str="bottle")->None:
    cat_dir=root/category
    (cat_dir / "train" / "good").mkdir(parents=True)
    (cat_dir / "test" / "good").mkdir(parents=True)
    (cat_dir / "test" / "broken_large").mkdir(parents=True)
    (cat_dir / "ground_truth" / "broken_large").mkdir(parents=True)

    def save_random_image(path: Path, size=(32, 32)):
        arr=(np.random.rand(*size, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

    for i in range(20):
        save_random_image(cat_dir / "train" / "good" / f"{i:03d}.png")
    for i in range(5):
        save_random_image(cat_dir / "test" / "good" / f"{i:03d}.png")
    for i in range(5):
        save_random_image(cat_dir / "test" / "broken_large" / f"{i:03d}.png")
        mask=(np.random.rand(32, 32) > 0.8).astype(np.uint8) * 255
        Image.fromarray(mask).save(cat_dir / "ground_truth" / "broken_large" / f"{i:03d}_mask.png")


@pytest.fixture()
def fake_dataset_root():
    tmpdir=tempfile.mkdtemp()
    root=Path(tmpdir)
    _make_fake_mvtec(root)
    yield root
    shutil.rmtree(tmpdir)


class TestBuildSamples:
    def test_ae_train_contains_only_normal_images(self, fake_dataset_root):
        samples=build_samples(fake_dataset_root, "bottle", "ae_train")
        assert len(samples)>0
        assert all(s.label==0 for s in samples)
        assert all(s.defect_type=="good" for s in samples)

    def test_ae_train_and_ae_val_are_disjoint(self, fake_dataset_root):
        train_samples=build_samples(fake_dataset_root, "bottle", "ae_train")
        val_samples=build_samples(fake_dataset_root, "bottle", "ae_val")
        train_paths={s.image_path for s in train_samples}
        val_paths={s.image_path for s in val_samples}
        assert train_paths.isdisjoint(val_paths)
        # every train/good image should appear in exactly one of the two splits
        assert len(train_paths)+len(val_paths)==20

    def test_labeled_val_and_labeled_test_are_disjoint(self, fake_dataset_root):
        val_samples=build_samples(fake_dataset_root, "bottle", "labeled_val", val_fraction=0.3)
        test_samples=build_samples(fake_dataset_root, "bottle", "labeled_test", val_fraction=0.3)
        val_paths={s.image_path for s in val_samples}
        test_paths={s.image_path for s in test_samples}
        assert val_paths.isdisjoint(test_paths)
        assert len(val_paths)+len(test_paths)==10  # 5 good + 5 broken_large

    def test_labeled_split_is_deterministic_given_seed(self, fake_dataset_root):
        val_a=build_samples(fake_dataset_root, "bottle", "labeled_val", seed=42)
        val_b=build_samples(fake_dataset_root, "bottle", "labeled_val", seed=42)
        assert {s.image_path for s in val_a} == {s.image_path for s in val_b}

    def test_different_seeds_produce_different_splits(self, fake_dataset_root):
        val_seed1={s.image_path for s in build_samples(fake_dataset_root, "bottle", "labeled_val", seed=1)}
        val_seed2={s.image_path for s in build_samples(fake_dataset_root, "bottle", "labeled_val", seed=999)}
        assert val_seed1 != val_seed2

    def test_defective_samples_have_mask_path(self, fake_dataset_root):
        test_samples=build_samples(fake_dataset_root, "bottle", "labeled_test", val_fraction=0.0)
        defective=[s for s in test_samples if s.label==1]
        assert len(defective)>0
        assert all(s.mask_path is not None and s.mask_path.exists() for s in defective)

    def test_normal_samples_have_no_mask_path(self, fake_dataset_root):
        test_samples=build_samples(fake_dataset_root, "bottle", "labeled_test", val_fraction=0.0)
        normal=[s for s in test_samples if s.label==0]
        assert all(s.mask_path is None for s in normal)

    def test_missing_category_raises_file_not_found(self, fake_dataset_root):
        with pytest.raises(FileNotFoundError):
            build_samples(fake_dataset_root, "nonexistent_category", "ae_train")

    def test_invalid_split_name_raises(self, fake_dataset_root):
        with pytest.raises(ValueError):
            build_samples(fake_dataset_root, "bottle", "not_a_real_split")


class TestMVTecDataset:
    def test_dataset_length_matches_samples(self, fake_dataset_root):
        ds=MVTecDataset(fake_dataset_root, "bottle", split="ae_train", image_size=32)
        assert len(ds)==len(build_samples(fake_dataset_root, "bottle", "ae_train"))

    def test_getitem_returns_correct_shapes(self, fake_dataset_root):
        ds=MVTecDataset(fake_dataset_root, "bottle", split="ae_train", image_size=64)
        image, label, mask=ds[0]
        assert image.shape==(3, 64, 64)
        assert isinstance(label, int)
        assert mask.shape==(1, 64, 64)

    def test_normalize_false_keeps_values_in_unit_range(self, fake_dataset_root):
        ds=MVTecDataset(fake_dataset_root, "bottle", split="ae_train", image_size=32, normalize=False)
        image, _, _ = ds[0]
        assert image.min()>=0.0 and image.max()<=1.0

    def test_normalize_true_shifts_values_outside_unit_range(self, fake_dataset_root):
        ds = MVTecDataset(fake_dataset_root, "bottle", split="ae_train", image_size=32, normalize=True)
        image, _, _ = ds[0]
        # ImageNet-normalized images are centered near 0 and can go negative/above 1
        assert image.min()<0.0

    def test_empty_split_raises_value_error(self, fake_dataset_root):
        # val_fraction=0.0 sends every labeled image to "labeled_test", so
        # the "labeled_val" split is empty and construction should fail loudly
        # rather than silently yielding a zero-length dataset.
        with pytest.raises(ValueError):
            MVTecDataset(fake_dataset_root, "bottle", split="labeled_val", val_fraction=0.0, image_size=32)


class TestBuildTransforms:
    def test_train_transform_without_augmentation_matches_eval_ops_count(self):
        eval_t=build_transforms(64, train=False)
        train_t=build_transforms(64, train=True, augmentation_cfg={"enabled": False})
        assert len(eval_t.transforms)==len(train_t.transforms)

    def test_augmentation_only_applied_when_train_true(self):
        aug_cfg={"enabled": True, "horizontal_flip": True, "rotation_degrees": 10}
        eval_t=build_transforms(64, train=False, augmentation_cfg=aug_cfg)
        train_t=build_transforms(64, train=True, augmentation_cfg=aug_cfg)
        assert len(train_t.transforms) > len(eval_t.transforms)


class TestImageValidation:
    def test_valid_image_passes(self, fake_dataset_root):
        sample_path=next((fake_dataset_root / "bottle" / "train" / "good").glob("*.png"))
        validate_image_file(sample_path)  # should not raise

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            validate_image_file("/tmp/does_not_exist_12345.png")

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file=tmp_path / "not_an_image.txt"
        bad_file.write_text("hello")
        with pytest.raises(ImageValidationError):
            validate_image_file(bad_file)

    def test_corrupt_image_raises(self, tmp_path):
        corrupt_file=tmp_path / "corrupt.png"
        corrupt_file.write_bytes(b"not a real png")
        with pytest.raises(ImageValidationError):
            validate_image_file(corrupt_file)
