from __future__ import annotations
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXPECTED_BOTTLE_COUNTS = {
    ("train", "good"): 209,
    ("test", "good"): 20,
    ("test", "broken_large"): 20,
    ("test", "broken_small"): 22,
    ("test", "contamination"): 21,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class Sample:
    image_path: Path
    label: int              # 0 = normal, 1 = defective
    defect_type: str        # "good" or the specific defect subfolder name
    mask_path: Path | None  # pixel-level ground truth, only for defective test images

def _stable_split_bucket(filename: str, seed: int, val_fraction: float) -> str:
    digest=hashlib.sha256(f"{filename}_{seed}".encode()).hexdigest()
    # Map the first 8 hex chars to a float in [0, 1)
    fraction=int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if fraction < val_fraction else "test"

def _list_defect_subdirs(test_dir: Path) -> list[str]:
    return sorted(p.name for p in test_dir.iterdir() if p.is_dir())

def build_samples(
    root: str | Path,
    category: str,
    split: str,
    seed: int=42,
    val_fraction: float=0.3,
)->list[Sample]:
    category_dir=Path(root)/category
    if not category_dir.exists():
        raise FileNotFoundError(
            f"MVTec AD category directory not found: {category_dir}\n"
            f"See data/README.md for download instructions."
        )
    if split in ("ae_train", "ae_val"):
        good_dir=category_dir/"train"/"good"
        files=sorted(good_dir.glob("*.png"))
        samples=[]
        for f in files:
            bucket=_stable_split_bucket(f.name, seed, val_fraction=0.15)
            target_split="ae_val" if bucket == "val" else "ae_train"
            if target_split==split:
                samples.append(Sample(image_path=f, label=0, defect_type="good", mask_path=None))
        return samples

    if split in ("labeled_val", "labeled_test"):
        test_dir=category_dir/"test"
        gt_dir=category_dir/"ground_truth"
        samples=[]
        for defect_type in _list_defect_subdirs(test_dir):
            label=0 if defect_type=="good" else 1
            for f in sorted((test_dir/defect_type).glob("*.png")):
                bucket=_stable_split_bucket(f.name, seed, val_fraction)
                target_split="labeled_val" if bucket=="val" else "labeled_test"
                if target_split!=split:
                    continue
                mask_path=None
                if label==1:
                    candidate=gt_dir/defect_type/f"{f.stem}_mask.png"
                    mask_path=candidate if candidate.exists() else None
                samples.append(
                    Sample(image_path=f, label=label, defect_type=defect_type, mask_path=mask_path)
                )
        return samples
    raise ValueError(
        f"Unknown split '{split}'. Expected one of: "
        "ae_train, ae_val, labeled_val, labeled_test."
    )

def build_transforms(
    image_size: int,
    train: bool,
    augmentation_cfg: dict | None = None,
    normalize: bool=True,
):
    augmentation_cfg=augmentation_cfg or {}
    ops = [transforms.Resize((image_size, image_size))]

    if train and augmentation_cfg.get("enabled", False):
        if augmentation_cfg.get("horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if augmentation_cfg.get("vertical_flip", False):
            ops.append(transforms.RandomVerticalFlip(p=0.5))
        rotation=augmentation_cfg.get("rotation_degrees", 0)
        if rotation:
            ops.append(transforms.RandomRotation(degrees=rotation))
        brightness=augmentation_cfg.get("brightness_jitter", 0.0)
        contrast=augmentation_cfg.get("contrast_jitter", 0.0)
        if brightness or contrast:
            ops.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
        crop_scale=augmentation_cfg.get("random_crop_scale")
        if crop_scale:
            ops.append(
                transforms.RandomResizedCrop(
                    image_size, scale=tuple(crop_scale), ratio=(0.95, 1.05)
                )
            )
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)

class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_size: int=224,
        seed: int=42,
        val_fraction: float=0.3,
        augmentation_cfg: dict|None=None,
        normalize: bool=True,
    ):
        is_train_split=split in ("ae_train",)
        self.samples=build_samples(root, category, split, seed=seed, val_fraction=val_fraction)
        if len(self.samples)==0:
            raise ValueError(
                f"No samples found for split='{split}', category='{category}', root='{root}'. "
                f"Check data/README.md — has the dataset been downloaded and extracted?"
            )
        self.transform=build_transforms(
            image_size, train=is_train_split, augmentation_cfg=augmentation_cfg, normalize=normalize
        )
        self.mask_transform=transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )
        self.image_size=image_size
    def __len__(self)->int:
        return len(self.samples)
    def __getitem__(self, idx: int)->tuple[torch.Tensor, int, torch.Tensor]:
        sample=self.samples[idx]
        image=Image.open(sample.image_path).convert("RGB")
        image_tensor=self.transform(image)
        if sample.mask_path is not None:
            mask=Image.open(sample.mask_path).convert("L")
            mask_tensor=self.mask_transform(mask)
            mask_tensor=(mask_tensor>0.5).float()
        else:
            mask_tensor=torch.zeros((1, self.image_size, self.image_size))
        return image_tensor, sample.label, mask_tensor

def verify_download(root:str|Path, category:str)->None:
    category_dir=Path(root)/category
    print(f"Verifying: {category_dir}")
    if not category_dir.exists():
        print(f"[FAIL] Category directory does not exist: {category_dir}")
        raise SystemExit(1)
    ok=True
    for (split, subclass), expected_count in EXPECTED_BOTTLE_COUNTS.items():
        d=category_dir/split/subclass
        if not d.exists():
            print(f"[FAIL] Missing directory: {d}")
            ok=False
            continue
        actual_count=len(list(d.glob("*.png")))
        status="OK" if actual_count==expected_count else "MISMATCH"
        if status=="MISMATCH":
            ok=False
        print(f"[{status}] {split}/{subclass}: found {actual_count}, expected {expected_count}")
    gt_dir=category_dir/"ground_truth"
    if gt_dir.exists():
        print(f"[OK] ground_truth directory present: {gt_dir}")
    else:
        print(f"[WARN] ground_truth directory missing — pixel-level localization metrics unavailable.")
    if ok:
        print("\nAll checks passed. Dataset is ready.")
    else:
        print("\nSome checks failed — see data/README.md for the expected layout.")
        raise SystemExit(1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="MVTec AD dataset utilities")
    parser.add_argument("--verify", action="store_true", help="Verify a downloaded category")
    parser.add_argument("--root", type=str, default="data/mvtec_ad")
    parser.add_argument("--category", type=str, default="bottle")
    args=parser.parse_args()
    if args.verify:
        verify_download(args.root, args.category)
    else:
        parser.print_help()
