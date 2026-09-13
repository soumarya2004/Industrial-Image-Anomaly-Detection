from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class ImageValidationError(ValueError):
    pass

def validate_image_file(path: str|Path)->None:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )
    img=cv2.imread(str(path))
    if img is None:
        raise ImageValidationError(
            f"File exists but could not be decoded as an image: {path}"
        )
    if img.ndim!=3 or img.shape[2]!=3:
        raise ImageValidationError(
            f"Expected a 3-channel color image, got shape {img.shape} for {path}"
        )


def load_and_resize_bgr(path: str|Path, size: int)->np.ndarray:
    img=cv2.imread(str(path))
    resized=cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def compute_reconstruction_heatmap(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
)->np.ndarray:
    diff=np.abs(
        original_rgb.astype(np.float32)
        - reconstructed_rgb.astype(np.float32)
    )
    diff_gray=diff.mean(axis=2)
    diff_gray=cv2.GaussianBlur(
        diff_gray,
        ksize=(5, 5),
        sigmaX=1.0,
    )
    diff_min, diff_max=diff_gray.min(), diff_gray.max()

    if diff_max-diff_min<1e-8:
        return np.zeros_like(diff_gray, dtype=np.uint8)
    normalized=(diff_gray-diff_min)/(diff_max-diff_min)
    return (normalized*255).astype(np.uint8)


def overlay_heatmap(
    base_bgr_uint8: np.ndarray,
    heatmap_uint8: np.ndarray,
    alpha: float=0.45,
)->np.ndarray:
    colored=cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    return cv2.addWeighted(
        colored,
        alpha,
        base_bgr_uint8,
        1 - alpha,
        0,
    )