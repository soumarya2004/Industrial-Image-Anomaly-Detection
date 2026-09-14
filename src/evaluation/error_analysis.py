from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from src.data.dataset import Sample, build_transforms
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier
from src.visualization.anomaly_map import generate_anomaly_visualization
from src.visualization.gradcam import generate_gradcam_overlay

LABEL_NAMES = {0: "normal", 1: "defective"}


@dataclass
class ErrorRecord:
    image_path: str
    defect_type: str
    true_label: int
    predicted_label: int
    score: float               # P(defective) for classifier, reconstruction MSE for autoencoder
    error_type: str | None     # "false_positive" | "false_negative" | "low_margin_correct" | None
    mask_coverage: float | None
    saved_visualization_path: str | None


def _mask_coverage_fraction(mask_path: Path|None)->float|None:
    if mask_path is None:
        return None
    mask=np.array(Image.open(mask_path).convert("L"))
    return float((mask>127).mean())


def _compose_side_by_side(
    panels_rgb: list[np.ndarray], captions: list[str], banner_text: str
)->np.ndarray:
    h, w = panels_rgb[0].shape[:2]
    caption_h, banner_h = 24, 28
    n=len(panels_rgb)
    canvas=np.full((banner_h+h+caption_h, w*n, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas, banner_text, (6, banner_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    for i, (panel, caption) in enumerate(zip(panels_rgb, captions)):
        canvas[banner_h:banner_h+h, i*w:(i+1)*w]=panel
        cv2.putText(
            canvas, caption, (i*w+6, banner_h+h+caption_h-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return canvas


def analyze_classifier_errors(
    model: DefectClassifier,
    cfg: dict,
    device: torch.device,
    samples: list[Sample],
    output_dir: str | Path,
    low_confidence_margin: float = 0.15,
)->dict:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=True)
    raw_transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
    counts={"false_positive": 0, "false_negative": 0, "low_margin_correct": 0}
    records: list[ErrorRecord] = []

    for sample in samples:
        image=Image.open(sample.image_path).convert("RGB")
        image_tensor=transform(image).unsqueeze(0)

        with torch.no_grad():
            probs=torch.softmax(model(image_tensor.to(device)), dim=1)[0]
        defect_prob=float(probs[1].item())
        predicted_label=int(defect_prob>=0.5)

        error_type=None
        if predicted_label!=sample.label:
            error_type="false_positive" if predicted_label==1 else "false_negative"
        elif abs(defect_prob-0.5)<=low_confidence_margin:
            error_type="low_margin_correct"

        saved_path=None
        if error_type is not None:
            raw_image=raw_transform(image).permute(1, 2, 0).numpy()
            overlay=generate_gradcam_overlay(
                model, image_tensor, raw_image, target_class=predicted_label, device=device
            )
            original_uint8=(raw_image*255).astype(np.uint8)
            banner=(
                f"{error_type} | true={LABEL_NAMES[sample.label]} "
                f"pred={LABEL_NAMES[predicted_label]} P(defective)={defect_prob:.3f}"
            )
            composite=_compose_side_by_side(
                [original_uint8, overlay], ["original", "grad-cam"], banner
            )
            saved_path=str(output_dir/f"classifier_{error_type}_{sample.image_path.stem}.png")
            cv2.imwrite(saved_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            counts[error_type]+=1

        records.append(
            ErrorRecord(
                image_path=str(sample.image_path),
                defect_type=sample.defect_type,
                true_label=sample.label,
                predicted_label=predicted_label,
                score=defect_prob,
                error_type=error_type,
                mask_coverage=_mask_coverage_fraction(sample.mask_path),
                saved_visualization_path=saved_path,
            )
        )

    return {"counts": counts, "records": [asdict(r) for r in records]}


def analyze_autoencoder_errors(
    model: ConvAutoencoder,
    cfg: dict,
    device: torch.device,
    samples: list[Sample],
    threshold: float,
    output_dir: str | Path,
    low_margin_relative: float = 0.2,
)->dict:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
    margin_absolute=threshold*low_margin_relative

    counts={"false_positive": 0, "false_negative": 0, "low_margin_correct": 0}
    records: list[ErrorRecord]=[]

    for sample in samples:
        image=Image.open(sample.image_path).convert("RGB")
        image_tensor=transform(image).unsqueeze(0)

        viz=generate_anomaly_visualization(model, image_tensor, device)
        score=viz["anomaly_score"]
        predicted_label=int(score>=threshold)

        error_type=None
        if predicted_label!=sample.label:
            error_type="false_positive" if predicted_label==1 else "false_negative"
        elif abs(score-threshold)<=margin_absolute:
            error_type="low_margin_correct"

        saved_path=None
        if error_type is not None:
            original_uint8=(viz["original_rgb"]*255).astype(np.uint8)
            reconstructed_uint8=(viz["reconstructed_rgb"]*255).astype(np.uint8)
            overlay_rgb=viz["overlay_bgr"][:, :, ::-1]
            banner=(
                f"{error_type} | true={LABEL_NAMES[sample.label]} "
                f"pred={LABEL_NAMES[predicted_label]} score={score:.5f} threshold={threshold:.5f}"
            )
            composite=_compose_side_by_side(
                [original_uint8, reconstructed_uint8, overlay_rgb],
                ["original", "reconstruction", "anomaly heatmap"],
                banner,
            )
            saved_path=str(output_dir/f"autoencoder_{error_type}_{sample.image_path.stem}.png")
            cv2.imwrite(saved_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            counts[error_type]+=1

        records.append(
            ErrorRecord(
                image_path=str(sample.image_path),
                defect_type=sample.defect_type,
                true_label=sample.label,
                predicted_label=predicted_label,
                score=score,
                error_type=error_type,
                mask_coverage=_mask_coverage_fraction(sample.mask_path),
                saved_visualization_path=saved_path,
            )
        )

    return {"counts": counts, "records": [asdict(r) for r in records]}


def summarize_defect_area_vs_failure(records: list[dict])->dict:
    fn_coverages=[
        r["mask_coverage"] for r in records
        if r["error_type"]=="false_negative" and r["mask_coverage"] is not None
    ]
    tp_coverages=[
        r["mask_coverage"] for r in records
        if r["true_label"]==1 and r["error_type"]!="false_negative" and r["mask_coverage"] is not None
    ]

    mean_fn=float(np.mean(fn_coverages)) if fn_coverages else float("nan")
    mean_tp=float(np.mean(tp_coverages)) if tp_coverages else float("nan")

    return {
        "n_false_negatives_with_mask": len(fn_coverages),
        "n_true_positives_with_mask": len(tp_coverages),
        "mean_defect_area_fraction_false_negatives": mean_fn,
        "mean_defect_area_fraction_true_positives": mean_tp,
        "difference": mean_tp - mean_fn if fn_coverages and tp_coverages else float("nan"),
    }


def save_error_analysis_summary(summary: dict, output_path: str | Path)->None:
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
