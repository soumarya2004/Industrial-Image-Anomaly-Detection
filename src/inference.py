from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from src.data.dataset import build_transforms
from src.data.preprocessing import validate_image_file
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier
from src.utils import get_device
from src.visualization.anomaly_map import generate_anomaly_visualization
from src.visualization.gradcam import generate_gradcam_overlay


def load_classifier(checkpoint_path: str, device: torch.device)->tuple[DefectClassifier, dict]:
    checkpoint=torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg=checkpoint["config"]
    model=DefectClassifier(
        architecture=cfg["model"]["architecture"],
        pretrained=False,  # weights come from the checkpoint, not ImageNet, at load time
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        finetune_mode=cfg["model"]["finetune_mode"],
        unfrozen_layers=cfg["model"].get("unfrozen_layers", []),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def load_autoencoder(checkpoint_path: str, device: torch.device)->tuple[ConvAutoencoder, dict]:
    checkpoint=torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg=checkpoint["config"]
    model=ConvAutoencoder(
        image_size=cfg["data"]["image_size"],
        latent_dim=cfg["model"]["latent_dim"],
        base_channels=cfg["model"]["base_channels"],
        num_downsample_blocks=cfg["model"]["num_downsample_blocks"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def run_classifier_inference(
    model: DefectClassifier, cfg: dict, image_path: str, device: torch.device, save_gradcam: bool = True
)->dict:
    validate_image_file(image_path)
    image=Image.open(image_path).convert("RGB")
    transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=True)
    image_tensor=transform(image).unsqueeze(0)

    start=time.perf_counter()
    with torch.no_grad():
        logits=model(image_tensor.to(device))
        probs=torch.softmax(logits, dim=1)[0]
    latency_ms=(time.perf_counter()-start)*1000

    predicted_class=int(torch.argmax(probs).item())
    confidence=float(probs[predicted_class].item())
    label="DEFECTIVE" if predicted_class == 1 else "NORMAL"

    result={
        "prediction": label,
        "confidence": confidence,
        "anomaly_score": float(probs[1].item()),
        "latency_ms": latency_ms,
        "gradcam_path": None,
    }

    if save_gradcam:
        raw_transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
        image_rgb01=raw_transform(image).permute(1, 2, 0).numpy()
        overlay=generate_gradcam_overlay(
            model, image_tensor, image_rgb01, target_class=predicted_class, device=device
        )
        out_dir=Path("results/heatmaps")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path=out_dir/f"{Path(image_path).stem}_gradcam.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        result["gradcam_path"]=str(out_path)

    return result


def run_autoencoder_inference(
    model: ConvAutoencoder, cfg: dict, image_path: str, device: torch.device, threshold: float
)->dict:
    validate_image_file(image_path)
    image=Image.open(image_path).convert("RGB")
    transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
    image_tensor=transform(image).unsqueeze(0)

    start=time.perf_counter()
    viz=generate_anomaly_visualization(model, image_tensor, device)
    latency_ms=(time.perf_counter()-start)*1000

    is_anomalous=viz["anomaly_score"]>=threshold
    label="DEFECTIVE" if is_anomalous else "NORMAL"

    out_dir=Path("results/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path=out_dir/f"{Path(image_path).stem}_anomaly_map.png"
    cv2.imwrite(str(out_path), viz["overlay_bgr"])

    return{
        "prediction": label,
        "anomaly_score": viz["anomaly_score"],
        "threshold": threshold,
        "latency_ms": latency_ms,
        "heatmap_path": str(out_path),
    }


def print_report(image_path: str, result: dict)->None:
    print("=" * 40)
    print("Industrial Inspection Result")
    print("=" * 40)
    print(f"Image: {image_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence'] * 100:.1f}%")
    if "threshold" in result:
        print(f"Threshold: {result['threshold']:.4f}")
    print(f"Latency: {result['latency_ms']:.1f} ms")
    heatmap_path = result.get("gradcam_path") or result.get("heatmap_path")
    if heatmap_path:
        print(f"Detected Region: [heatmap saved to {heatmap_path}]")
    print("=" * 40)


def main()->None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--image", type=str, help="Path to a single image.")
    parser.add_argument("--image_dir", type=str, help="Path to a directory of images (batch mode).")
    parser.add_argument("--model", type=str, choices=["classifier", "autoencoder"], required=True)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override default checkpoint path (models/<model>/best_model.pt).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Anomaly score threshold for autoencoder mode (defaults to the value logged during training).",
    )
    args=parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image_dir")

    device=get_device(prefer_cuda=True)
    checkpoint_path=args.checkpoint or f"models/{args.model}/best_model.pt"

    if args.model=="classifier":
        model, cfg=load_classifier(checkpoint_path, device)
    else:
        model, cfg=load_autoencoder(checkpoint_path, device)
        threshold=args.threshold
        if threshold is None:
            raise ValueError(
                "--threshold is required for autoencoder inference. Use the value "
                "printed/logged at the end of training (results/experiments/*.json)."
            )

    image_paths=[args.image] if args.image else sorted(
        str(p) for p in Path(args.image_dir).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    latencies=[]
    for image_path in image_paths:
        if args.model=="classifier":
            result=run_classifier_inference(model, cfg, image_path, device)
        else:
            result=run_autoencoder_inference(model, cfg, image_path, device, threshold)
        latencies.append(result["latency_ms"])
        print_report(image_path, result)

    if len(image_paths)>1:
        total_time_sec=sum(latencies)/1000
        print(
            f"\nBatch summary: {len(image_paths)} images | "
            f"mean latency {np.mean(latencies):.1f} ms | "
            f"throughput {len(image_paths) / total_time_sec:.2f} images/sec"
        )


if __name__=="__main__":
    main()
