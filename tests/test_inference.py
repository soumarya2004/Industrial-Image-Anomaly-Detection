from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.inference import load_autoencoder, load_classifier, run_autoencoder_inference, run_classifier_inference
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier


def _make_minimal_classifier_config(image_size=64):
    return{
        "data": {"image_size": image_size},
        "model": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 2,
            "dropout": 0.3,
            "finetune_mode": "frozen",
            "unfrozen_layers": [],
        },
    }


def _make_minimal_autoencoder_config(image_size=32):
    return{
        "data": {"image_size": image_size},
        "model": {"latent_dim": 16, "base_channels": 8, "num_downsample_blocks": 3},
    }


def _save_random_image(path: Path, size=(64, 64)):
    arr = (np.random.rand(*size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestClassifierInference:
    def test_checkpoint_roundtrip_and_inference(self, tmp_path):
        cfg=_make_minimal_classifier_config()
        model=DefectClassifier(
            architecture=cfg["model"]["architecture"], pretrained=False,
            num_classes=cfg["model"]["num_classes"], finetune_mode="frozen",
        )
        ckpt_path=tmp_path / "best_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        device=torch.device("cpu")
        loaded_model, loaded_cfg=load_classifier(str(ckpt_path), device)
        assert loaded_cfg["model"]["architecture"]=="resnet18"

        image_path= tmp_path / "sample.png"
        _save_random_image(image_path, size=(64, 64))

        result=run_classifier_inference(loaded_model, loaded_cfg, str(image_path), device, save_gradcam=False)
        assert result["prediction"] in ("NORMAL", "DEFECTIVE")
        assert 0.0<=result["confidence"]<=1.0
        assert 0.0<=result["anomaly_score"]<=1.0
        assert result["latency_ms"]>=0


class TestAutoencoderInference:
    def test_checkpoint_roundtrip_and_inference(self, tmp_path):
        cfg=_make_minimal_autoencoder_config(image_size=32)
        model=ConvAutoencoder(
            image_size=cfg["data"]["image_size"],
            latent_dim=cfg["model"]["latent_dim"],
            base_channels=cfg["model"]["base_channels"],
            num_downsample_blocks=cfg["model"]["num_downsample_blocks"],
        )
        ckpt_path= tmp_path / "best_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        device=torch.device("cpu")
        loaded_model, loaded_cfg = load_autoencoder(str(ckpt_path), device)

        image_path= tmp_path / "sample.png"
        _save_random_image(image_path, size=(32, 32))

        result=run_autoencoder_inference(loaded_model, loaded_cfg, str(image_path), device, threshold=0.05)
        assert result["prediction"] in ("NORMAL", "DEFECTIVE")
        assert result["anomaly_score"] >= 0.0
        assert Path(result["heatmap_path"]).exists()
