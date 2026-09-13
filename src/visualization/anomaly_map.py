from __future__ import annotations
import numpy as np
import torch
from src.data.preprocessing import compute_reconstruction_heatmap, overlay_heatmap
from src.models.autoencoder import ConvAutoencoder


def tensor_to_rgb01(tensor: torch.Tensor)->np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def generate_anomaly_visualization(
    model: ConvAutoencoder, image_tensor: torch.Tensor, device: torch.device
)->dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        image_tensor=image_tensor.to(device)
        reconstructed=model(image_tensor)

    original_rgb=tensor_to_rgb01(image_tensor[0])
    reconstructed_rgb=tensor_to_rgb01(reconstructed[0])

    heatmap_gray=compute_reconstruction_heatmap(original_rgb, reconstructed_rgb)

    original_bgr_uint8=(original_rgb[:, :, ::-1]*255).astype(np.uint8)
    overlay_bgr=overlay_heatmap(original_bgr_uint8, heatmap_gray)

    anomaly_score=float(((original_rgb-reconstructed_rgb)**2).mean())

    return{
        "original_rgb": original_rgb,
        "reconstructed_rgb": reconstructed_rgb,
        "heatmap_gray": heatmap_gray,
        "overlay_bgr": overlay_bgr,
        "anomaly_score": anomaly_score,
    }
