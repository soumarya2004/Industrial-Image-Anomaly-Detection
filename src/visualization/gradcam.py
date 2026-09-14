from __future__ import annotations
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.classifier import DefectClassifier


def generate_gradcam_overlay(
    model: DefectClassifier,
    image_tensor: torch.Tensor,
    image_rgb_float01: np.ndarray,
    target_class: int,
    device: torch.device,
)->np.ndarray:
    model.eval()
    target_layer=model.get_target_layer_for_gradcam()
    cam=GradCAM(model=model, target_layers=[target_layer])

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    input_tensor=image_tensor.to(device).clone().requires_grad_(True)
    grayscale_cam=cam(
        input_tensor=image_tensor.to(device),
        targets=[ClassifierOutputTarget(target_class)],
    )[0]  #(H, W) in [0, 1]

    overlay=show_cam_on_image(image_rgb_float01, grayscale_cam, use_rgb=True)
    return overlay
