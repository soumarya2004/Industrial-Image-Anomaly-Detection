from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
from src.models.autoencoder import ConvAutoencoder, reconstruction_error_per_image
from src.models.classifier import DefectClassifier


class TestDefectClassifier:
    def test_forward_pass_output_shape(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False, num_classes=2)
        x=torch.randn(4, 3, 224, 224)
        logits=model(x)
        assert logits.shape==(4, 2)

    def test_frozen_mode_freezes_all_backbone_params(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="frozen")
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

    def test_full_mode_unfreezes_all_backbone_params(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="full")
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_partial_mode_only_unfreezes_specified_layers(self):
        model=DefectClassifier(
            architecture="resnet18", pretrained=False, finetune_mode="partial",
            unfrozen_layers=["layer4"],
        )
        for name, param in model.backbone.named_parameters():
            if name.startswith("layer4"):
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_trainable_param_count_less_than_total_when_frozen(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="frozen")
        assert model.trainable_parameter_count() < model.total_parameter_count()

    def test_invalid_finetune_mode_raises(self):
        with pytest.raises(ValueError):
            DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="bogus")

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError):
            DefectClassifier(architecture="not_a_real_arch", pretrained=False)

    def test_gradcam_target_layer_exists(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False)
        layer=model.get_target_layer_for_gradcam()
        assert isinstance(layer, torch.nn.Module)

    def test_gradcam_works_with_frozen_backbone(self):
            from src.visualization.gradcam import generate_gradcam_overlay
    
            model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="frozen")
            model.eval()
            image_tensor=torch.randn(1, 3, 64, 64)
            image_rgb01=torch.rand(64, 64, 3).numpy()
            overlay=generate_gradcam_overlay(
                model, image_tensor, image_rgb01, target_class=0, device=torch.device("cpu")
            )
            assert overlay.shape==(64, 64, 3)
            assert overlay.dtype==np.uint8

    def test_checkpoint_save_and_load_roundtrip(self):
        model=DefectClassifier(architecture="resnet18", pretrained=False)
        model.eval()  # disable dropout so the forward pass is deterministic
        x=torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            original_output=model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path= Path(tmpdir) / "model.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            reloaded=DefectClassifier(architecture="resnet18", pretrained=False)
            reloaded.eval()
            checkpoint=torch.load(ckpt_path, map_location="cpu", weights_only=False)
            reloaded.load_state_dict(checkpoint["model_state_dict"])
            with torch.no_grad():
                reloaded_output = reloaded(x)

        assert torch.allclose(original_output, reloaded_output)


class TestConvAutoencoder:
    def test_forward_pass_preserves_shape(self):
        model=ConvAutoencoder(image_size=128, latent_dim=64, num_downsample_blocks=4)
        x=torch.rand(4, 3, 128, 128)
        reconstruction=model(x)
        assert reconstruction.shape==x.shape

    def test_output_is_in_valid_pixel_range(self):
        model=ConvAutoencoder(image_size=64, latent_dim=32, num_downsample_blocks=3)
        x=torch.rand(2, 3, 64, 64)
        reconstruction=model(x)
        assert reconstruction.min()>=0.0
        assert reconstruction.max()<=1.0

    def test_latent_vector_has_correct_dimension(self):
        model=ConvAutoencoder(image_size=128, latent_dim=100, num_downsample_blocks=4)
        x=torch.rand(3, 3, 128, 128)
        z=model.encode(x)
        assert z.shape==(3, 100)

    def test_incompatible_image_size_raises(self):
        with pytest.raises(ValueError):
            ConvAutoencoder(image_size=100, num_downsample_blocks=4)  # 100 not divisible by 16

    def test_reconstruction_error_shape_and_nonnegativity(self):
        original=torch.rand(5, 3, 32, 32)
        reconstructed=torch.rand(5, 3, 32, 32)
        errors=reconstruction_error_per_image(original, reconstructed)
        assert errors.shape==(5,)
        assert torch.all(errors>=0)

    def test_reconstruction_error_is_zero_for_identical_images(self):
        original=torch.rand(2, 3, 16, 16)
        errors=reconstruction_error_per_image(original, original.clone())
        assert torch.allclose(errors, torch.zeros(2), atol=1e-6)

    def test_different_latent_dims_produce_different_param_counts(self):
        small=ConvAutoencoder(image_size=64, latent_dim=32, num_downsample_blocks=3)
        large=ConvAutoencoder(image_size=64, latent_dim=512, num_downsample_blocks=3)
        small_params=sum(p.numel() for p in small.parameters())
        large_params=sum(p.numel() for p in large.parameters())
        assert large_params>small_params
