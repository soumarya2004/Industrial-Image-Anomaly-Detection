from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "resnet50", "efficientnet_b0")


def _build_backbone(architecture: str, pretrained: bool)->tuple[nn.Module, int]:
    if architecture=="resnet18":
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net=models.resnet18(weights=weights)
        feature_dim=net.fc.in_features
        net.fc=nn.Identity()
        return net, feature_dim

    if architecture=="resnet50":
        weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net=models.resnet50(weights=weights)
        feature_dim=net.fc.in_features
        net.fc=nn.Identity()
        return net, feature_dim

    if architecture=="efficientnet_b0":
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net=models.efficientnet_b0(weights=weights)
        feature_dim=net.classifier[1].in_features
        net.classifier=nn.Identity()
        return net, feature_dim

    raise ValueError(
        f"Unsupported architecture '{architecture}'. Supported: {SUPPORTED_ARCHITECTURES}"
    )


class DefectClassifier(nn.Module):
    def __init__(
        self,
        architecture: str="resnet18",
        pretrained: bool=True,
        num_classes: int=2,
        dropout: float=0.3,
        finetune_mode: str="partial",
        unfrozen_layers: list[str]|None=None,
    ):
        super().__init__()
        self.architecture=architecture
        self.backbone, feature_dim=_build_backbone(architecture, pretrained)
        self.head=nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.apply_finetune_mode(finetune_mode, unfrozen_layers or [])

    def apply_finetune_mode(self, mode: str, unfrozen_layers: list[str])->None:
        if mode=="frozen":
            for param in self.backbone.parameters():
                param.requires_grad=False
        elif mode=="full":
            for param in self.backbone.parameters():
                param.requires_grad=True
        elif mode=="partial":
            for name, param in self.backbone.named_parameters():
                param.requires_grad=any(name.startswith(prefix) for prefix in unfrozen_layers)
        else:
            raise ValueError(f"Unknown finetune_mode '{mode}'. Expected frozen/partial/full.")
        self.finetune_mode=mode

    def trainable_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor)->torch.Tensor:
        features=self.backbone(x)
        return self.head(features)

    def get_target_layer_for_gradcam(self)->nn.Module:
        if self.architecture in ("resnet18", "resnet50"):
            return self.backbone.layer4[-1]
        if self.architecture=="efficientnet_b0":
            return self.backbone.features[-1]
        raise ValueError(f"No Grad-CAM target layer defined for '{self.architecture}'")
