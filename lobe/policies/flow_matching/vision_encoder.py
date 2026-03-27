"""Vision encoder for FM policy — ResNet18 with global average pooling.

Produces a 512-d feature vector per image (vs DiffusionRgbEncoder's 64-d spatial softmax).
Matches VITA's observer architecture for stronger visual conditioning.
"""

from __future__ import annotations

import torch
import torchvision
from torch import Tensor, nn


class ResNetPoolEncoder(nn.Module):
    """ResNet18 + GlobalAvgPool vision encoder. 512-d output per image.

    Uses ImageNet-pretrained weights with frozen BatchNorm (standard for fine-tuning).
    Optionally resizes and center-crops images.
    """

    feature_dim: int = 512

    def __init__(self, resize_shape: tuple[int, int] | None = None, crop_shape: tuple[int, int] | None = None):
        super().__init__()
        # ImageNet-pretrained ResNet18, remove final fc
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # up to last conv (512, H/32, W/32)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze BatchNorm (standard for fine-tuning, matches VITA)
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # ImageNet normalization (pretrained backbone expects this)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Optional resize/crop
        self.resize = torchvision.transforms.Resize(resize_shape) if resize_shape else None
        self.crop = torchvision.transforms.CenterCrop(crop_shape) if crop_shape else None

    def forward(self, x: Tensor) -> Tensor:
        """Encode images to 512-d features.

        Args:
            x: (B, C, H, W) images in [0, 1].
        Returns:
            (B, 512) feature vectors.
        """
        # ImageNet normalize
        x = (x - self.img_mean) / self.img_std
        if self.resize is not None:
            x = self.resize(x)
        if self.crop is not None:
            x = self.crop(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)  # (B, 512)
        return x
