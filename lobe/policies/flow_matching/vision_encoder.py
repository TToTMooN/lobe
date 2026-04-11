"""Vision encoders for FM/Diffusion policies.

ResNetPoolEncoder: ResNet18 + GlobalAvgPool → 512-d (original, fast).
DINOv2Encoder: DINOv2 ViT + CLS token → 384/768/1024-d (pretrained, stronger features).
SigLIPEncoder: SigLIP ViT + pooled output → 768/1152-d (vision-language pretrained).
"""

from __future__ import annotations

import torch
import torchvision
from torch import Tensor, nn

# Feature dimensions for each supported encoder
ENCODER_FEATURE_DIMS: dict[str, int] = {
    "resnet18": 512,
    "resnet50": 2048,
    "dinov2_small": 384,
    "dinov2_base": 768,
    "dinov2_large": 1024,
    "siglip_base": 768,
    "siglip_large": 1152,
}


def get_feature_dim(encoder_name: str) -> int:
    if encoder_name not in ENCODER_FEATURE_DIMS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from {list(ENCODER_FEATURE_DIMS.keys())}")
    return ENCODER_FEATURE_DIMS[encoder_name]


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
        x = (x - self.img_mean) / self.img_std
        if self.resize is not None:
            x = self.resize(x)
        if self.crop is not None:
            x = self.crop(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)  # (B, 512)
        return x


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT encoder — strong self-supervised visual features.

    Uses the CLS token output. Frozen by default (fine-tune with frozen=False).
    DINOv2 models are excellent for spatial understanding and transfer well to robotics.
    """

    _MODEL_IDS = {
        "dinov2_small": "facebook/dinov2-small",
        "dinov2_base": "facebook/dinov2-base",
        "dinov2_large": "facebook/dinov2-large",
    }

    def __init__(
        self,
        model_name: str = "dinov2_base",
        frozen: bool = True,
        resize_shape: tuple[int, int] | None = (224, 224),
    ):
        super().__init__()
        from transformers import AutoModel

        hf_id = self._MODEL_IDS.get(model_name, model_name)
        self.model = AutoModel.from_pretrained(hf_id)
        self.feature_dim = self.model.config.hidden_size
        self.frozen = frozen

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False

        # DINOv2 expects ImageNet-normalized 224×224 images
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = torchvision.transforms.Resize(resize_shape) if resize_shape else None

    def forward(self, x: Tensor) -> Tensor:
        """Encode images to feature vectors via CLS token.

        Args:
            x: (B, C, H, W) images in [0, 1].
        Returns:
            (B, feature_dim) feature vectors.
        """
        x = (x - self.img_mean) / self.img_std
        if self.resize is not None:
            x = self.resize(x)
        if self.frozen:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)
        return out.last_hidden_state[:, 0, :]  # CLS token


class SigLIPEncoder(nn.Module):
    """SigLIP vision encoder — vision-language pretrained features.

    SigLIP provides strong vision-language aligned features, used by pi0 and GROOT.
    Better semantic understanding than DINOv2 for language-conditioned policies.
    """

    _MODEL_IDS = {
        "siglip_base": "google/siglip-base-patch16-224",
        "siglip_large": "google/siglip-large-patch16-384",
    }

    def __init__(
        self,
        model_name: str = "siglip_base",
        frozen: bool = True,
        resize_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        from transformers import SiglipVisionModel

        hf_id = self._MODEL_IDS.get(model_name, model_name)
        self.model = SiglipVisionModel.from_pretrained(hf_id)
        self.feature_dim = self.model.config.hidden_size
        self.frozen = frozen

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False

        # SigLIP has its own normalization built into the processor, but we handle it here
        # for consistency with the pipeline (images arrive as [0,1] tensors)
        self.register_buffer("img_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        # Default resize: 224 for base, 384 for large
        if resize_shape is None:
            img_size = self.model.config.image_size
            resize_shape = (img_size, img_size)
        self.resize = torchvision.transforms.Resize(resize_shape) if resize_shape else None

    def forward(self, x: Tensor) -> Tensor:
        """Encode images to pooled feature vectors.

        Args:
            x: (B, C, H, W) images in [0, 1].
        Returns:
            (B, feature_dim) feature vectors.
        """
        x = (x - self.img_mean) / self.img_std
        if self.resize is not None:
            x = self.resize(x)
        if self.frozen:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)
        return out.pooler_output  # (B, feature_dim)


def create_vision_encoder(
    encoder_name: str,
    resize_shape: tuple[int, int] | None = None,
    crop_shape: tuple[int, int] | None = None,
    frozen: bool = True,
) -> nn.Module:
    """Factory for vision encoders. Returns an encoder with a `.feature_dim` attribute."""
    if encoder_name in ("resnet18", "resnet50"):
        return ResNetPoolEncoder(resize_shape=resize_shape, crop_shape=crop_shape)
    elif encoder_name.startswith("dinov2"):
        return DINOv2Encoder(model_name=encoder_name, frozen=frozen, resize_shape=resize_shape or (224, 224))
    elif encoder_name.startswith("siglip"):
        return SigLIPEncoder(model_name=encoder_name, frozen=frozen, resize_shape=resize_shape)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from {list(ENCODER_FEATURE_DIMS.keys())}")
