import torch.nn as nn
import torchvision.models as tv_models


def _parse_args(args, kwargs):
    if len(args) == 1:
        out_dim = args[0]
    elif len(args) == 2:
        # LoadModels may call non-CNN models as (input_dim, output_dim).
        out_dim = args[1]
    else:
        out_dim = kwargs.get("output_dim", 1024)
    in_channels = kwargs.get("in_channels", 3)
    return in_channels, out_dim


class VGG16NewBottom(nn.Module):
    """
    Bottom model for CIFAR10 VFL with horizontal split.
    Input per party: [B, 3, 32, 16] (equal horizontal half).
    Output embedding: flatten([B, 512, 2, 1]) => [B, 1024].
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # VGG16-style conv stack with 4 pooling stages (not 5), so width=16
        # remains valid and produces a compact embedding for each party.
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
        ]
        self.features = self._make_layers(cfg, in_channels)

    @staticmethod
    def _make_layers(cfg, in_channels):
        layers = []
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def vgg16_new(*args, **kwargs):
    _in_channels, _out_dim = _parse_args(args, kwargs)
    # _out_dim is intentionally ignored because this bottom model is an
    # embedding extractor with fixed output dim for 32x16 input.
    return VGG16NewBottom(in_channels=_in_channels)


class VGG16ImageNetBottom(nn.Module):
    """
    Bottom model using ImageNet-pretrained torchvision VGG16 features.
    Returns a 512-d embedding via global average pooling + flatten.

    Intended for **full** small inputs such as CIFAR **32×32** (typical ``k=1``).
    Very non-square small tensors (e.g. CIFAR **32×16** halves for ``k=2``) are not
    supported here: five ``MaxPool2d(2)`` stages can shrink one side to zero. Use
    ``vgg16_new`` (``VGG16NewBottom``) for horizontal half-images—much less memory
    than two ImageNet VGG16 backbones.
    """

    def __init__(self):
        super().__init__()
        try:
            # torchvision >= 0.13
            backbone = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            # Backward compatibility for older torchvision
            backbone = tv_models.vgg16(pretrained=True)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        if min(h, w) < 32:
            raise ValueError(
                "vgg16_imagenet_bottom: input spatial size %dx%d is too small or too "
                "non-square for torchvision VGG16 features (needs min(H,W) >= 32, "
                "e.g. full CIFAR 32x32 for k=1). For CIFAR k=2 horizontal halves "
                "(32x16), use model type `vgg16_new` with `return_embedding: true` "
                "and `embedding_dim: 1024` in the party config instead of two "
                "ImageNet VGG16s—similar activation footprint per party, much lower "
                "memory than upsampling to 32x32 on two full pretrained backbones."
                % (h, w)
            )
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def vgg16_imagenet_bottom(*args, **kwargs):
    _in_channels, _out_dim = _parse_args(args, kwargs)
    # In-channels and out-dim are fixed by pretrained VGG16 features + GAP.
    return VGG16ImageNetBottom()
