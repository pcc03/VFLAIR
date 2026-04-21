import os, sys

sys.path.append(os.pardir)

import torch.nn as nn


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # vgg16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # vgg19
}

# VFL split inputs (e.g., CIFAR10 split to 16x32) need fewer pooling layers.
cfgs_vfl = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],  # vgg16 without last pool
}


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        # For VFL use: return the flattened convolutional features
        # (do not apply avgpool or classifier here). This produces the
        # spatially-flattened embedding vector that the global head
        # expects as `output_dim` in the config (e.g., 10752).
        x = x.view(x.size(0), -1)
        return x


def _parse_vgg_args(args, kwargs):
    if len(args) == 1:
        num_classes = args[0]
        in_channels = kwargs.get("in_channels", 3)
        return num_classes, in_channels
    if len(args) == 2:
        # called as (input_dim, output_dim) by LoadModels for non-CNN types
        num_classes = args[1]
        in_channels = kwargs.get("in_channels", 3)
        return num_classes, in_channels
    raise TypeError("vgg expects (num_classes) or (input_dim, output_dim)")


def vgg11(*args, **kwargs):
    num_classes, in_channels = _parse_vgg_args(args, kwargs)
    return VGG(make_layers(cfgs["A"], in_channels=in_channels), num_classes)


def vgg13(*args, **kwargs):
    num_classes, in_channels = _parse_vgg_args(args, kwargs)
    return VGG(make_layers(cfgs["B"], in_channels=in_channels), num_classes)


def vgg16(*args, **kwargs):
    num_classes, in_channels = _parse_vgg_args(args, kwargs)
    return VGG(make_layers(cfgs["D"], in_channels=in_channels), num_classes)


def vgg19(*args, **kwargs):
    num_classes, in_channels = _parse_vgg_args(args, kwargs)
    return VGG(make_layers(cfgs["E"], in_channels=in_channels), num_classes)


def vgg16_vfl(*args, **kwargs):
    num_classes, in_channels = _parse_vgg_args(args, kwargs)
    # Use the original VGG cfg (full pooling) for ADI-style VFL:
    # when inputs are resized to ImageNet-like dimensions and then
    # split (e.g., 224x112 halves), the full VGG pooling produces
    # a spatial map of 7x3 which flattens to 512*7*3 = 10752.
    return VGG(make_layers(cfgs["D"], in_channels=in_channels), num_classes)
