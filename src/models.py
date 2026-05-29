from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth per sample.

    This class is provided so students do not need to depend on timm internals
    for ConvNetV3 regularization experiments.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------------------------------------------------------------
# Problem 1: Linear Classifier
# -----------------------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int = 3072, num_classes: int = 10):
        super().__init__()
        # Fill this: define a linear classifier head.
        # Hint: CIFAR-10 image input is flattened in forward().
        raise NotImplementedError("Problem 1: implement LinearClassifier.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], -1))
        return self.head(x)


# -----------------------------------------------------------------------------
# Problem 2: Multilayer Perceptron (MLP)
# -----------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_channels: int = 3072, hidden_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fill this: apply Linear followed by ReLU.
        raise NotImplementedError("Problem 2: implement MLPBlock.forward")


class MLP(nn.Module):
    def __init__(self, in_channels: int = 3072, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        # Fill this: build two hidden MLPBlock layers using nn.ModuleList.
        raise NotImplementedError("Problem 2: implement MLP.__init__")
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], -1))
        # Fill this: pass x through all hidden layers before self.head.
        raise NotImplementedError("Problem 2: implement MLP.forward")
        return self.head(x)


# -----------------------------------------------------------------------------
# Problem 3: Convolutional Neural Network (ConvNetV1)
# -----------------------------------------------------------------------------
class ConvNetV1Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class ConvNetV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dims: Sequence[int] = (64, 128, 256, 512),
        strides: Sequence[int] = (1, 2, 2, 1),
        num_classes: int = 10,
    ):
        super().__init__()
        if len(dims) != len(strides):
            raise ValueError("dims and strides must have the same length.")

        # Fill this: build four ConvNetV1Block layers using nn.ModuleList.
        # Requirements: 3x3 kernels, padding 1, channels from dims, strides from strides.
        raise NotImplementedError("Problem 3: implement ConvNetV1.__init__")
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-1, -2))
        return self.head(x)


# -----------------------------------------------------------------------------
# Problem 4: Advanced Convolutional Neural Network (ConvNetV2)
# -----------------------------------------------------------------------------
class ConvNetV2DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ConvNetV2Block(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, droppath: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Fill this: define conv1/norm1/conv2/norm2 for a residual block.
        raise NotImplementedError("Problem 4: implement ConvNetV2Block.__init__")
        self.droppath = DropPath(droppath) if droppath > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = F.relu(self.norm2(self.conv2(h)), inplace=True)
        return x + self.droppath(h)


class ConvNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        blocks: Sequence[int] = (1, 2, 6, 1),
        dims: Sequence[int] = (96, 192, 384, 768),
        strides: Sequence[int] = (1, 2, 2, 1),
        num_classes: int = 10,
        droppath: float = 0.0,
        dropout: float = 0.0,
        droprate: float | None = None,
    ):
        super().__init__()
        if droprate is not None:
            dropout = droprate
        if not (len(blocks) == len(dims) == len(strides)):
            raise ValueError("blocks, dims, and strides must have the same length.")

        self.downsamples = nn.ModuleList()
        # Fill this: add one ConvNetV2DownsampleBlock per stage.
        raise NotImplementedError("Problem 4: implement ConvNetV2 downsamples")

        self.layers = nn.ModuleList()
        # Fill this: add the requested number of ConvNetV2Block layers per stage.
        # Hint: use nn.Sequential for each stage and optionally vary droppath per block.
        raise NotImplementedError("Problem 4: implement ConvNetV2 stages")

        self.norm = nn.BatchNorm1d(dims[-1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsamples, self.layers):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(-1, -2))
        x = self.dropout(self.norm(x))
        return self.head(x)


# Backward-compatible aliases for last year's naming.
ConvNet = ConvNetV1
ResNet = ConvNetV2


# -----------------------------------------------------------------------------
# Problem 6: ResNet-50 experiments
# -----------------------------------------------------------------------------
def build_resnet50(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "ResNet50 experiments require timm. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = timm.create_model(
        model_name="resnet50.a1_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model.conv1.stride = (1, 1)
    model.maxpool.stride = (1, 1)
    return model


def build_model(model_cfg: Dict) -> nn.Module:
    name = str(model_cfg.get("name", "linear")).lower()
    num_classes = int(model_cfg.get("num_classes", 10))

    if name in {"linear", "linear_classifier", "linearclassifier"}:
        return LinearClassifier(
            in_channels=int(model_cfg.get("in_channels", 3072)),
            num_classes=num_classes,
        )
    if name in {"mlp", "mlp-512", "mlp-2048"}:
        return MLP(
            in_channels=int(model_cfg.get("in_channels", 3072)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_classes=num_classes,
        )
    if name in {"convnet_v1", "convnet", "convnetv1"}:
        return ConvNetV1(
            in_channels=int(model_cfg.get("in_channels", 3)),
            dims=model_cfg.get("dims", [64, 128, 256, 512]),
            strides=model_cfg.get("strides", [1, 2, 2, 1]),
            num_classes=num_classes,
        )
    if name in {"convnet_v2", "convnet_v3_reg", "convnet_v3_aug", "resnet", "convnetv2"}:
        return ConvNetV2(
            in_channels=int(model_cfg.get("in_channels", 3)),
            blocks=model_cfg.get("blocks", [1, 2, 6, 1]),
            dims=model_cfg.get("dims", [96, 192, 384, 768]),
            strides=model_cfg.get("strides", [1, 2, 2, 1]),
            num_classes=num_classes,
            droppath=float(model_cfg.get("droppath", 0.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name in {"resnet50_scratch", "resnet50_finetune", "resnet50"}:
        return build_resnet50(
            num_classes=num_classes,
            pretrained=bool(model_cfg.get("pretrained", name == "resnet50_finetune")),
        )
    raise ValueError(f"Unknown model name: {name}")
