# Model definition (CNN / ResNet-18) - implemented in Part B"""CNN / ResNet-18 image classifier for CIFAR-10 (10 classes, 32x32x3 images)."""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """A small from-scratch CNN baseline for 32x32 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _build_resnet18_for_cifar(num_classes: int) -> nn.Module:
    """torchvision's resnet18, adapted for small 32x32 inputs.

    The stock ResNet-18 stem (7x7 stride-2 conv + maxpool) is designed for
    224x224 ImageNet images. Applied directly to 32x32 CIFAR images, it
    throws away too much spatial resolution before the residual blocks even
    start. We swap in the standard CIFAR-friendly stem: a 3x3 stride-1 conv
    and no initial maxpool, so 32x32 detail survives into the network.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    """Factory used by train.py / serve.py to build a model from a config string."""
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _build_resnet18_for_cifar(num_classes)
    if architecture in ("cnn", "simplecnn"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r} (expected 'resnet18' or 'cnn')")