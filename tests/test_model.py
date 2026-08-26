"""Unit tests for src/model.py."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["resnet18", "cnn"])
def test_forward_pass_output_shape(architecture):
    model = get_model(architecture, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_cnn_architecture_returns_simplecnn():
    model = get_model("cnn", num_classes=10)
    assert isinstance(model, SimpleCNN)


def test_unknown_architecture_raises_value_error():
    with pytest.raises(ValueError):
        get_model("not-a-real-architecture", num_classes=10)


def test_custom_num_classes_changes_output_dim():
    model = get_model("resnet18", num_classes=4)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4)
