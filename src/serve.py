"""Flask app serving predictions from a trained CIFAR-10 checkpoint.

Exposes:
  GET  /health   -> 200 if a model checkpoint is loaded, 503 otherwise
  POST /predict  -> multipart form field "image" -> class probabilities
"""

import io
import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image

from dataset import get_transforms
from model import get_model

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = Flask(__name__)

_INFERENCE_TRANSFORM = get_transforms(train=False)  # ToTensor + CIFAR-10 normalization
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None  # populated by load_model(); stays None if loading fails


def load_model() -> None:
    """Load the checkpoint at CHECKPOINT_PATH. Failure is logged, not fatal -
    /health correctly reports the model as unavailable so a Kubernetes
    readiness probe can hold traffic until a real checkpoint appears (e.g.
    once the training Job has written one to the shared PVC)."""
    global _model
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    if not os.path.exists(checkpoint_path):
        app.logger.warning(
            "Checkpoint not found at %s - /predict will return 503 until it appears.",
            checkpoint_path,
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location=_device)
    model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    _model = model
    app.logger.info(
        "Loaded checkpoint from %s (val_accuracy=%.4f)",
        checkpoint_path, checkpoint.get("val_accuracy", -1),
    )


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    tensor = _INFERENCE_TRANSFORM(image)
    return tensor.unsqueeze(0).to(_device)


@app.get("/health")
def health():
    if _model is None:
        return jsonify({"status": "unavailable", "model_loaded": False}), 503
    return jsonify({"status": "ok", "model_loaded": True}), 200


@app.post("/predict")
def predict():
    if _model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file in multipart form data"}), 400

    try:
        image_bytes = request.files["image"].read()
        input_tensor = preprocess(image_bytes)
    except Exception as exc:  # invalid/corrupt image upload
        return jsonify({"error": f"could not process image: {exc}"}), 400

    with torch.no_grad():
        logits = _model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return jsonify({
        "predicted_class": CIFAR10_CLASSES[predicted_index],
        "predicted_index": predicted_index,
        "probabilities": {
            CIFAR10_CLASSES[i]: round(p, 4) for i, p in enumerate(probabilities)
        },
    })


# Loaded at import time so it works both for `python serve.py` (dev) and a
# WSGI server like gunicorn importing this module (production, in Docker).
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
