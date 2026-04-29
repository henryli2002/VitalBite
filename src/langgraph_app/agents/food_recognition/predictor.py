import os
import json
import io
import base64
import numpy as np
from typing import Dict
from pathlib import Path

_finetuned_model = None
_finetuned_norm = None
_finetuned_transform = None


def get_model():
    global _finetuned_model, _finetuned_norm, _finetuned_transform
    if _finetuned_model is not None:
        return _finetuned_model, _finetuned_norm, _finetuned_transform

    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T

    current_dir = Path(__file__).parent
    model_dir = current_dir / "model"
    model_path = model_dir / "best_model.pt"
    norm_path = model_dir / "norm_stats.json"

    if not model_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")

    with open(norm_path, "r") as f:
        norm = json.load(f)
    _finetuned_norm = norm

    backbone = models.efficientnet_b0(weights=None)
    # EfficientNet-B0 classifier: 1280 -> 256 -> 5
    backbone.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5),
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    _finetuned_model = (backbone, device)

    _finetuned_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return _finetuned_model, _finetuned_norm, _finetuned_transform


def predict_nutrition(image_bytes: bytes) -> Dict[str, float]:
    import torch
    from PIL import Image as PILImage

    (model, device), norm, transform = get_model()
    target_mean = np.array(norm["target_mean"], dtype=np.float32)
    target_std = np.array(norm["target_std"], dtype=np.float32)

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(img_tensor).cpu().numpy()[0]

    pred_raw = pred_norm * target_std + target_mean
    pred_raw = np.maximum(pred_raw, 0)

    return {
        "calculated_weight_g": round(float(pred_raw[0]), 2),
        "total_calories": round(float(pred_raw[1]), 2),
        "total_fat": round(float(pred_raw[2]), 2),
        "total_carb": round(float(pred_raw[3]), 2),
        "total_protein": round(float(pred_raw[4]), 2),
    }


_UUID_MARKER_RE = None  # lazy-compiled


def _parse_image_uuid(text: str):
    """Return the first UUID found in an <attached_image uuid=.../> marker, or None."""
    global _UUID_MARKER_RE
    if _UUID_MARKER_RE is None:
        import re
        _UUID_MARKER_RE = re.compile(r"<attached_image\s+uuid=([0-9a-f]{32})", re.IGNORECASE)
    m = _UUID_MARKER_RE.search(text or "")
    return m.group(1) if m else None


def _load_image_by_uuid(uuid: str) -> bytes:
    """Search data/images/**/{uuid}.jpg and return bytes, or None if not found."""
    data_root = Path(__file__).resolve().parents[4] / "data" / "images"
    for candidate in data_root.rglob(f"{uuid}.jpg"):
        return candidate.read_bytes()
    return None


def extract_image_bytes(messages) -> bytes:
    for msg in reversed(messages):
        if not (getattr(msg, "type", "") == "human" or msg.__class__.__name__ == "HumanMessage"):
            continue

        content = msg.content

        # Legacy path: multipart list with base64 image_url
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if "base64," in url:
                        return base64.b64decode(url.split("base64,")[-1])

        # Current path: plain text with <attached_image uuid=.../> marker
        if isinstance(content, str):
            uuid = _parse_image_uuid(content)
            if uuid:
                img = _load_image_by_uuid(uuid)
                if img:
                    return img

    return None
