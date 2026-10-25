"""DINOv2 ONNX embedder. Requires the `ml` extra (onnxruntime)."""

from pathlib import Path

import numpy as np
from PIL import Image

from deco.embed.embedder import l2_normalize

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "DinoV2OnnxEmbedder requires the 'ml' extra. Install it with: uv sync --extra ml"
    ) from exc

_PREFERRED_PROVIDERS = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
_RESIZE_SHORT_SIDE = 256
_CROP_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize shorter side to 256, center-crop 224, normalize, return NCHW float32."""
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        new_width = _RESIZE_SHORT_SIDE
        new_height = round(height * _RESIZE_SHORT_SIDE / width)
    else:
        new_height = _RESIZE_SHORT_SIDE
        new_width = round(width * _RESIZE_SHORT_SIDE / height)
    image = image.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - _CROP_SIZE) // 2
    top = (new_height - _CROP_SIZE) // 2
    image = image.crop((left, top, left + _CROP_SIZE, top + _CROP_SIZE))

    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return array[np.newaxis, :, :, :].astype(np.float32)  # add batch dim -> NCHW


class DinoV2OnnxEmbedder:
    """Embeds images with a DINOv2 model exported to ONNX."""

    def __init__(self, model_path: Path, model_version: str) -> None:
        available = ort.get_available_providers()
        providers = [p for p in _PREFERRED_PROVIDERS if p in available]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.model_version = model_version

    def embed(self, image: Image.Image) -> np.ndarray:
        array = _preprocess(image)
        (output,) = self._session.run(None, {self._input_name: array})
        if output.ndim == 3:
            output = output.mean(axis=1)  # mean-pool patch tokens: (B, tokens, D) -> (B, D)
        vector = output[0].astype(np.float32)
        return l2_normalize(vector)
