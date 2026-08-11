"""ALIKED + LightGlue geometric reranker. Requires the `ml` extra (onnxruntime, opencv).

Model exports come from the Maple pano-stitching pipeline (fatih/aliked and
LightGlue ONNX community exports). Export tensor names vary between builds,
so both the ALIKED and LightGlue input/output names are constructor params.

Expected ALIKED session I/O (defaults match the fatih/aliked ONNX export):
    input:  `image`       — float32 NCHW, RGB, normalized to [0, 1]
    output: `keypoints`   — float32 (N, 2) pixel coordinates (x, y)
    output: `descriptors` — float32 (N, D)
    output: detection scores — float32 (N,); optional, pass `aliked_scores_name`
            if the export emits them (used to pick the top `max_keypoints`)

Expected LightGlue session I/O (defaults match the community ONNX export):
    inputs:  `kpts0`, `kpts1`   — float32 (1, N, 2) pixel coordinates
             `desc0`, `desc1`   — float32 (1, N, D) descriptors
    outputs: `matches0`         — int64 (1, N0), index into kpts1 or -1 if unmatched
             `mscores0`         — float32 (1, N0), per-match confidence
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "AlikedLightGlueReranker requires the 'ml' extra. Install it with: uv sync --extra ml"
    ) from exc

from deco.rerank.base import MatchResult

_PREFERRED_PROVIDERS = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
_MAX_SIDE = 1024
_MIN_MATCHES_FOR_HOMOGRAPHY = 4
_RANSAC_REPROJ_THRESHOLD = 5.0


def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Resize `image` so its longer side is at most `max_side`, keeping aspect ratio."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.BILINEAR)


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize (max side <= 1024), normalize to [0, 1], return NCHW float32."""
    image = _resize_max_side(image.convert("RGB"), _MAX_SIDE)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return array[np.newaxis, :, :, :].astype(np.float32)


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    return [p for p in _PREFERRED_PROVIDERS if p in available]


def _truncate_keypoints(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    scores: np.ndarray | None,
    max_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most `max_keypoints` keypoint/descriptor rows, aligned.

    With `scores`, keeps the highest-scoring rows (descending score order).
    Without scores, keeps the first rows — ALIKED exports typically emit
    keypoints already sorted by detection score, so this is the same cut.
    """
    if len(keypoints) <= max_keypoints:
        return keypoints, descriptors
    if scores is None:
        return keypoints[:max_keypoints], descriptors[:max_keypoints]
    top = np.argsort(scores)[::-1][:max_keypoints]
    return keypoints[top], descriptors[top]


class AlikedLightGlueReranker:
    """Geometric reranker: ALIKED keypoints + LightGlue matching + RANSAC homography."""

    def __init__(
        self,
        aliked_path: Path,
        lightglue_path: Path,
        max_keypoints: int = 1024,
        aliked_input_name: str = "image",
        aliked_keypoints_name: str = "keypoints",
        aliked_descriptors_name: str = "descriptors",
        aliked_scores_name: str | None = None,
        lightglue_kpts0_name: str = "kpts0",
        lightglue_kpts1_name: str = "kpts1",
        lightglue_desc0_name: str = "desc0",
        lightglue_desc1_name: str = "desc1",
        lightglue_matches_name: str = "matches0",
        lightglue_scores_name: str = "mscores0",
    ) -> None:
        providers = _select_providers()
        self._aliked_session = ort.InferenceSession(str(aliked_path), providers=providers)
        self._lightglue_session = ort.InferenceSession(str(lightglue_path), providers=providers)
        self.max_keypoints = max_keypoints

        self._aliked_input_name = aliked_input_name
        self._aliked_keypoints_name = aliked_keypoints_name
        self._aliked_descriptors_name = aliked_descriptors_name
        self._aliked_scores_name = aliked_scores_name
        self._lightglue_kpts0_name = lightglue_kpts0_name
        self._lightglue_kpts1_name = lightglue_kpts1_name
        self._lightglue_desc0_name = lightglue_desc0_name
        self._lightglue_desc1_name = lightglue_desc1_name
        self._lightglue_matches_name = lightglue_matches_name
        self._lightglue_scores_name = lightglue_scores_name

    def _extract(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Run ALIKED, returning (keypoints (N,2) pixel coords, descriptors (N,D)).

        Truncates to at most `self.max_keypoints` rows: by detection score when
        the export names a scores output (`aliked_scores_name`), otherwise by
        keeping the first rows (ALIKED exports typically emit score-sorted
        keypoints).
        """
        array = _preprocess(image)
        output_names = [self._aliked_keypoints_name, self._aliked_descriptors_name]
        if self._aliked_scores_name is not None:
            output_names.append(self._aliked_scores_name)
        outputs = self._aliked_session.run(output_names, {self._aliked_input_name: array})
        keypoints = np.asarray(outputs[0], dtype=np.float32)
        descriptors = np.asarray(outputs[1], dtype=np.float32)
        scores = None
        if self._aliked_scores_name is not None:
            scores = np.asarray(outputs[2], dtype=np.float32)
            if scores.ndim == 2:  # strip batch dim if present: (1, N) -> (N,)
                scores = scores[0]
        if keypoints.ndim == 3:  # strip batch dim if present: (1, N, 2) -> (N, 2)
            keypoints = keypoints[0]
        if descriptors.ndim == 3:
            descriptors = descriptors[0]
        return _truncate_keypoints(keypoints, descriptors, scores, self.max_keypoints)

    def _match_keypoints(
        self,
        kpts0: np.ndarray,
        desc0: np.ndarray,
        kpts1: np.ndarray,
        desc1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run LightGlue, returning matched (src_points, dst_points) pixel coordinate pairs."""
        inputs = {
            self._lightglue_kpts0_name: kpts0[np.newaxis, :, :].astype(np.float32),
            self._lightglue_kpts1_name: kpts1[np.newaxis, :, :].astype(np.float32),
            self._lightglue_desc0_name: desc0[np.newaxis, :, :].astype(np.float32),
            self._lightglue_desc1_name: desc1[np.newaxis, :, :].astype(np.float32),
        }
        outputs = self._lightglue_session.run(
            [self._lightglue_matches_name, self._lightglue_scores_name], inputs
        )
        matches0 = np.asarray(outputs[0])
        if matches0.ndim == 2:  # strip batch dim: (1, N0) -> (N0,)
            matches0 = matches0[0]
        matches0 = matches0.astype(np.int64)

        valid = matches0 > -1
        src = kpts0[valid]
        dst = kpts1[matches0[valid]]
        return src, dst

    def match(self, query: Image.Image, candidate: Image.Image, candidate_id: str) -> MatchResult:
        kpts0, desc0 = self._extract(query)
        kpts1, desc1 = self._extract(candidate)

        if len(kpts0) == 0 or len(kpts1) == 0:
            return MatchResult(inliers=0, total_matches=0)

        src, dst = self._match_keypoints(kpts0, desc0, kpts1, desc1)
        total_matches = len(src)

        if total_matches < _MIN_MATCHES_FOR_HOMOGRAPHY:
            return MatchResult(inliers=0, total_matches=total_matches)

        _homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD)
        inliers = int(mask.sum()) if mask is not None else 0
        return MatchResult(inliers=inliers, total_matches=total_matches)
