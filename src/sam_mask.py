# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Segment Anything masking for the ``sam`` arm of the 2 by 3 ablation.

The detector gives one axis-aligned box ``B`` per animal. SAM is prompted with
``B`` only and returns a binary mask ``M`` of the coat inside that box. The
pixels handed to the recogniser are

    I_sam = Crop(I * M, B_M)

where ``B_M`` is the tight box of the foreground pixels of ``M`` (see
``paper/method.tex``, eq. ``sam-crop``). Everything outside the mask is zeroed,
so grass, hard shadow, and limbs are absent from the crop rather than merely
boxed out.

Design notes
------------
* The segmenter is loaded lazily, on the first :meth:`SamMasker.mask` call, so
  running the ``bbox`` arm never imports or allocates a SAM model.
* Masks are cached on disk, keyed by image identity plus the detection box plus
  a fingerprint of the segmenter configuration. Repeated epochs over the same
  manifest therefore segment each image once and then read one small ``.npz``.
* Missing weights raise :class:`SamWeightsMissing`. There is deliberately no
  silent fallback to the plain box crop: a fallback would make the two arms of
  the ablation byte-identical, which is the defect this module exists to fix.
* Backends: the reference ``segment_anything`` package (``vit_h``/``vit_l``/
  ``vit_b``), the ``mobile_sam`` package (``vit_t``), or ``ultralytics`` SAM,
  which is already a dependency of the detector path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

try:  # opencv is used for connected components and hole filling only.
    import cv2
except ImportError:  # pragma: no cover - exercised only on minimal installs
    cv2 = None  # type: ignore[assignment]

# Bump when the meaning of a cached mask changes, so stale entries are ignored.
CACHE_SCHEMA = "sam-mask-v1"

SAM_MODEL_TYPES = ("vit_h", "vit_l", "vit_b", "vit_t")
SAM_BACKENDS = ("auto", "segment_anything", "mobile_sam", "ultralytics")

Box = tuple[float, float, float, float]

# Model types served by the ``mobile_sam`` package rather than ``segment_anything``.
_MOBILE_MODEL_TYPES = frozenset({"vit_t", "tiny_vit", "vit_tiny"})


class SamError(RuntimeError):
    """Base class for a masking failure that must stop the run."""


class SamWeightsMissing(SamError):
    """Raised when the configured SAM checkpoint does not exist."""


class SamPackageMissing(SamError):
    """Raised when no segmenter package is importable."""


class DegenerateMaskError(SamError):
    """Raised when SAM returns a mask too small to be a coat."""


def _warn(message: str) -> None:
    print(f"[sam_mask] {message}", file=sys.stderr, flush=True)


def image_identity(path: str | Path) -> str:
    """Stable id for a source image: relative-ish name plus size and mtime.

    Two different frames that happen to share a filename in different folders
    stay distinct, and replacing a frame in place invalidates its cached masks.
    """

    file_path = Path(path)
    name = file_path.name
    try:
        stat = file_path.stat()
        return f"{name}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return name


def clip_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a detector box to the image and return integer ``(x1, y1, x2, y2)``."""

    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(np.clip(round(x1), 0, max(width - 1, 0)))
    y1 = int(np.clip(round(y1), 0, max(height - 1, 0)))
    x2 = int(np.clip(round(x2), x1 + 1, width))
    y2 = int(np.clip(round(y2), y1 + 1, height))
    return x1, y1, x2, y2


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected blob. Falls back to the input without opencv."""

    if cv2 is None:
        return mask
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of the mask (coat patches between legs, glare, ...)."""

    if cv2 is None:
        return mask
    mask_u8 = mask.astype(np.uint8)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return (padded | holes)[1:-1, 1:-1].astype(bool)


def refine_mask(
    raw: np.ndarray,
    box: Box,
    *,
    min_area_frac: float = 0.005,
    largest_component: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """Post-process a raw SAM mask into the mask that is actually applied.

    Steps, in order: binarise, clip to the detection box (a box prompt defines
    the animal region, so anything SAM leaks outside it is not coat), keep the
    largest connected component, fill interior holes, and reject a mask that is
    smaller than ``min_area_frac`` of the box.
    """

    mask = np.asarray(raw)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise SamError(f"Expected a single mask, got shape {tuple(mask.shape)}")
    if mask.dtype != np.bool_:
        mask = mask > 0
    if mask.ndim != 2:
        raise SamError(f"Expected a 2-D mask, got shape {tuple(mask.shape)}")

    height, width = mask.shape
    x1, y1, x2, y2 = clip_box(box, width, height)
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    if largest_component:
        clipped = _keep_largest_component(clipped)
    if fill_holes:
        clipped = _fill_holes(clipped)

    area = int(clipped.sum())
    box_area = max((x2 - x1) * (y2 - y1), 1)
    if area < max(1, int(round(min_area_frac * box_area))):
        raise DegenerateMaskError(
            f"SAM mask covers {area} px of a {box_area} px box "
            f"({area / box_area:.4%}) in box {x1, y1, x2, y2}. "
            "Refusing to emit an empty coat crop; check the detector box and the "
            "SAM checkpoint."
        )
    return clipped


def purify_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Return ``Crop(I * M, B_M)``: masked pixels, tight-cropped to the mask.

    The crop is exactly the tight bounding box of the foreground, so the input
    aspect ratio is preserved rather than forced to the box ratio. Background
    pixels inside ``B_M`` are zero (black), matching the element-wise product in
    ``eq. sam-crop``. The plain ``bbox`` arm in :mod:`src.preprocess` does not
    call this function, so it keeps its original pixels.
    """

    pixels = np.asarray(image.convert("RGB"))
    if mask.shape != pixels.shape[:2]:
        raise SamError(
            f"Mask shape {tuple(mask.shape)} does not match image shape {pixels.shape[:2]}"
        )
    fore = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(fore)
    if xs.size == 0:
        raise DegenerateMaskError("Mask has no foreground pixels; cannot form a crop.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    masked = np.where(fore[..., None], pixels, 0).astype(np.uint8)
    return Image.fromarray(masked[y0:y1, x0:x1])


@dataclass(frozen=True)
class SamConfig:
    """Everything that determines the pixels of a masked crop."""

    weights: str = ""
    model_type: str = "vit_b"
    backend: str = "auto"
    device: str = ""
    cache_dir: str = ""
    min_area_frac: float = 0.005
    multimask: bool = True
    largest_component: bool = True
    fill_holes: bool = True

    def validate(self) -> "SamConfig":
        if self.backend not in SAM_BACKENDS:
            raise ValueError(
                f"sam backend must be one of {', '.join(SAM_BACKENDS)}; got {self.backend!r}"
            )
        if not 0.0 <= float(self.min_area_frac) < 1.0:
            raise ValueError("sam min_area_frac must be in [0, 1).")
        return self

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dependency
            return "cpu"

    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        return "mobile_sam" if self.model_type in _MOBILE_MODEL_TYPES else "segment_anything"

    def weights_fingerprint(self) -> dict[str, Any]:
        path = Path(self.weights) if self.weights else None
        if path is None:
            return {"weights": ""}
        exists = path.is_file()
        stat = path.stat() if exists else None
        return {
            "weights_name": path.name,
            "weights_size": stat.st_size if stat else None,
            "weights_mtime_ns": stat.st_mtime_ns if stat else None,
        }

    def fingerprint(self) -> str:
        payload = {
            "schema": CACHE_SCHEMA,
            "model_type": self.model_type,
            "backend": self.resolved_backend(),
            "min_area_frac": round(float(self.min_area_frac), 6),
            "multimask": bool(self.multimask),
            "largest_component": bool(self.largest_component),
            "fill_holes": bool(self.fill_holes),
            **self.weights_fingerprint(),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]


class MaskCache:
    """Disk cache of masks keyed by image identity plus detection box.

    A cache miss costs one SAM forward pass; a hit costs one ``np.load`` of a
    bit-packed mask. The key covers the image file's size and mtime, so editing
    or replacing a source image invalidates its entry without manual cleanup.
    """

    def __init__(self, root: str | Path | None, config: SamConfig) -> None:
        self.root = Path(root) if root else None
        self.config = config
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, image_id: str, image_size: tuple[int, int], box: Box) -> str:
        x1, y1, x2, y2 = (round(float(v), 2) for v in box)
        payload = "|".join(
            [
                CACHE_SCHEMA,
                str(image_id),
                f"{image_size[0]}x{image_size[1]}",
                f"{x1},{y1},{x2},{y2}",
                self.config.fingerprint(),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.npz"

    def get(self, key: str) -> np.ndarray | None:
        if self.root is None:
            return None
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                shape = tuple(int(v) for v in data["shape"])
                packed = data["mask"]
            n_pixels = shape[0] * shape[1]
            if packed.size * 8 < n_pixels:
                raise ValueError(f"packed mask holds {packed.size * 8} bits, need {n_pixels}")
            mask = np.unpackbits(packed)[:n_pixels].reshape(shape).astype(bool)
        except (OSError, KeyError, ValueError) as exc:
            _warn(f"Discarding unreadable mask cache entry {path}: {exc}")
            self.misses += 1
            return None
        self.hits += 1
        return mask

    def put(self, key: str, mask: np.ndarray) -> None:
        if self.root is None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(np.asarray(mask, dtype=bool).reshape(-1))
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, mask=packed, shape=np.asarray(mask.shape, dtype=np.int64))
        tmp.replace(path)
        self.writes += 1

    def stats(self) -> dict[str, int | str]:
        return {
            "cache_dir": str(self.root) if self.root else "",
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


class _Backend(Protocol):
    name: str

    def predict(self, image_rgb: np.ndarray, box: Box) -> np.ndarray: ...


class _SegmentAnythingBackend:
    """Reference SAM / MobileSAM via ``SamPredictor`` (identical API in both)."""

    def __init__(self, config: SamConfig) -> None:
        self.config = config
        self.name = config.resolved_backend()
        self._predictor: Any = None

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        weights = Path(self.config.weights)
        if not weights.is_file():
            raise SamWeightsMissing(
                f"SAM checkpoint not found: {weights}. The sam arm needs real weights: "
                "download sam_vit_b_01ec64.pth (Segment Anything) or mobile_sam.pt and pass "
                "--sam-weights, or use the bbox arm, which never loads SAM."
            )
        package = self.name
        try:
            if package == "mobile_sam":
                from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
            else:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except ImportError as exc:
            install = "mobile-sam" if package == "mobile_sam" else "segment-anything"
            raise SamPackageMissing(
                f"SAM masking needs the {package!r} package (pip install {install}) "
                "or --sam-backend ultralytics."
            ) from exc
        if self.config.model_type not in sam_model_registry:
            raise SamError(
                f"{package} has no model type {self.config.model_type!r}; "
                f"available: {', '.join(sorted(sam_model_registry))}"
            )
        model = sam_model_registry[self.config.model_type](checkpoint=str(weights))
        model.to(self.config.resolved_device())
        model.eval()
        self._predictor = SamPredictor(model)
        return self._predictor

    def predict(self, image_rgb: np.ndarray, box: Box) -> np.ndarray:
        predictor = self._load()
        predictor.set_image(np.ascontiguousarray(image_rgb))
        masks, scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(box, dtype=np.float32)[None, :],
            multimask_output=bool(self.config.multimask),
        )
        return select_mask(masks, scores)


class _UltralyticsBackend:
    """SAM through ultralytics, which is already required by the detector path."""

    def __init__(self, config: SamConfig) -> None:
        self.config = config
        self.name = "ultralytics"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        weights = Path(self.config.weights)
        if not weights.is_file():
            raise SamWeightsMissing(
                f"SAM checkpoint not found: {weights}. With --sam-backend ultralytics, "
                "point --sam-weights at sam_b.pt / mobile_sam.pt / sam2_t.pt."
            )
        try:
            from ultralytics import SAM  # type: ignore
        except ImportError as exc:
            raise SamPackageMissing(
                "ultralytics is not installed; pip install ultralytics or use "
                "--sam-backend segment_anything."
            ) from exc
        self._model = SAM(str(weights))
        return self._model

    def predict(self, image_rgb: np.ndarray, box: Box) -> np.ndarray:
        import torch.nn.functional as F

        model = self._load()
        results = model.predict(
            source=np.ascontiguousarray(image_rgb),
            bboxes=[list(float(v) for v in box)],
            verbose=False,
            device=self.config.resolved_device(),
        )
        if not results or results[0].masks is None or results[0].masks.data is None:
            raise DegenerateMaskError("ultralytics SAM returned no mask for the box prompt.")
        data = results[0].masks.data
        if data.ndim == 2:
            data = data[None]
        height, width = image_rgb.shape[:2]
        if tuple(data.shape[-2:]) != (height, width):
            data = F.interpolate(data[None].float(), size=(height, width), mode="bilinear")[0]
        soft = data.float()
        # The first mask is the one ultralytics ranks highest for the prompt.
        return (soft[0] > 0.5).cpu().numpy()


def select_mask(masks: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """Pick one mask out of SAM's multimask output.

    Highest predicted-IoU score wins; ties go to the smaller mask, because a
    prompt box on a coat should not resolve to the whole frame.
    """

    masks = np.asarray(masks)
    if masks.ndim == 2:
        return masks.astype(bool)
    if masks.shape[0] == 1:
        return masks[0].astype(bool)
    if scores is None:
        scores = np.zeros(masks.shape[0], dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best = float(np.max(scores))
    candidates = np.flatnonzero(scores >= best - 1e-6)
    pick = int(candidates[np.argmin(areas[candidates])]) if candidates.size else 0
    return masks[pick].astype(bool)


def build_backend(config: SamConfig) -> _Backend:
    """Instantiate the configured backend. The model itself still loads lazily."""

    resolved = config.resolved_backend()
    if resolved == "ultralytics":
        return _UltralyticsBackend(config)
    if resolved in ("segment_anything", "mobile_sam"):
        return _SegmentAnythingBackend(config)
    raise ValueError(f"Unsupported SAM backend: {resolved!r}")


class SamMasker:
    """Lazily-loaded segmenter plus the disk mask cache.

    ``mask`` is the only entry point used by :mod:`src.preprocess`. Passing
    ``backend`` replaces the real segmenter, which is how the test suite
    exercises this path without downloading any weights.
    """

    def __init__(
        self,
        config: SamConfig,
        *,
        backend: _Backend | None = None,
        cache: MaskCache | None = None,
    ) -> None:
        self.config = config.validate()
        self._backend = backend
        self.cache = cache if cache is not None else MaskCache(config.cache_dir, self.config)
        self.loads = 0

    @property
    def backend_name(self) -> str:
        if self._backend is not None:
            return getattr(self._backend, "name", type(self._backend).__name__)
        return self.config.resolved_backend()

    def _ensure_backend(self) -> _Backend:
        if self._backend is None:
            # Missing weights fail here, on first use, loudly and before any crop
            # is written, so a partial sam directory can never masquerade as bbox.
            weights = Path(self.config.weights) if self.config.weights else None
            if weights is None or not weights.is_file():
                raise SamWeightsMissing(
                    f"SAM checkpoint not found: {self.config.weights or '(unset)'}. "
                    "The sam arm needs real weights; there is no silent fallback to the "
                    "box crop."
                )
            self._backend = build_backend(self.config)
            self.loads += 1
        return self._backend

    def mask(
        self,
        image: Image.Image,
        box: Box,
        *,
        image_id: str = "",
        use_cache: bool = True,
    ) -> np.ndarray:
        """Return the refined binary mask for ``box`` in ``image`` coordinates."""

        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        box = tuple(float(v) for v in box)
        key = self.cache.key(image_id, (width, height), box) if use_cache else ""
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None and cached.shape == (height, width):
                return cached
        backend = self._ensure_backend()
        raw = backend.predict(rgb, box)
        mask = refine_mask(
            raw,
            box,
            min_area_frac=self.config.min_area_frac,
            largest_component=self.config.largest_component,
            fill_holes=self.config.fill_holes,
        )
        if use_cache:
            self.cache.put(key, mask)
        return mask

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_type": self.config.model_type,
            "weights": self.config.weights,
            "device": self.config.resolved_device(),
            "loads": self.loads,
            **self.cache.stats(),
        }


def build_masker(
    *,
    weights: str,
    model_type: str = "vit_b",
    backend: str = "auto",
    device: str = "",
    cache_dir: str = "",
    min_area_frac: float = 0.005,
    multimask: bool = True,
) -> SamMasker:
    """Convenience constructor used by the CLI and the scripts."""

    return SamMasker(
        SamConfig(
            weights=weights,
            model_type=model_type,
            backend=backend,
            device=device,
            cache_dir=cache_dir,
            min_area_frac=min_area_frac,
            multimask=multimask,
        )
    )
