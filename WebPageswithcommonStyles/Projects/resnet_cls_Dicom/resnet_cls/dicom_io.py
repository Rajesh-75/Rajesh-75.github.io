"""DICOM -> 8-bit PIL image, plus input validation.  Imports no torch, so it can be
reused by serving code. Handles: rescale slope/intercept, windowing, MONOCHROME1
inversion, 12/16-bit data, compressed transfer syntaxes, RGB (YBR) colour data."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

try:                                    # pydicom >= 3
    from pydicom.pixels import apply_modality_lut, convert_color_space
except ImportError:                     # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut, convert_color_space

log = logging.getLogger("resnet_cls")

# Radiographs only by default; widen this set per project, deliberately.
ALLOWED_MODALITIES = {"CR", "DX"}
ALLOWED_VIEWS = {"PA", "AP"}            # frontal chest views; rejects laterals


class InvalidInputError(ValueError):
    """The file is readable but must not be sent to the model."""


def is_dicom(path) -> bool:
    """Detect DICOM by content (128-byte preamble + 'DICM'), not by file extension."""
    p = Path(path)
    if p.suffix.lower() in {".dcm", ".dicom"}:
        return True
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def hash_patient_id(patient_id: str, salt: str) -> str:
    """Stable pseudonym used only for patient-level splitting. The raw ID is never stored."""
    return hashlib.sha256((salt + str(patient_id)).encode()).hexdigest()[:16]


def validate(ds, allowed_modalities=ALLOWED_MODALITIES, allowed_views=ALLOWED_VIEWS) -> None:
    """Gate the input: a model should only ever see the kind of image it was trained on."""
    modality = str(ds.get("Modality", ""))
    if allowed_modalities and modality not in allowed_modalities:
        raise InvalidInputError(f"Modality '{modality}' not allowed (expected {sorted(allowed_modalities)})")
    view = str(ds.get("ViewPosition", "")).strip()
    if allowed_views and modality in {"CR", "DX"} and view and view not in allowed_views:
        raise InvalidInputError(f"ViewPosition '{view}' not allowed (expected {sorted(allowed_views)})")
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        raise InvalidInputError("Multi-frame DICOM: this is a 2D model; select one frame upstream")
    if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        log.warning("BurnedInAnnotation=YES: pixels may contain patient text (PHI / shortcut risk)")


def _first(value):
    """WindowCenter/Width can be a single number or a list; take the first."""
    return float(value[0]) if isinstance(value, (list, tuple, pydicom.multival.MultiValue)) else float(value)


def dicom_to_pil(path, window: tuple[float, float] | None = None, check: bool = True) -> Image.Image:
    ds = pydicom.dcmread(str(path))
    if check:
        validate(ds)
    arr = ds.pixel_array                               # decompresses if needed (see requirements)

    if int(ds.get("SamplesPerPixel", 1)) == 3:         # colour DICOM (fundus, dermoscopy, pathology)
        if ds.PhotometricInterpretation.startswith("YBR"):
            arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")
        if arr.dtype != np.uint8:
            arr = (arr.astype(np.float32) / max(float(arr.max()), 1.0) * 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    arr = apply_modality_lut(arr, ds).astype(np.float32)   # stored value -> real units (e.g. HU)
    if window is not None:                              # explicit override, e.g. CT lung (-600, 1500)
        lo, hi = window[0] - window[1] / 2, window[0] + window[1] / 2
    elif "WindowCenter" in ds and "WindowWidth" in ds:  # the acquisition's own display window
        c, w = _first(ds.WindowCenter), _first(ds.WindowWidth)
        lo, hi = c - w / 2, c + w / 2
    else:                                               # fallback: robust percentile stretch
        lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        raise InvalidInputError("Degenerate intensity range (blank or constant image)")
    arr = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0
    if ds.PhotometricInterpretation == "MONOCHROME1":   # 0 = white: invert so bone is always bright
        arr = 255.0 - arr
    return Image.fromarray(arr.astype(np.uint8), "L")


def load_image(path, window: tuple[float, float] | None = None) -> Image.Image:
    """Single entry point for training AND inference: DICOM or ordinary image file."""
    if is_dicom(path):
        return dicom_to_pil(path, window)
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I"):           # 16-bit PNG/TIFF: scale instead of clipping
            a = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(a, [0.5, 99.5])
            return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), "L")
        return im.convert("RGB") if im.mode not in ("L", "RGB") else im.copy()


def parse_window(text: str) -> tuple[float, float] | None:
    """'-600,1500' -> (-600.0, 1500.0); '' -> None."""
    if not text:
        return None
    c, w = (float(t) for t in text.split(","))
    return c, w
