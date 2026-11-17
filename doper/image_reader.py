# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in PNG decoding with the dataset's OpenCV BGR/unchanged contract.

Optimize plain RGB8 and grayscale PNGs. Delegate palettes, alpha, transparency,
EXIF orientation and other formats to OpenCV to retain its exact semantics.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# EXIF orientations whose display transform swaps width and height.
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_EXIF_ORIENTATION_TAG = 0x0112


def read_image(path: str | Path, flag: int, decoder: str = "opencv") -> np.ndarray | None:
    """Read an image like ``cv2.imread``, optionally decoding plain PNGs with pyspng.

    Args:
        path: Image file path.
        flag: OpenCV imread flag, e.g. ``cv2.IMREAD_COLOR``.
        decoder: ``"opencv"`` or ``"pyspng"``. The latter needs the optional
            ``pyspng`` dependency and only handles plain RGB8/grayscale PNGs itself.

    Returns:
        The decoded array in OpenCV BGR/unchanged layout, or None if unreadable.

    Raises:
        ValueError: If ``decoder`` is unknown.
    """
    if decoder not in ("opencv", "pyspng"):
        raise ValueError(f"Unknown PNG decoder: {decoder}")
    if decoder == "opencv":
        return cv2.imread(str(path), flag)
    # Fail clearly for an explicitly selected but unavailable optional dependency.
    import pyspng

    try:
        payload = Path(path).read_bytes()
    except OSError:
        return None

    def fallback() -> np.ndarray | None:
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return cv2.imread(str(path), flag)
    if len(payload) < 33 or payload[12:16] != b"IHDR":
        return fallback()
    depth, color = payload[24:26]
    if flag not in (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED):
        return fallback()
    if not ((color == 2 and depth == 8) or (color == 0 and depth in (8, 16))):
        return fallback()
    if flag == cv2.IMREAD_COLOR and depth != 8:
        return fallback()
    offset = 8
    while offset + 12 <= len(payload):
        size = int.from_bytes(payload[offset : offset + 4], "big")
        kind = payload[offset + 4 : offset + 8]
        if kind in (b"eXIf", b"tRNS"):
            # imread also applies EXIF orientation for IMREAD_COLOR.
            return cv2.imread(str(path), flag)
        offset += size + 12
    try:
        value = pyspng.load(payload)
    except (RuntimeError, ValueError):
        return fallback()
    if color == 0:
        # libspng can add opaque alpha to 16-bit grayscale.
        if value.ndim == 3:
            value = value[..., 0]
        if flag == cv2.IMREAD_COLOR:
            value = np.repeat(value[..., None], 3, axis=-1)
    else:
        value = value[..., :3][..., ::-1]
    return np.ascontiguousarray(value)


def read_image_size(path: str | Path) -> tuple[int, int] | None:
    """Return the ``(width, height)`` that ``cv2.imread`` would decode, from the header.

    EXIF orientation is applied like ``cv2.IMREAD_COLOR`` does. Formats Pillow
    cannot open fall back to a full OpenCV decode.

    Args:
        path: Image file path.

    Returns:
        The decoded size, or None if the file is unreadable.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
    except (OSError, ValueError):
        decoded = cv2.imread(str(path))
        return None if decoded is None else (decoded.shape[1], decoded.shape[0])
    if orientation in _TRANSPOSING_ORIENTATIONS:
        return height, width
    return width, height
