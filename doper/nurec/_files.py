# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File helpers shared by the NuRec stages: hard-link-or-copy and checked image reads."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Hard links fail across file systems, where the file system refuses them, or at the link cap.
COPY_ERRNOS = (errno.EXDEV, errno.EPERM, errno.EMLINK)


def link_or_copy(source: Path, dest: Path) -> None:
    """Hard-link ``source`` to ``dest`` (creating its parent); copy it where links fail."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError as error:
        if error.errno not in COPY_ERRNOS:
            raise
        shutil.copy2(source, dest)


def read_image(path: Path, flags: int) -> NDArray[np.uint8]:
    """``cv2.imread`` as uint8, raising ``FileNotFoundError`` for an unreadable file."""
    import cv2

    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    return np.asarray(image, dtype=np.uint8)
