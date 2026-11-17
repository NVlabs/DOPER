# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a source golden with float leaves to a relative tolerance.

The SHIRT goldens hold OpenCV and BLAS results, whose last digits depend on the CPU code path a
machine takes (they drift by about 1e-8 relative between runners). Booleans, integers and
strings still compare exactly.
"""

from typing import Any

import pytest

GOLDEN_REL = 1e-6


def approx_floats(value: Any, rel: float = GOLDEN_REL) -> Any:  # noqa: ANN401
    """``value`` with every float replaced by ``pytest.approx(float, rel=rel)``, recursively."""
    if isinstance(value, float):
        return pytest.approx(value, rel=rel)
    if isinstance(value, dict):
        return {key: approx_floats(item, rel) for key, item in value.items()}
    if isinstance(value, list):
        return [approx_floats(item, rel) for item in value]
    return value
