# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT: the SPEED+ model on the SHIRT ROE1 and ROE2 trajectories (Table 2).

``doper.shirt.data`` reads a SHIRT tree (labels, camera, metadata), ``doper.shirt.inputs``
holds what the stages share (records, checkpoint and layout checks, the model),
``doper.shirt.candidates`` runs the framewise evaluation and generates the 0/180 re-crop pose
candidates, and ``doper.shirt.policies`` filters them (PnP UKF, UKF + RTS, keypoint UKF).
SHIRT data and any record that carries its ground truth are user-local: nothing derived from
them is shipped.

This module is plain Python, so recipe parsing can import its names without numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

# The released trajectories and image domains; Table 2 reports the lightbox domain.
TRAJECTORIES: tuple[str, ...] = ("roe1", "roe2")
DOMAINS: tuple[str, ...] = ("synthetic", "lightbox")
PAPER_DOMAIN = "lightbox"


def check_request(
    trajectories: Sequence[str],
    domain: str,
    known_trajectories: Sequence[str] = TRAJECTORIES,
    known_domains: Sequence[str] = DOMAINS,
) -> None:
    """Refuse an empty, unknown or repeated trajectory, or an unknown domain.

    Plain Python, so ``doper evaluate shirt`` checks a request at plan time without numpy.

    Raises:
        ValueError: Naming the trajectory or domain at fault.
    """
    if isinstance(trajectories, str) or not trajectories:
        raise ValueError(f"name at least one SHIRT trajectory, got {trajectories!r}")
    unknown = [roe for roe in trajectories if roe not in known_trajectories]
    if unknown:
        raise ValueError(
            f"unknown SHIRT trajectories {unknown}; known: {tuple(known_trajectories)}"
        )
    if len(set(trajectories)) != len(trajectories):
        raise ValueError(f"SHIRT trajectories repeat: {list(trajectories)}")
    if domain not in known_domains:
        raise ValueError(f"unknown SHIRT domain {domain!r}; known: {tuple(known_domains)}")
