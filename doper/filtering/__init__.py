# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keypoint measurement covariances and Kalman/RTS pose filters.

``measurements`` needs only NumPy. ``pose_basis``, ``shirt_pose`` and ``shirt_ephemeris``
need the ``filter`` extra (``pip install doper[filter]``). Nothing is imported here so that
``import doper.filtering`` never pulls in bayesfilter.
"""
