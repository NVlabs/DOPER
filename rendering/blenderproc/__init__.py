# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkout-only BlenderProc campaign for the generated BOP objects (HANDAL obj40, HOT3D obj33).

``scene`` runs inside ``blenderproc run`` and is the only module that imports ``blenderproc`` or
``bpy``. The host modules (``plan``, ``campaign``, ``audit``, ``assets``, ``install``) need
neither, and ``doper`` never imports this package.
"""
