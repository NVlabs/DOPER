# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference for the DOPER unified model.

Runs the full-image detection/mask path, then re-crops each detected box at high
resolution and runs the shared-backbone keypoint head. Optionally solves PnP
against the object's 3D keypoints to recover 6-DoF pose, using the same
``keypoints_3d.json`` convention as training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .cad_curves import (
    CADCurveSchema,
    PoseCurveRefinement,
    checkpoint_curve_metadata,
    overlay_named_curve_heatmaps,
    refine_pose_with_named_curves,
)
from .checkpoint_io import checkpoint_configs, load_checkpoint
from .config import (
    DEFAULT_DOPER_BACKBONE,
    DataConfig,
    DoperConfig,
    ModelConfig,
)
from .console import log
from .dataset import normalize_image
from .keypoints import keypoints_3d_fingerprint, load_keypoints_3d
from .model import DoperNet
from .performance import resolve_device
from .pnp import usac_pnp_ransac
from .targets import (
    bbox_to_center_size,
    decode_centernet,
    decode_keypoint_heatmaps,
    decode_simcc,
    heatmap_nms,
    letterbox_image,
    letterbox_params,
    topdown_warp_matrix,
)


@dataclass
class Detection:
    """One detected object with its keypoints and optional pose and raw outputs.

    Attributes:
        bbox_xyxy: ``(4,)`` box in original image pixels.
        score: Detection confidence.
        keypoints: ``(K, 2)`` keypoints in original image pixels.
        keypoint_scores: ``(K,)`` keypoint confidences.
        rvec: PnP rotation vector, when a pose was solved.
        tvec: PnP translation vector, when a pose was solved.
        reproj_error_px: Mean PnP reprojection error in pixels.
        simcc_x: Raw SimCC x logits, when requested.
        simcc_y: Raw SimCC y logits, when requested.
        keypoint_heatmap: Raw keypoint heatmaps, when requested.
        curve_heatmap: Named CAD-curve probability maps in crop space.
        curve_names: Ordered curve names matching ``curve_heatmap`` channels.
        curve_schema_sha256: Geometry hash of the curve schema the model was trained on.
        crop_to_image: ``(2, 3)`` affine from crop to image pixels.
    """

    bbox_xyxy: np.ndarray  # (4,) in original image pixels
    score: float
    keypoints: np.ndarray  # (K, 2) in original image pixels
    keypoint_scores: np.ndarray  # (K,)
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reproj_error_px: float | None = None
    simcc_x: np.ndarray | None = None
    simcc_y: np.ndarray | None = None
    keypoint_heatmap: np.ndarray | None = None
    curve_heatmap: np.ndarray | None = None
    curve_names: tuple[str, ...] = ()
    curve_schema_sha256: str = ""
    crop_to_image: np.ndarray | None = None


MIN_PNP_POINTS = 4
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MAX_DETECTIONS = 5
LETTERBOX_PAD_VALUE = 114


def load_model(checkpoint: str | Path, cfg: DoperConfig, device: str) -> DoperNet:
    """Build a model from a checkpoint and put it in eval mode.

    This mutates the caller's ``cfg`` in place: architecture and preprocessing
    fields of ``cfg.model`` are overwritten from the checkpoint, and
    ``cfg.model.pretrained`` is set to ``False`` because the checkpoint
    supplies every weight, so no download is attempted. The checkpoint is
    read with ``weights_only=True``.

    Args:
        checkpoint: Path to a ``.pth`` file written by training.
        cfg: Config whose ``model`` section is filled from the checkpoint;
            modified in place.
        device: Torch device string.

    Returns:
        The model on ``device`` with ``keypoints_3d_fingerprint`` set to the
        layout hash recorded at training time (``None`` for older checkpoints).

    Raises:
        ValueError: If the checkpoint is not a mapping or lacks weights.
        RuntimeError: If the weights do not match the resolved architecture.
    """
    ckpt, checkpoint_cfg, checkpoint_model_cfg = checkpoint_configs(
        load_checkpoint(checkpoint, device), f"checkpoint {checkpoint!r}"
    )
    recorded_fingerprint = checkpoint_cfg.get("keypoints_3d_fingerprint")

    # Architecture and preprocessing fields must come from the checkpoint.
    # Existing checkpoints lack ``keypoint_head`` and therefore remain SimCC.
    checkpoint_fields = (
        "backbone",
        "num_classes",
        "num_keypoints",
        "det_input_size",
        "det_stride",
        "letterbox",
        "neck_channels",
        "head_channels",
        "crop_size",
        "crop_pad_factor",
        "keypoint_head",
        "keypoint_heatmap_sigma",
        "simcc_split_ratio",
        "simcc_sigma",
        "gau_hidden_dims",
        "gau_s",
        "gau_expansion",
        "rotation_aux_enabled",
        "rotation_num_bins",
        "rotation_pool_size",
        "rotation_bin_seed",
        "rotation_logit_scale",
        "curve_head_enabled",
        "num_curves",
        "curve_head_channels",
        "curve_target_sigma",
        "curve_schema_version",
        "curve_names",
        "curve_geometry_sha256",
        "curve_object_id",
    )
    for field_name in checkpoint_fields:
        if field_name in checkpoint_model_cfg:
            setattr(cfg.model, field_name, checkpoint_model_cfg[field_name])
    if "keypoint_head" not in checkpoint_model_cfg:
        cfg.model.keypoint_head = "simcc"

    cfg.model.rotation_aux_enabled = bool(checkpoint_model_cfg.get("rotation_aux_enabled", False))

    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint {checkpoint!r} has no model weights")
    state_keys = tuple(state)
    curve_metadata = checkpoint_curve_metadata(checkpoint_model_cfg, state_keys)
    if not curve_metadata["curve_head_enabled"]:
        cfg.model.curve_head_enabled = False
        cfg.model.num_curves = 0
        cfg.model.curve_names = ()
        cfg.model.curve_schema_version = 0
        cfg.model.curve_geometry_sha256 = ""
        cfg.model.curve_object_id = ""
    else:
        for field_name, value in curve_metadata.items():
            setattr(cfg.model, field_name, value)
    state_head = None
    if any(key.startswith("kpt_head.cls_") for key in state_keys):
        state_head = "simcc"
    elif any(
        key.startswith(("kpt_head.neck.", "kpt_head.tower.", "kpt_head.out.")) for key in state_keys
    ):
        state_head = "heatmap"
    if state_head is not None and state_head != cfg.model.keypoint_head:
        raise RuntimeError(
            f"checkpoint {checkpoint!r} contains a {state_head} keypoint head, "
            f"but cfg.model.keypoint_head is {cfg.model.keypoint_head!r}"
        )

    cfg.model.pretrained = False
    model = DoperNet(cfg.model).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint {checkpoint!r} is incompatible with the resolved "
            f"{cfg.model.backbone}/{cfg.model.keypoint_head} architecture: {exc}"
        ) from exc
    model.eval()
    model.keypoints_3d_fingerprint = recorded_fingerprint
    return model


@torch.no_grad()
def infer_keypoints_in_box(
    model: DoperNet,
    image_bgr: np.ndarray,
    cfg: DoperConfig,
    device: str,
    box_xyxy: np.ndarray,
    keypoints_3d: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    box_pad: float = 0.0,
    kp_dark: bool = True,
) -> Detection:
    """Run the keypoint head in a supplied image-space box, bypassing detection."""
    mcfg = cfg.model
    bx = np.asarray(box_xyxy, dtype=np.float32)
    if bx.shape != (4,):
        raise ValueError(f"box_xyxy must have shape (4,), got {bx.shape}")
    x1, y1, x2, y2 = bx
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width <= 1.0 or height <= 1.0:
        raise ValueError(f"degenerate box {bx.tolist()}")

    crop_x = float(x1)
    crop_y = float(y1)
    crop_width = width
    crop_height = height
    if box_pad > 0.0:
        crop_x -= 0.5 * box_pad * width
        crop_y -= 0.5 * box_pad * height
        crop_width *= 1.0 + box_pad
        crop_height *= 1.0 + box_pad

    center, src_size = bbox_to_center_size(
        (crop_x, crop_y, crop_width, crop_height),
        mcfg.crop_pad_factor,
    )
    warp = topdown_warp_matrix(center, src_size, mcfg.crop_size)
    crop_bgr = cv2.warpAffine(
        image_bgr,
        warp,
        (mcfg.crop_size, mcfg.crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(LETTERBOX_PAD_VALUE,) * 3,
    )
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_batch = torch.from_numpy(np.stack([normalize_image(crop_rgb)])).to(device)
    crop_out = model.forward_crop(crop_batch)

    if mcfg.keypoint_head == "simcc":
        simcc_x = crop_out["simcc_x"]
        simcc_y = crop_out["simcc_y"]
        coords_crop, kp_scores = decode_simcc(
            simcc_x.cpu().numpy()[0],
            simcc_y.cpu().numpy()[0],
            mcfg.simcc_split_ratio,
            beta=cfg.loss.simcc_beta,
            dark=kp_dark,
        )
    else:
        coords_crop, kp_scores = decode_keypoint_heatmaps(
            crop_out["keypoint_heatmap"].cpu().numpy()[0],
            mcfg.crop_size,
            beta=cfg.loss.keypoint_heatmap_beta,
            dark=kp_dark,
        )

    inv = cv2.invertAffineTransform(warp)
    ones = np.ones((coords_crop.shape[0], 1), dtype=np.float32)
    kp_img = (np.concatenate([coords_crop, ones], axis=1) @ inv.T).astype(np.float32)
    detection = Detection(
        bbox_xyxy=bx,
        score=1.0,
        keypoints=kp_img,
        keypoint_scores=kp_scores,
        curve_heatmap=(
            torch.sigmoid(crop_out["curve_heatmap"]).cpu().numpy()[0]
            if "curve_heatmap" in crop_out
            else None
        ),
        curve_names=tuple(mcfg.curve_names) if mcfg.curve_head_enabled else (),
        curve_schema_sha256=(mcfg.curve_geometry_sha256 if mcfg.curve_head_enabled else ""),
        crop_to_image=inv if mcfg.curve_head_enabled else None,
    )
    if keypoints_3d is not None and camera_matrix is not None:
        solve_pnp(detection, keypoints_3d, camera_matrix, dist_coeffs)
    return detection


def assert_keypoints_match(model: DoperNet, keypoints_3d: np.ndarray | None) -> None:
    """Fail loudly if inference uses a different keypoint layout than training."""
    recorded = getattr(model, "keypoints_3d_fingerprint", None)
    if recorded is None or keypoints_3d is None:
        return  # pre-fingerprint checkpoint, nothing to compare
    actual = keypoints_3d_fingerprint(keypoints_3d)
    if actual != recorded:
        raise ValueError(
            "keypoints_3d mismatch: this checkpoint was trained against layout "
            f"{recorded}, inference was given {actual}. The 2D predictions would "
            "look fine and PnP would return a silently wrong pose."
        )


@torch.no_grad()
def infer_image(
    model: DoperNet,
    image_bgr: np.ndarray,
    cfg: DoperConfig,
    device: str,
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    keypoints_3d: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    box_pad: float = 0.0,
    return_mask: bool = False,
    kp_dark: bool = True,
    kp_decode_method: str = "argmax",
    kp_coord_beta: float = 1.0,
    return_raw_keypoint_outputs: bool = False,
) -> Any:  # noqa: ANN401 - list, or (list, mask) when return_mask
    """Detect + keypoints (+ optional PnP) on one BGR image.

    Returns a ``list[Detection]``. If ``return_mask`` is True, instead returns
    ``(detections, mask)`` where ``mask`` is an ``(H, W)`` float32 foreground
    probability map in *original image* pixels (the detection mask head output,
    un-letterboxed), or ``None`` if unavailable.

    Set ``return_raw_keypoint_outputs`` to retain the head's raw SimCC arrays or
    2D heatmaps, plus the crop-to-image transform, on each detection.
    """
    assert_keypoints_match(model, keypoints_3d)
    mcfg = cfg.model
    orig_h, orig_w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    det_size = mcfg.det_input_size
    lb_new_w = lb_new_h = det_size
    if mcfg.letterbox:
        lb_scale, lb_pad_x, lb_pad_y, lb_new_w, lb_new_h = letterbox_params(
            orig_h, orig_w, det_size
        )
        det_rgb = letterbox_image(
            rgb, det_size, pad_value=LETTERBOX_PAD_VALUE, interp=cv2.INTER_LINEAR
        )
    else:
        det_rgb = cv2.resize(rgb, (det_size, det_size))
    det_tensor = torch.from_numpy(normalize_image(det_rgb)).unsqueeze(0).to(device)

    out = model.forward_det(det_tensor)

    mask_full = None
    if return_mask:
        mp = torch.sigmoid(out["mask"])[0, 0].cpu().numpy().astype(np.float32)
        mp = cv2.resize(mp, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
        if mcfg.letterbox:
            mp = mp[lb_pad_y : lb_pad_y + lb_new_h, lb_pad_x : lb_pad_x + lb_new_w]
        mask_full = cv2.resize(mp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    hm = heatmap_nms(torch.sigmoid(out["heatmap"]))
    boxes, scores, _ = decode_centernet(
        hm[0].cpu().numpy(),
        out["wh"][0].cpu().numpy(),
        out["offset"][0].cpu().numpy(),
        mcfg.det_stride,
        topk=max_det,
        score_thr=score_thr,
    )

    # Detection ran at det_input_size; map boxes back to original pixels.
    if not mcfg.letterbox:
        sx = orig_w / det_size
        sy = orig_h / det_size

    detections: list[Detection] = []
    crops = []
    warps = []
    boxes_orig = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        if mcfg.letterbox:
            bx = np.array(
                [
                    (x1 - lb_pad_x) / lb_scale,
                    (y1 - lb_pad_y) / lb_scale,
                    (x2 - lb_pad_x) / lb_scale,
                    (y2 - lb_pad_y) / lb_scale,
                ],
                dtype=np.float32,
            )
        else:
            bx = np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)
        w = bx[2] - bx[0]
        h = bx[3] - bx[1]
        if w <= 1 or h <= 1:
            continue
        # Optional extra margin on the detected box before cropping for the
        # keypoint head. The detector can produce slightly tight boxes that clip
        # the object; padding gives the keypoint crop some breathing room. This
        # is applied on top of mcfg.crop_pad_factor (so total pad compounds) and
        # only affects the crop - the reported bbox stays the raw detection.
        cx0, cy0, cw, ch = bx[0], bx[1], w, h
        if box_pad > 0.0:
            cx0 = bx[0] - 0.5 * box_pad * w
            cy0 = bx[1] - 0.5 * box_pad * h
            cw = w * (1.0 + box_pad)
            ch = h * (1.0 + box_pad)
        center, src_size = bbox_to_center_size((cx0, cy0, cw, ch), mcfg.crop_pad_factor)
        warp = topdown_warp_matrix(center, src_size, mcfg.crop_size)
        crop_bgr = cv2.warpAffine(
            image_bgr,
            warp,
            (mcfg.crop_size, mcfg.crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(LETTERBOX_PAD_VALUE,) * 3,
        )
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append(normalize_image(crop_rgb))
        warps.append(warp)
        boxes_orig.append((bx, float(scores[i])))

    if not crops:
        return (detections, mask_full) if return_mask else detections

    crop_batch = torch.from_numpy(np.stack(crops)).to(device)
    crop_out = model.forward_crop(crop_batch)
    if mcfg.keypoint_head == "simcc":
        simcc_x = crop_out["simcc_x"]
        simcc_y = crop_out["simcc_y"]
        simcc_x = simcc_x.cpu().numpy()
        simcc_y = simcc_y.cpu().numpy()
        keypoint_heatmap = None
    else:
        keypoint_heatmap = crop_out["keypoint_heatmap"].cpu().numpy()
        simcc_x = simcc_y = None
    curve_heatmap = (
        torch.sigmoid(crop_out["curve_heatmap"]).cpu().numpy()
        if "curve_heatmap" in crop_out
        else None
    )

    for i, (bx, score) in enumerate(boxes_orig):
        if simcc_x is not None and simcc_y is not None:
            coords_crop, kp_scores = decode_simcc(
                simcc_x[i],
                simcc_y[i],
                mcfg.simcc_split_ratio,
                beta=cfg.loss.simcc_beta,
                dark=kp_dark,
                method=kp_decode_method,
                coord_beta=kp_coord_beta,
            )
        else:
            assert keypoint_heatmap is not None
            coords_crop, kp_scores = decode_keypoint_heatmaps(
                keypoint_heatmap[i],
                mcfg.crop_size,
                beta=cfg.loss.keypoint_heatmap_beta,
                dark=kp_dark,
            )
        inv = cv2.invertAffineTransform(warps[i])
        ones = np.ones((coords_crop.shape[0], 1), dtype=np.float32)
        homog = np.concatenate([coords_crop, ones], axis=1)
        kp_img = (homog @ inv.T).astype(np.float32)

        det = Detection(
            bbox_xyxy=bx,
            score=score,
            keypoints=kp_img,
            keypoint_scores=kp_scores,
            simcc_x=(simcc_x[i] if return_raw_keypoint_outputs and simcc_x is not None else None),
            simcc_y=(simcc_y[i] if return_raw_keypoint_outputs and simcc_y is not None else None),
            keypoint_heatmap=(
                keypoint_heatmap[i]
                if return_raw_keypoint_outputs and keypoint_heatmap is not None
                else None
            ),
            curve_heatmap=(curve_heatmap[i] if curve_heatmap is not None else None),
            curve_names=(tuple(mcfg.curve_names) if curve_heatmap is not None else ()),
            curve_schema_sha256=(mcfg.curve_geometry_sha256 if curve_heatmap is not None else ""),
            crop_to_image=(
                inv if return_raw_keypoint_outputs or curve_heatmap is not None else None
            ),
        )
        if keypoints_3d is not None and camera_matrix is not None:
            solve_pnp(det, keypoints_3d, camera_matrix, dist_coeffs)
        detections.append(det)
    return (detections, mask_full) if return_mask else detections


def solve_pnp(
    det: Detection,
    keypoints_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
) -> None:
    """Solve a PnP pose for ``det`` in place with sequential USAC RANSAC.

    Leaves ``det`` untouched when fewer than ``MIN_PNP_POINTS`` keypoints pass
    ``score_thr`` or the solver fails.

    Args:
        det: Detection with ``keypoints`` and ``keypoint_scores``; ``rvec``,
            ``tvec`` and ``reproj_error_px`` are filled on success.
        keypoints_3d: ``(K, 3)`` object-frame keypoints, ordered like ``det``.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        dist_coeffs: Distortion coefficients, or None for zeros.
        score_thr: Minimum keypoint score used as a PnP correspondence.

    Raises:
        RuntimeError: If OpenCV lacks ``cv2.UsacParams``.
    """
    mask = det.keypoint_scores >= score_thr
    if mask.sum() < MIN_PNP_POINTS:
        return
    obj = keypoints_3d[mask].astype(np.float64)
    img = det.keypoints[mask].astype(np.float64)
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
    ok, rvec, tvec, _inliers = usac_pnp_ransac(obj, img, camera_matrix, dist)
    if not ok:
        return
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix.astype(np.float64), dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    det.rvec = rvec
    det.tvec = tvec
    det.reproj_error_px = err


def refine_detection_pose_with_curves(
    detection: Detection,
    schema: CADCurveSchema,
    camera_matrix: np.ndarray,
    crop_size: int,
    steps: int = 75,
    rotation_lr: float = 1e-3,
    translation_lr: float = 1e-4,
    device: str = "cpu",
    keypoints_3d: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    keypoint_score_threshold: float = 0.3,
) -> PoseCurveRefinement:
    """Refine a PnP pose against the detection's named CAD-curve maps."""
    if detection.rvec is None or detection.tvec is None:
        raise ValueError("detection needs an initial PnP pose")
    if detection.curve_heatmap is None or detection.crop_to_image is None:
        raise ValueError("detection does not carry curve maps/crop transform")
    if detection.curve_schema_sha256 and detection.curve_schema_sha256 != schema.geometry_sha256:
        raise ValueError("detection curve schema hash does not match supplied CAD curves")
    if tuple(detection.curve_names) != schema.names:
        raise ValueError("detection ordered curve names do not match supplied CAD curves")
    image_to_crop = cv2.invertAffineTransform(detection.crop_to_image)
    evaluation_points_3d = None
    evaluation_points_2d = None
    if keypoints_3d is not None:
        points_3d = np.asarray(keypoints_3d)
        if points_3d.shape != (detection.keypoints.shape[0], 3):
            raise ValueError("keypoints_3d must match the detection keypoint count")
        selected = detection.keypoint_scores >= keypoint_score_threshold
        if selected.any():
            evaluation_points_3d = points_3d[selected]
            evaluation_points_2d = detection.keypoints[selected]
    return refine_pose_with_named_curves(
        detection.curve_heatmap,
        schema,
        detection.rvec,
        detection.tvec,
        camera_matrix,
        image_to_crop,
        crop_size,
        steps=steps,
        rotation_lr=rotation_lr,
        translation_lr=translation_lr,
        device=device,
        evaluation_points_3d=evaluation_points_3d,
        evaluation_points_2d=evaluation_points_2d,
        dist_coeffs=dist_coeffs,
    )


_ROT_FWD = {
    1: cv2.ROTATE_90_CLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
# Inverse: given a map produced in the k-rotated frame, rotate it back to original.
_ROT_INV = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}


def _unrotate_points(pts: np.ndarray, k: int, height: int, width: int) -> np.ndarray:
    """Map points from a k*90deg-CW-rotated image back to the original frame.

    ``height, width`` are the *original* image dimensions.
    """
    k = k % 4
    x = pts[:, 0]
    y = pts[:, 1]
    if k == 0:
        ox, oy = x, y
    elif k == 1:  # image was rotated 90 CW
        ox, oy = y, (height - 1) - x
    elif k == 2:  # 180
        ox, oy = (width - 1) - x, (height - 1) - y
    else:  # k == 3, image was rotated 90 CCW (270 CW)
        ox, oy = (width - 1) - y, x
    return np.stack([ox, oy], axis=1).astype(np.float32)


def _unrotate_box(box: np.ndarray, k: int, height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    c = _unrotate_points(corners, k, height, width)
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], dtype=np.float32)


@torch.no_grad()
def infer_image_tta(
    model: DoperNet,
    image_bgr: np.ndarray,
    cfg: DoperConfig,
    device: str,
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    keypoints_3d: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    box_pad: float = 0.0,
    return_mask: bool = False,
    kp_dark: bool = True,
    rots: tuple[int, ...] = (0, 1, 2, 3),
    depth_range: tuple[float, float] = (0.03, 1.0),
) -> Any:  # noqa: ANN401 - list, or (list, mask) when return_mask
    """90-degree rotation test-time augmentation / ensembling.

    Runs the full pipeline on the image rotated by each ``k*90`` degrees in
    ``rots``, maps every prediction back to the original frame, then produces a
    single best detection:

    - If 3D keypoints + intrinsics are given, PnP is solved per rotation (in the
      original frame) and the candidate with the lowest reprojection error whose
      depth falls in ``depth_range`` is selected. This is what breaks discrete
      pose flips: a flipped view is usually *not* self-consistent under PnP.
    - Otherwise (no PnP), keypoints are aggregated across rotations by the
      per-keypoint coordinate-wise median.

    Returns the same shape as :func:`infer_image` (a ``list[Detection]``, or
    ``(detections, mask)`` when ``return_mask`` is True). The returned list has
    at most one detection (the ensembled object).
    """
    height, width = image_bgr.shape[:2]
    candidates: list[Detection] = []
    mask_accum = None
    n_masks = 0

    for k in rots:
        k = k % 4
        img_r = image_bgr if k == 0 else cv2.rotate(image_bgr, _ROT_FWD[k])
        res = infer_image(
            model,
            img_r,
            cfg,
            device,
            score_thr=score_thr,
            max_det=max_det,
            keypoints_3d=None,
            camera_matrix=None,
            dist_coeffs=None,
            box_pad=box_pad,
            return_mask=return_mask,
            kp_dark=kp_dark,
        )
        dets_r, mask_r = res if return_mask else (res, None)
        if mask_r is not None:
            mask_o = mask_r if k == 0 else cv2.rotate(mask_r, _ROT_INV[k])
            mask_accum = mask_o if mask_accum is None else mask_accum + mask_o
            n_masks += 1
        if not dets_r:
            continue
        det_r = max(dets_r, key=lambda d: d.score)
        kp_o = _unrotate_points(det_r.keypoints, k, height, width)
        box_o = _unrotate_box(det_r.bbox_xyxy, k, height, width)
        cand = Detection(
            bbox_xyxy=box_o,
            score=det_r.score,
            keypoints=kp_o,
            keypoint_scores=det_r.keypoint_scores,
        )
        if keypoints_3d is not None and camera_matrix is not None:
            solve_pnp(cand, keypoints_3d, camera_matrix, dist_coeffs)
        candidates.append(cand)

    mask_full = (mask_accum / n_masks) if (return_mask and mask_accum is not None) else None

    if not candidates:
        empty: list[Detection] = []
        return (empty, mask_full) if return_mask else empty

    def _plausible(d: Detection) -> bool:
        if d.reproj_error_px is None or d.tvec is None:
            return False
        z = float(d.tvec.reshape(3)[2])
        return depth_range[0] <= z <= depth_range[1]

    valid = [d for d in candidates if _plausible(d)]
    if valid:
        best = min(valid, key=lambda d: d.reproj_error_px or 0.0)
    else:
        # No self-consistent PnP: fall back to per-keypoint median aggregation.
        kps = np.stack([c.keypoints for c in candidates], axis=0)
        scs = np.stack([c.keypoint_scores for c in candidates], axis=0)
        best = Detection(
            bbox_xyxy=max(candidates, key=lambda d: d.score).bbox_xyxy,
            score=float(np.median([c.score for c in candidates])),
            keypoints=np.median(kps, axis=0).astype(np.float32),
            keypoint_scores=np.median(scs, axis=0).astype(np.float32),
        )
        if keypoints_3d is not None and camera_matrix is not None:
            solve_pnp(best, keypoints_3d, camera_matrix, dist_coeffs)

    dets = [best]
    return (dets, mask_full) if return_mask else dets


def main() -> None:
    """Run inference on one image from the command line (``python -m doper.infer``)."""
    parser = argparse.ArgumentParser(description="Run DOPER inference on an image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="optional overlay PNG path")
    parser.add_argument("--backbone", default=DEFAULT_DOPER_BACKBONE)
    parser.add_argument("--num-keypoints", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--keypoints-3d", default=None)
    parser.add_argument("--intrinsics", default=None, help="camera_intrinsics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--show-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlay named CAD-curve probabilities when the checkpoint has them",
    )
    parser.add_argument(
        "--kp-dark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DARK sub-bin keypoint decoding (default on). --no-kp-dark for legacy argmax.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    cfg = DoperConfig(
        data=DataConfig(root="."),
        model=ModelConfig(
            backbone=args.backbone,
            pretrained=False,
            num_classes=args.num_classes,
            num_keypoints=args.num_keypoints,
            det_input_size=args.det_size,
            crop_size=args.crop_size,
        ),
    )
    model = load_model(args.checkpoint, cfg, device)

    keypoints_3d = None
    if args.keypoints_3d:
        keypoints_3d = load_keypoints_3d(args.keypoints_3d)

    camera_matrix = None
    dist = None
    if args.intrinsics:
        with open(args.intrinsics) as f:
            intrinsics = json.load(f)
        camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
        dist = np.asarray(intrinsics.get("distortion_coefficients", np.zeros(5)), dtype=np.float64)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image {args.image!r}")
    dets = infer_image(
        model,
        image,
        cfg,
        device,
        score_thr=args.score_thr,
        keypoints_3d=keypoints_3d,
        camera_matrix=camera_matrix,
        dist_coeffs=dist,
        kp_dark=args.kp_dark,
    )
    log(f"{len(dets)} detection(s)")
    for d in dets:
        line = f"  score={d.score:.3f} bbox={d.bbox_xyxy.round(1).tolist()}"
        if d.reproj_error_px is not None:
            line += f" reproj_err_px={d.reproj_error_px:.2f} depth_m={float(d.tvec[2]):.4f}"
        print(line)

    if args.out:
        vis = image.copy()
        for d in dets:
            if args.show_curves and d.curve_heatmap is not None and d.crop_to_image is not None:
                vis = overlay_named_curve_heatmaps(
                    vis,
                    d.curve_heatmap,
                    d.crop_to_image,
                    cfg.model.crop_size,
                )
            x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 255), 2)
            for (kx, ky), s in zip(d.keypoints, d.keypoint_scores, strict=False):
                if s >= args.score_thr:
                    cv2.circle(vis, (int(kx), int(ky)), 2, (0, 255, 0), -1)
        cv2.imwrite(args.out, vis)
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
