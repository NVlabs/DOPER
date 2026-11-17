# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint detection + mask + keypoint dataset for the DOPER model.

Reads the same COCO-format render output used by the existing RTMDet/RTMPose
pipeline (``images/``, ``masks/``, ``annotations.json``) and produces fixed-shape
training samples carrying targets for all three heads:

* full detection image + CenterNet targets (heatmap / wh / offset),
* a binarized foreground mask target,
* a high-resolution top-down keypoint crop + SimCC targets.

Detection targets use all annotations on the image. Legacy ``primary`` sampling
uses one largest-area/explicitly marked instance for the crop and its mask.
Opt-in ``all`` sampling visits every instance as a separate sample; each sample
still has all image boxes. ``union`` mask targets supervise the union of instance
foreground, not the arbitrary crop instance. Thus all-instance epochs are
instance-weighted: multi-instance images occur more often, not larger batches.

Samples with no keypoint annotations (e.g. real-world bbox/mask-only data) set
``has_kp = 0`` so the keypoint loss is masked out while detection and mask
supervision still flow through the shared backbone.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import targets
from .augment import lens_flare_augment, photometric_augment, random_erase
from .background_data import next_worker_occurrence, worker_index_rng
from .cad_curves import (
    ProjectedCurves,
    load_projected_curves,
    rasterize_curve_heatmaps,
    transform_projected_curves,
)
from .config import DoperConfig
from .console import log
from .image_reader import read_image, read_image_size
from .keypoints import load_keypoints_3d

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PIXEL_MAX = 255.0
# Letterbox / out-of-frame fill value for RGB image warps.
_PAD_VALUE = 114
# Fraction of sampled keypoints off the mask above which the data check fails.
_MAX_OFF_MASK_FRACTION = 0.25
# Median reprojection residual (px) above which the pose check fails.
_MAX_POSE_RESIDUAL_PX = 5.0
# Independent RNG streams per sample (see ``worker_index_rng``).
_STREAM_COMPOSITE = 0
_STREAM_DET_AUG = 1
_STREAM_CROP_GEOMETRY = 2
_STREAM_CROP_PHOTO = 3
_STREAM_LENS_FLARE = 4


def normalize_image(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 RGB image to a normalized CHW float32 array.

    Args:
        img_rgb: HWC uint8 RGB image.

    Returns:
        Contiguous CHW float32 array, scaled to [0, 1] and ImageNet-normalized.
    """
    arr = img_rgb.astype(np.float32) / _PIXEL_MAX
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


class DoperDataset(Dataset):
    """COCO-format joint detection, mask and keypoint dataset (see module docstring).

    Args:
        cfg: Full DOPER config.
        split: ``"train"``, ``"val"`` or a split directory name.
        force_composite: Replace every foreground-bearing sample's background.
        composite_prob: Probability of background compositing when not forced.
        keypoint_only: Disable mask supervision (keypoint/detection streams only).

    Raises:
        ValueError: On invalid sampling/config options or failed dataset checks.
    """

    def __init__(
        self,
        cfg: DoperConfig,
        split: str,
        force_composite: bool = False,
        composite_prob: float = 0.0,
        keypoint_only: bool = False,
    ) -> None:
        """Load annotations for ``split`` and run the startup data checks."""
        self.cfg = cfg
        self.split = split
        self.keypoint_only = bool(keypoint_only)
        if cfg.data.instance_sampling not in {"primary", "all"}:
            raise ValueError("instance_sampling must be primary or all")
        if cfg.data.mask_target_mode not in {"primary", "union"}:
            raise ValueError("mask_target_mode must be primary or union")
        if cfg.data.instance_sampling == "all" and cfg.data.mask_target_mode != "union":
            raise ValueError(
                "all-instance sampling requires union masks to avoid contradictory "
                "foreground labels"
            )
        if cfg.model.curve_head_enabled and cfg.model.num_curves <= 0:
            raise ValueError("num_curves must be positive when curve_head_enabled is true")
        # When True (the composite stream) every foreground-bearing sample
        # has its rendered background replaced by a background crop, keeping the
        # existing bbox / mask / keypoint labels exactly valid.
        self.force_composite = force_composite
        self.composite_prob = 1.0 if force_composite else float(composite_prob)
        if not 0.0 <= self.composite_prob <= 1.0:
            raise ValueError(f"composite_prob must be in [0, 1], got {self.composite_prob}")
        self._bg_pool = None
        self._bg_fp_boxes = {}
        self._epoch = 0
        self._rng_counters: dict[tuple[int, int], int] = {}
        if self.composite_prob > 0:
            from .background_data import (
                BackgroundImagePool,
                load_fp_boxes,
                split_file_paths,
            )

            cfg_bg = cfg.background
            paths = split_file_paths(cfg_bg, cfg_bg.train_split)
            self._bg_pool = BackgroundImagePool(paths, cfg_bg.max_image_side)
            self._bg_fp_boxes = load_fp_boxes(cfg_bg.fp_mine_json)
        split_dir = {
            "train": cfg.data.train_split,
            "val": cfg.data.val_split,
        }.get(split, split)
        self.base = os.path.join(cfg.data.root, split_dir)
        self.img_dir = os.path.join(self.base, cfg.data.images_dirname)
        # Appearance augmentation is train-only.
        self.do_aug = split == "train" and cfg.aug.enabled
        ann_path = os.path.join(self.base, cfg.data.ann_filename)
        with open(ann_path) as f:
            coco = json.load(f)

        self.images = {im["id"]: im for im in coco["images"]}
        # Zero-based class id per COCO category id (sorted for determinism).
        cat_ids = sorted(c["id"] for c in coco["categories"])
        self.cat_to_label = {cid: i for i, cid in enumerate(cat_ids)}

        anns_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            # An empty box (e.g. a fully hidden robustness render, kept for
            # review) cannot supervise detection or the keypoint crop.
            bbox = ann.get("bbox")
            if bbox is not None and (bbox[2] <= 0 or bbox[3] <= 0):
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)
        # Only keep images that actually have annotations.
        self.ids = [i for i in self.images if i in anns_by_img]
        self.anns_by_img = anns_by_img
        self.image_count = len(self.ids)
        self._primary_by_sample = []
        expanded_ids = []
        for image_id in self.ids:
            annotations = anns_by_img[image_id]
            marked = [ann for ann in annotations if ann.get("is_primary", False)]
            if cfg.data.instance_sampling == "all" and len(marked) > 1:
                raise ValueError("all-instance data has multiple is_primary markers on one image")
            # Pre-expanded pseudo roots already mark one primary per image
            # record: honor that marker rather than expanding them a second time.
            selected = (
                (marked if len(marked) == 1 else annotations)
                if cfg.data.instance_sampling == "all"
                else [
                    marked[0]
                    if len(marked) == 1
                    else max(annotations, key=lambda a: a.get("area", 0.0))
                ]
            )
            expanded_ids.extend([image_id] * len(selected))
            self._primary_by_sample.extend(selected)
        self.ids = expanded_ids
        self._data_faults: list[str] = []
        self._report_image_shape(split)
        self._report_mask_supervision(split)
        self._report_pose_consistency(split)
        self._report_keypoint_mask_agreement(split)
        if self._data_faults and os.environ.get("DOPER_ALLOW_BAD_DATA") != "1":
            raise ValueError(
                f"dataset checks failed for split {split!r}:\n  - "
                + "\n  - ".join(self._data_faults)
                + "\n"
                "These print at startup and were previously easy to miss, so they "
                "now stop the run. Set DOPER_ALLOW_BAD_DATA=1 to train anyway."
            )

    def __len__(self) -> int:
        """Return the number of samples after instance expansion."""
        return len(self.ids)

    def primary_annotation(self, index: int) -> dict:
        """Return the exact annotation used for this sample's keypoint/curve crop."""
        return self._primary_by_sample[index]

    def set_epoch(self, epoch: int) -> None:
        """Set the RNG epoch for direct or non-persistent-worker use."""
        self._epoch = int(epoch)
        self._rng_counters.clear()

    def _report_image_shape(self, split: str) -> None:
        """Check the annotation's image size matches the file on disk.

        A renderer that writes one resolution into the records and another to
        the PNG silently teaches the detector the wrong geometry: boxes and
        keypoints are mapped into the network input against the wrong height
        while every self-consistent check still passes. Caught exactly this on
        4:3 HANDal renders, where the records said 640x640 and the files were
        640x480.
        """
        imgs = list(self.images.values()) if isinstance(self.images, dict) else list(self.images)
        if not imgs:
            return
        sample = imgs[:: max(1, len(imgs) // 20)][:20]
        bad = []
        for rec in sample:
            if not isinstance(rec, dict) or "width" not in rec or not rec.get("file_name"):
                continue
            name = rec["file_name"]
            for cand in (
                os.path.join(self.base, "images", os.path.basename(name)),
                os.path.join(self.base, name),
                os.path.join(self.base, "rendered", name),
            ):
                if os.path.exists(cand):
                    size = read_image_size(cand)
                    if size is not None and size != (rec["width"], rec["height"]):
                        bad.append((os.path.basename(name), rec["width"], rec["height"], *size))
                    break
        tag = f"image shape ({split}):"
        if bad:
            n, aw, ah, fw, fh = bad[0]
            msg = (
                f"image shape: {len(bad)}/{len(sample)} sampled images disagree with "
                f"their annotation, e.g. {n}: records say {aw}x{ah}, the file is "
                f"{fw}x{fh}. Detection targets would be built against wrong geometry."
            )
            log(f"{tag} MISMATCH - {msg}")
            self._data_faults.append(msg)
        else:
            log(f"{tag} ok - {len(sample)} sampled images match their records")

    def _report_keypoint_mask_agreement(self, split: str) -> None:
        """Do the stored 2D keypoints actually land on the object?

        Independent of _report_pose_consistency, which reprojects the stored
        pose through the stored intrinsics and so passes whenever those two are
        wrong in the same way. This compares labels against the rendered mask,
        catching a projection or convention error that is internally consistent
        but does not match the pixels.
        """
        anns = [a for anns in self.anns_by_img.values() for a in anns]
        sample = [a for a in anns if a.get("keypoints") and a.get("mask_file")]
        sample = sample[:: max(1, len(sample) // 100)][:100]
        if not sample:
            return
        offs, checked = [], 0
        for a in sample:
            path = next(
                (
                    c
                    for c in (
                        os.path.join(self.base, a["mask_file"]),
                        os.path.join(self.base, "rendered", a["mask_file"]),
                    )
                    if os.path.exists(c)
                ),
                None,
            )
            if path is None:
                continue
            m = cv2.imread(path, 0)
            if m is None or m.max() == 0:
                continue
            kp = np.asarray(a["keypoints"], dtype=np.float64).reshape(-1, 3)
            vis = kp[:, 2] == 2
            if vis.sum() == 0:
                continue
            xy = kp[vis, :2]
            finite = np.isfinite(xy).all(axis=1)
            pts = np.floor(np.where(np.isfinite(xy), xy, -1)).astype(int)
            inb = (
                finite
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < m.shape[1])
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < m.shape[0])
            )
            if a.get("bop"):
                # BOP conversion uses geometry.keypoint_visibility's one-pixel
                # depth neighborhood. Boundary points can therefore legitimately
                # miss the single floor-sampled mask pixel. Match that footprint;
                # keep the threshold, OOB accounting and other geometry gates.
                m = cv2.dilate(m, np.ones((3, 3), dtype=np.uint8))
            # A point marked visible that falls outside the image is off-mask
            # by definition. Scoring only the in-bounds ones lets a sample
            # whose keypoints mostly land off-image report a low off-mask rate
            # and pass.
            on = int((m[pts[inb, 1], pts[inb, 0]] > 127).sum()) if inb.any() else 0
            offs.append(1.0 - on / float(len(pts)))
            checked += 1
        if not checked:
            return
        off = float(np.mean(offs))
        tag = f"keypoint/mask agreement ({split}; BOP boundary tolerance=1px):"
        if off > _MAX_OFF_MASK_FRACTION:
            msg = (
                f"keypoint/mask: {100 * off:.0f}% of keypoints marked visible fall "
                f"outside the object mask over {checked} samples; the 2D labels "
                f"do not describe these pixels"
            )
            log(f"{tag} MISMATCH - {msg}")
            self._data_faults.append(msg)
        else:
            log(
                f"{tag} ok - {100 * off:.1f}% of visible keypoints fall off the mask "
                f"({checked} samples)"
            )

    def _report_pose_consistency(self, split: str) -> None:
        """Check that the stored pose reproduces the stored 2D keypoints.

        Any frame/unit/convention error between ``object_pose``,
        ``camera_extrinsics`` and ``camera_intrinsics`` shows up here as a large
        reprojection residual. Consumers of the pose (the rotation auxiliary,
        any pose-supervised head) otherwise train happily on a target that does
        not correspond to the pixels, and nothing else in the pipeline notices.
        """
        path = getattr(self.cfg.data, "keypoints_3d_json", None)
        if not path or not os.path.exists(path):
            return
        try:
            kp3d = load_keypoints_3d(path)
        except Exception:
            return
        anns = [a for anns in self.anns_by_img.values() for a in anns]
        usable = [
            a
            for a in anns
            if a.get("object_pose") is not None
            and a.get("camera_intrinsics") is not None
            and a.get("keypoints")
        ][:20]
        if not usable:
            return
        errs = []
        for a in usable:
            ci = a["camera_intrinsics"]
            if isinstance(ci, dict):  # {'fx':..,'fy':..,'cx':..,'cy':..}
                intrinsics = np.array(
                    [[ci["fx"], 0.0, ci["cx"]], [0.0, ci["fy"], ci["cy"]], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            else:
                intrinsics = np.asarray(ci, dtype=np.float64).reshape(3, 3)
            pose = np.asarray(a["object_pose"], dtype=np.float64)
            extr = a.get("camera_extrinsics")
            if extr is not None and pose.shape == (4, 4):
                flip = np.diag([1.0, -1.0, -1.0, 1.0])
                pose = flip @ np.linalg.inv(np.asarray(extr, dtype=np.float64)) @ pose
            cam = (pose[:3, :3] @ kp3d.T).T + pose[:3, 3]
            if (cam[:, 2] <= 1e-9).any():
                continue
            uv = (intrinsics @ cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            gt = np.asarray(a["keypoints"], dtype=np.float64).reshape(-1, 3)[:, :2]
            if gt.shape != uv.shape:
                return
            errs.append(float(np.median(np.linalg.norm(uv - gt, axis=1))))
        if not errs:
            return
        med = float(np.median(errs))
        tag = f"pose/keypoint consistency ({split}):"
        if med > _MAX_POSE_RESIDUAL_PX:
            self._data_faults.append(
                f"pose/keypoint: median reprojection residual {med:.1f} px over "
                f"{len(errs)} samples; object_pose does not reproduce the stored "
                f"keypoints, so a pose-supervised loss trains on the wrong target"
            )
            log(
                f"{tag} MISMATCH - median reprojection residual {med:.1f} px "
                f"over {len(errs)} samples; object_pose does not reproduce the "
                f"stored keypoints, so any pose-supervised loss is training on "
                f"the wrong target"
            )
        else:
            log(f"{tag} ok - median reprojection residual {med:.3f} px")

    def _report_mask_supervision(self, split: str) -> None:
        """Say plainly whether mask supervision will actually happen.

        Two ways it silently does not, both of which have bitten this pipeline:
        annotations that carry no ``mask_file`` at all (the BOP PBR builder
        records the filename under ``segmentation``, which nothing reads), and
        annotations whose ``mask_file`` does not resolve on disk (the split
        script joined the mask dir twice). Either way the mask loss sits at
        exactly 0.0000 for the whole run with no other warning.
        """
        anns = (
            [a for anns in self.anns_by_img.values() for a in anns]
            if hasattr(self, "anns_by_img")
            else []
        )
        if not anns:
            return
        sample = anns[:: max(1, len(anns) // 200)][:200]
        with_field = [a for a in sample if a.get("mask_file")]
        resolved = 0
        for a in with_field:
            mf = a["mask_file"]
            if os.path.exists(os.path.join(self.base, mf)) or os.path.exists(
                os.path.join(self.base, "rendered", mf)
            ):
                resolved += 1
        tag = f"mask supervision ({split}, {self.base}):"
        if not with_field:
            log(
                f"{tag} DISABLED - 0/{len(sample)} sampled annotations carry "
                f"a 'mask_file' key; the mask loss will stay at 0.0000"
            )
        elif resolved == 0:
            # Printing is not enough: __init__ raises only from _data_faults,
            # so an unresolved mask path otherwise just silently sets
            # mask_weight=0 and the run trains with no mask supervision.
            msg = (
                f"mask supervision: {len(with_field)}/{len(sample)} "
                f"annotations reference a mask but none resolve under "
                f"{self.base!r}; the mask loss would stay at 0.0000"
            )
            log(f"{tag} BROKEN - {msg}")
            self._data_faults.append(msg)
        elif resolved < len(with_field):
            msg = (
                f"mask supervision: only {resolved}/{len(with_field)} "
                f"referenced masks resolve under {self.base!r}; the rest "
                f"would train with mask_weight=0"
            )
            log(f"{tag} PARTIAL - {msg}")
            self._data_faults.append(msg)
        else:
            log(
                f"{tag} active - {resolved}/{len(sample)} sampled annotations "
                f"have a mask that resolves"
            )

    def _load_mask(self, primary_ann: dict, orig_h: int, orig_w: int) -> tuple[np.ndarray, float]:
        mask_file = primary_ann.get("mask_file")
        if not mask_file:
            return np.zeros((orig_h, orig_w), dtype=np.uint8), 0.0
        mask_path = os.path.join(self.base, mask_file)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.base, "rendered", mask_file)
        raw = read_image(mask_path, cv2.IMREAD_UNCHANGED, self.cfg.data.png_decoder)
        if raw is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8), 0.0
        if raw.ndim == 3:
            raw = raw[..., 0]
        fg = (raw >= self.cfg.data.mask_fg_threshold).astype(np.uint8)
        return fg, 1.0

    def _load_union_mask(
        self, annotations: list[dict], orig_h: int, orig_w: int
    ) -> tuple[np.ndarray, float]:
        """Union visible masks; missing supervision is unknown, not background.

        Empty but valid masks remain available supervision (fully occluded
        instances). If any instance has no mask annotation, disable mask loss
        for the whole image rather than teaching it as background. Declared
        mask files that are missing/unreadable are data errors.
        """
        union = np.zeros((orig_h, orig_w), dtype=np.uint8)
        complete = True
        for ann in annotations:
            mask, available = self._load_mask(ann, orig_h, orig_w)
            if not available:
                if ann.get("mask_file"):
                    raise ValueError(
                        "declared instance mask is missing or unreadable in union mode"
                    )
                complete = False
                continue
            if mask.shape != union.shape:
                raise ValueError("instance mask dimensions differ from source RGB")
            union |= mask
        return union, float(complete and bool(annotations))

    def _load_curves(self, primary_ann: dict) -> ProjectedCurves | None:
        curve_file = primary_ann.get("curve_file")
        if not curve_file:
            return None
        curve_path = os.path.join(self.base, curve_file)
        if not os.path.isfile(curve_path):
            raise FileNotFoundError(curve_path)
        annotation_hash = primary_ann.get("curve_schema_sha256")
        expected_hash = self.cfg.model.curve_geometry_sha256 or None
        if annotation_hash and expected_hash and annotation_hash != expected_hash:
            raise ValueError(
                f"annotation curve schema hash {annotation_hash} does not match "
                f"model {expected_hash}"
            )
        return load_projected_curves(
            curve_path,
            expected_num_curves=self.cfg.model.num_curves,
            expected_schema_sha256=expected_hash,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one training sample (detection, mask and keypoint/curve targets).

        Args:
            index: Sample index after instance expansion.

        Returns:
            Dict of fixed-shape tensors, including normalized ``det_image`` and
            ``crop_image`` and the per-head targets and weights.

        Raises:
            FileNotFoundError: If the image file cannot be read.
        """
        mcfg = self.cfg.model
        occurrence = next_worker_occurrence(self._rng_counters, index)
        img_id = self.ids[index]
        meta = self.images[img_id]
        anns = self.anns_by_img[img_id]

        img_path = os.path.join(self.img_dir, meta["file_name"])
        bgr = read_image(img_path, cv2.IMREAD_COLOR, self.cfg.data.png_decoder)
        if bgr is None:
            raise FileNotFoundError(img_path)
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --- detection image + mask (built together in det space) ----------
        # The det image, its boxes, and the foreground mask are all carried
        # through the same geometric transforms so they stay pixel-aligned; the
        # keypoint crop is built separately from the original image below.
        det_size = mcfg.det_input_size
        primary = self.primary_annotation(index)
        if self.keypoint_only:
            fg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            mask_weight = 0.0
        elif self.cfg.data.mask_target_mode == "union":
            fg_mask, mask_weight = self._load_union_mask(anns, orig_h, orig_w)
        else:
            fg_mask, mask_weight = self._load_mask(primary, orig_h, orig_w)
        composite_rng = worker_index_rng(
            self.cfg.background.seed,
            index,
            self._epoch,
            stream=_STREAM_COMPOSITE,
            occurrence=occurrence,
        )
        should_composite = (
            self._bg_pool is not None
            and mask_weight > 0
            and composite_rng.random() < self.composite_prob
        )
        if should_composite:
            from .background_data import apply_background_to_synthetic

            assert self._bg_pool is not None
            rgb, bgr = apply_background_to_synthetic(
                rgb,
                bgr,
                fg_mask,
                self._bg_pool,
                self._bg_fp_boxes,
                self.cfg.background,
                composite_rng,
            )
        if self.do_aug and self.cfg.aug.lens_flare:
            flare_rng = worker_index_rng(
                self.cfg.train.seed,
                index,
                self._epoch,
                stream=_STREAM_LENS_FLARE,
                occurrence=occurrence,
            )
            rgb = lens_flare_augment(rgb, self.cfg.aug, flare_rng)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if mcfg.letterbox:
            scale, pad_x, pad_y, _, _ = targets.letterbox_params(orig_h, orig_w, det_size)
            det_rgb = targets.letterbox_image(
                rgb, det_size, pad_value=_PAD_VALUE, interp=cv2.INTER_LINEAR
            )
            mask_det = targets.letterbox_image(
                fg_mask, det_size, pad_value=0, interp=cv2.INTER_NEAREST
            )
        else:
            scale_x = det_size / orig_w
            scale_y = det_size / orig_h
            det_rgb = cv2.resize(rgb, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
            mask_det = cv2.resize(fg_mask, (det_size, det_size), interpolation=cv2.INTER_NEAREST)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if mcfg.letterbox:
                boxes.append([x * scale + pad_x, y * scale + pad_y, w * scale, h * scale])
            else:
                boxes.append([x * scale_x, y * scale_y, w * scale_x, h * scale_y])
            labels.append(self.cat_to_label.get(ann["category_id"], 0))
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.asarray(labels, dtype=np.int64)

        if self.do_aug:
            acfg = self.cfg.aug
            dgrng = worker_index_rng(
                self.cfg.train.seed,
                index,
                self._epoch,
                stream=_STREAM_DET_AUG,
                occurrence=occurrence,
            )
            det_rgb = photometric_augment(det_rgb, acfg, dgrng)

            # Lossless 90-degree rotation (image + mask + boxes together).
            if dgrng.random() < acfg.det_rot90_prob:
                det_k = int(dgrng.integers(1, 4))
                det_rgb = np.ascontiguousarray(np.rot90(det_rgb, det_k))
                mask_det = np.ascontiguousarray(np.rot90(mask_det, det_k))
                boxes_arr = targets.rot90_boxes(boxes_arr, det_k, det_size)

            # Continuous zoom + translation + small in-plane rotation, combined
            # into a single affine. Chirality-preserving (proper rotation, no
            # flips), so the keypoint head's handedness is never corrupted.
            do_scale = dgrng.random() < acfg.det_scale_prob
            do_trans = dgrng.random() < acfg.det_translate_prob
            do_rot = dgrng.random() < acfg.det_rotate_prob
            if do_scale or do_trans or do_rot:
                s = 1.0 + dgrng.uniform(-acfg.det_scale, acfg.det_scale) if do_scale else 1.0
                deg = float(dgrng.uniform(-acfg.det_rotate, acfg.det_rotate)) if do_rot else 0.0
                tx = (
                    dgrng.uniform(-acfg.det_translate, acfg.det_translate) * det_size
                    if do_trans
                    else 0.0
                )
                ty = (
                    dgrng.uniform(-acfg.det_translate, acfg.det_translate) * det_size
                    if do_trans
                    else 0.0
                )
                # getRotationMatrix2D(center, 0, s) == scale_translate_matrix, so
                # this reduces to the old behavior when deg == 0.
                c = det_size / 2.0
                affine = cv2.getRotationMatrix2D((c, c), deg, s)
                affine[0, 2] += tx
                affine[1, 2] += ty
                affine = affine.astype(np.float32)
                det_rgb = cv2.warpAffine(
                    det_rgb,
                    affine,
                    (det_size, det_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
                )
                mask_det = cv2.warpAffine(
                    mask_det,
                    affine,
                    (det_size, det_size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                boxes_arr, keep = targets.affine_boxes_xywh(boxes_arr, affine, det_size)
                labels_arr = labels_arr[keep]

            # Random erasing / cutout (det pixels only; boxes + mask untouched).
            if dgrng.random() < acfg.det_erase_prob:
                det_rgb = random_erase(det_rgb, acfg, dgrng)

        feat_h, feat_w = self.cfg.det_feat_size
        heatmap, wh, offset, reg_weight = targets.encode_centernet(
            boxes_arr,
            labels_arr,
            num_classes=mcfg.num_classes,
            out_h=feat_h,
            out_w=feat_w,
            stride=mcfg.det_stride,
        )
        mask_target = cv2.resize(
            mask_det, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)[None]

        # --- keypoint crop + coordinate targets ----------------------------
        center, src_size = targets.bbox_to_center_size(primary["bbox"], mcfg.crop_pad_factor)
        rotation_deg = 0.0
        if self.do_aug:
            acfg = self.cfg.aug
            rng = worker_index_rng(
                self.cfg.train.seed,
                index,
                self._epoch,
                stream=_STREAM_CROP_GEOMETRY,
                occurrence=occurrence,
            )
            if rng.random() < acfg.crop_scale_prob:
                src_size *= 1.0 + rng.uniform(-acfg.crop_scale, acfg.crop_scale)
            if rng.random() < acfg.crop_shift_prob:
                center = (
                    center[0] + rng.uniform(-acfg.crop_shift, acfg.crop_shift) * src_size,
                    center[1] + rng.uniform(-acfg.crop_shift, acfg.crop_shift) * src_size,
                )
            if rng.random() < acfg.crop_rotate_prob:
                rotation_deg = float(rng.uniform(-acfg.crop_rotate, acfg.crop_rotate))
            # Right-angle offset for full-circle orientation coverage. It is
            # folded into the continuous angle so pixels are resampled only
            # once; warp_points remaps keypoint targets by that same transform.
            if rng.random() < acfg.crop_rot90_prob:
                rotation_deg += 90.0 * int(rng.integers(1, 4))
        warp = targets.topdown_warp_matrix(center, src_size, mcfg.crop_size, rotation_deg)
        crop_bgr = cv2.warpAffine(
            bgr,
            warp,
            (mcfg.crop_size, mcfg.crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
        )
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.do_aug:
            crop_rng = worker_index_rng(
                self.cfg.train.seed,
                index,
                self._epoch,
                stream=_STREAM_CROP_PHOTO,
                occurrence=occurrence,
            )
            crop_rgb = photometric_augment(crop_rgb, self.cfg.aug, crop_rng)

        kp_raw = np.asarray(primary.get("keypoints", []), dtype=np.float32)
        if kp_raw.size >= 3 * mcfg.num_keypoints:
            kp = kp_raw.reshape(-1, 3)[: mcfg.num_keypoints]
            kp_xy = kp[:, :2]
            vis = kp[:, 2]
            kp_crop = targets.warp_points(kp_xy, warp)
            if mcfg.keypoint_head == "simcc":
                simcc_x, simcc_y, kp_weight = targets.encode_simcc(
                    kp_crop,
                    vis,
                    mcfg.crop_size,
                    mcfg.simcc_split_ratio,
                    mcfg.simcc_sigma,
                )
            else:
                keypoint_heatmap, kp_weight = targets.encode_keypoint_heatmaps(
                    kp_crop,
                    vis,
                    mcfg.crop_size,
                    mcfg.crop_size // 4,
                    mcfg.keypoint_heatmap_sigma,
                )
            has_kp = 1.0
        else:
            kp_weight = np.zeros((mcfg.num_keypoints,), dtype=np.float32)
            if mcfg.keypoint_head == "simcc":
                bins = int(round(mcfg.crop_size * mcfg.simcc_split_ratio))
                simcc_x = np.zeros((mcfg.num_keypoints, bins), dtype=np.float32)
                simcc_y = np.zeros((mcfg.num_keypoints, bins), dtype=np.float32)
            else:
                heatmap_size = mcfg.crop_size // 4
                keypoint_heatmap = np.zeros(
                    (mcfg.num_keypoints, heatmap_size, heatmap_size),
                    dtype=np.float32,
                )
            has_kp = 0.0

        if mcfg.curve_head_enabled:
            projected_curves = self._load_curves(primary)
            if projected_curves is None:
                curve_heatmap = np.zeros(
                    (mcfg.num_curves, mcfg.crop_size // 4, mcfg.crop_size // 4),
                    dtype=np.float32,
                )
                curve_weight = np.zeros((mcfg.num_curves,), dtype=np.float32)
                has_curves = 0.0
            else:
                curves_crop = transform_projected_curves(projected_curves, warp)
                curve_heatmap, curve_weight = rasterize_curve_heatmaps(
                    curves_crop,
                    crop_size=mcfg.crop_size,
                    heatmap_size=mcfg.crop_size // 4,
                    sigma=mcfg.curve_target_sigma,
                )
                has_curves = 1.0

        sample = {
            "det_image": torch.from_numpy(normalize_image(det_rgb)),
            "heatmap": torch.from_numpy(heatmap),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "reg_weight": torch.from_numpy(reg_weight),
            "mask_target": torch.from_numpy(mask_target),
            "mask_weight": torch.tensor(mask_weight, dtype=torch.float32),
            "det_weight": torch.tensor(
                0.0 if self.keypoint_only else 1.0,
                dtype=torch.float32,
            ),
            "crop_image": torch.from_numpy(normalize_image(crop_rgb)),
            "kp_weight": torch.from_numpy(kp_weight),
            "has_kp": torch.tensor(has_kp, dtype=torch.float32),
            "img_id": torch.tensor(int(img_id), dtype=torch.long),
            # Pose-aware wrappers must apply the corresponding camera-frame
            # roll to their rotation label. Keeping the sampled value in the
            # sample prevents them from having to reproduce dataset RNG state.
            "crop_rotation_deg": torch.tensor(rotation_deg, dtype=torch.float32),
        }
        if mcfg.rotation_aux_enabled:
            from .rotation_aux import annotation_quaternion

            quaternion = annotation_quaternion(primary)
            sample["rotation_weight"] = torch.tensor(float(quaternion is not None))
            if quaternion is None:
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            sample["rotation_quaternion"] = torch.from_numpy(
                targets.rotate_object_to_camera_quaternion_for_crop(
                    quaternion, rotation_deg
                ).astype(np.float32)
            )
        if mcfg.keypoint_head == "simcc":
            sample["simcc_x"] = torch.from_numpy(simcc_x)
            sample["simcc_y"] = torch.from_numpy(simcc_y)
        else:
            sample["keypoint_heatmap"] = torch.from_numpy(keypoint_heatmap)
        if mcfg.curve_head_enabled:
            sample["curve_heatmap"] = torch.from_numpy(curve_heatmap)
            sample["curve_weight"] = torch.from_numpy(curve_weight)
            sample["has_curves"] = torch.tensor(has_curves, dtype=torch.float32)
        return sample
