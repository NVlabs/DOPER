# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A SHIRT tree for the 0/180 re-crop candidates, framewise input records and a heatmap stub.

The tree reuses ``shirt_frames``' labels, camera and metadata (two 5-frame trajectories), but
every image is ``Nu`` x ``Nv`` and draws each body landmark's true projection as a small disc
whose gray level codes the landmark (``LANDMARK_LEVELS``) on a black background. A pose-projected
crop of such an image, rolled by any angle, still shows every landmark, so the stub model's
heatmaps (``StubCropModel``: a soft match of each crop pixel's gray level to each landmark's,
box-filtered) decode back to the true projections. ``BLANK_FRAMES`` have no discs: their crops
give flat heatmaps, so no re-crop candidate solves.

``write_framewise_records`` writes the framewise records the candidate stage reads: the raw
(``baseline``) pose of each frame is its true pose perturbed by ``RAW_ROTATION_DEG`` and
``RAW_TRANSLATION_M``, with null baselines for ``FAILED_RAW_FRAMES`` (frames whose raw PnP
failed). The stub loader fills the loaded config's crop geometry (``CROP_SIZE``,
``CROP_PAD_FACTOR``, not the defaults) as ``load_model`` does from a checkpoint.

``golden/source_shirt_candidates.json`` holds the research candidate generation's re-crop
candidates (``evaluate_shirt_auxiliary_flip_gate.score_and_generate_candidates`` in the
research source snapshot) for this tree with the stub, in the canonical form of
``canonical_candidates``, with the library versions it was produced with. Regenerate it with
``python tests/unit/shirt_crops.py <research-source-dir>``, an extracted copy of the research
source snapshot. The ``shirt_candidates`` comparison loads this module by path.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional


def _sibling(name: str) -> Any:  # noqa: ANN401
    """Import the test module ``name`` next to this file by path (for loading by path)."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sf = _sibling("shirt_frames")

TRAJECTORIES = sf.TRAJECTORIES
DOMAIN = sf.DOMAIN
FRAMES = sf.FRAMES
CROP_SIZE = 128
CROP_PAD_FACTOR = 1.3
DISC_RADIUS_PX = 3
LANDMARK_LEVELS = (30, 55, 80, 140, 165, 190, 215, 240)
LEVEL_SIGMA = 6.0
POOL_KERNEL = 3
LOGIT_SCALE = 30.0
# ImageNet normalization of the crops (``doper.dataset.normalize_image``), red channel.
RED_MEAN = 0.485
RED_STD = 0.229
PIXEL_MAX = 255.0
# Raw pose perturbation; (trajectory, frame) pairs whose raw PnP failed or whose image is blank.
RAW_ROTATION_DEG = 2.0
RAW_TRANSLATION_M = (0.02, -0.01, 0.05)
RAW_INLIERS = 8
FAILED_RAW_FRAMES = (("roe1", 3),)
BLANK_FRAMES = (("roe2", 2),)
ROTATION_TTA_ANGLES = "0,180"
SOURCE_BATCH_SIZE = 3
REGENERATE = (
    "regenerate it with `python tests/unit/shirt_crops.py <research-source-dir>` (the research "
    "candidate generation in the research source snapshot)"
)


def _poses(slot: int) -> list[tuple[np.ndarray, np.ndarray]]:
    shirt = sf.synthetic_shirt()
    metadata = shirt.make_metadata()
    return [shirt.make_truth_pose(metadata, index + FRAMES * slot) for index in range(FRAMES)]


def _image(rvec: np.ndarray, tvec: np.ndarray, blank: bool) -> np.ndarray:
    shirt = sf.synthetic_shirt()
    image = np.zeros((sf.CAMERA_SIZE, sf.CAMERA_SIZE, 3), dtype=np.uint8)
    if blank:
        return image
    points = shirt.make_keypoints_3d()
    pixels = cv2.projectPoints(points, rvec, tvec, shirt.CAMERA_MATRIX, shirt.DISTORTION)[0]
    for (x, y), level in zip(pixels.reshape(-1, 2), LANDMARK_LEVELS, strict=True):
        center = (round(float(x)), round(float(y)))
        cv2.circle(image, center, DISC_RADIUS_PX, (level, level, level), thickness=-1)
    return image


def write_crop_tree(root: Path) -> Path:
    """Write the SHIRT tree with landmark-disc images under ``root``; return ``root``."""
    sf.write_shirt_tree(root)
    for slot, roe in enumerate(TRAJECTORIES):
        images = root / roe / DOMAIN / "images"
        for index, (rvec, tvec) in enumerate(_poses(slot)):
            blank = (roe, index) in BLANK_FRAMES
            cv2.imwrite(str(images / sf.filename(index)), _image(rvec, tvec, blank))
    return root


def _raw_pose(rvec: np.ndarray, tvec: np.ndarray) -> dict[str, Any]:
    """The perturbed raw pose with its reporting fields (as the framewise records hold them)."""
    axis = np.asarray([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    delta = cv2.Rodrigues(axis * math.radians(RAW_ROTATION_DEG))[0]
    rotation = delta @ cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    truth = np.asarray(tvec, dtype=np.float64).reshape(3)
    translation = truth + np.asarray(RAW_TRANSLATION_M)
    translation_m = float(np.linalg.norm(translation - truth))
    relative = translation_m / float(np.linalg.norm(truth))
    return {
        "rvec": cv2.Rodrigues(rotation)[0].reshape(3).tolist(),
        "tvec": translation.tolist(),
        "ransac_inliers": RAW_INLIERS,
        "rotation_deg": RAW_ROTATION_DEG,
        "translation_m": translation_m,
        "translation_relative": relative,
        "pose_score": math.radians(RAW_ROTATION_DEG) + relative,
        "practical_success": relative < 0.02,
        "loose_success": relative < 0.05,
    }


def write_framewise_records(
    records_root: Path, failed: tuple[tuple[str, int], ...] = FAILED_RAW_FRAMES
) -> Path:
    """Write ``<records_root>/<roe>/<domain>/records.jsonl``; ``failed`` frames have no pose."""
    for slot, roe in enumerate(TRAJECTORIES):
        rows = []
        for index, (frame, (rvec, tvec)) in enumerate(
            zip(sf.trajectory_frames(slot), _poses(slot), strict=True)
        ):
            rows.append({
                "index": index,
                "filename": frame["filename"],
                "ground_truth_quaternion": frame["quaternion"].tolist(),
                "ground_truth_translation": frame["translation"].tolist(),
                "baseline": None if (roe, index) in failed
                else _raw_pose(rvec, tvec),
            })  # fmt: skip
        folder = records_root / roe / DOMAIN
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return records_root


def _identity(value: Any) -> Any:  # noqa: ANN401
    return value


class StubCropModel:
    """Stand-in heatmap model: ``backbone``, ``kpt_head.neck`` and ``kpt_head.tower`` pass the
    crops through and ``kpt_head.out`` returns one heatmap per landmark at crop resolution.

    It records the shape of every crop batch it served.
    """

    def __init__(self) -> None:
        self.batches: list[list[int]] = []
        self.kpt_head = SimpleNamespace(neck=_identity, tower=_identity, out=self._heatmaps)

    def eval(self) -> StubCropModel:
        """Match ``torch.nn.Module.eval``."""
        return self

    def backbone(self, crops: torch.Tensor) -> torch.Tensor:
        """Record the batch and pass the crops through."""
        self.batches.append(list(crops.shape))
        return crops

    def _heatmaps(self, crops: torch.Tensor) -> torch.Tensor:
        grey = (crops[:, 0] * RED_STD + RED_MEAN) * PIXEL_MAX
        levels = torch.tensor(LANDMARK_LEVELS, dtype=crops.dtype).view(1, -1, 1, 1)
        membership = torch.exp(-(((grey.unsqueeze(1) - levels) / LEVEL_SIGMA) ** 2))
        pooled = functional.avg_pool2d(membership, POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2)
        return LOGIT_SCALE * pooled


def make_stub_loader() -> Any:  # noqa: ANN401
    """A ``load_model(checkpoint, cfg, device)`` replacement returning one shared stub.

    Like the real loader it fills ``cfg.model``'s crop geometry from "the checkpoint".
    """
    model = StubCropModel()

    def load_model(checkpoint: Any, cfg: Any, device: str) -> StubCropModel:  # noqa: ANN401
        cfg.model.crop_size = CROP_SIZE
        cfg.model.crop_pad_factor = CROP_PAD_FACTOR
        cfg.model.keypoint_head = "heatmap"
        return model

    load_model.model = model  # type: ignore[attr-defined]
    return load_model


def write_source_checkpoint(path: Path) -> Path:
    """A checkpoint with the ``rotation_aux`` payload the research script requires."""
    feature_dim = 3 * 16
    generator = torch.Generator().manual_seed(0)
    bins = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    torch.save({"model": {}, "rotation_aux": {
        "weight": torch.randn(len(bins), feature_dim, generator=generator),
        "feature_dim": feature_dim, "num_bins": len(bins), "pool_size": 4,
        "logit_scale": 10.0, "bin_quaternions": bins,
    }}, path)  # fmt: skip
    return path


def source_args(
    checkpoint: Path, records_root: Path, shirt_root: Path, layout: Path, out: Path
) -> Namespace:
    """Arguments of the research candidate generation on this tree (lightbox, CPU)."""
    return Namespace(
        checkpoint=checkpoint, records_root=records_root, shirt_root=shirt_root,
        keypoints_3d=layout, output_dir=out, trajectories=",".join(TRAJECTORIES), domain=DOMAIN,
        rotation_tta_angles=ROTATION_TTA_ANGLES, soft_target_sigma_deg=15.0,
        aux_kl_threshold=0.4, prior_gate_deg=30.0, batch_size=SOURCE_BATCH_SIZE, num_workers=0,
        limit=0, device="cpu", reuse_scored=False,
    )  # fmt: skip


def canonical(records_by_roe: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Each frame's candidate sources, rvecs and tvecs, by trajectory; frames with candidates."""
    return {
        roe: [
            {
                "index": record["index"],
                "filename": record["filename"],
                "candidates": [
                    {key: candidate[key] for key in ("source", "rvec", "tvec")}
                    for candidate in record["candidates"]
                ],
            }
            for record in sorted(records, key=lambda record: record["index"])
            if record["candidates"]
        ]
        for roe, records in records_by_roe.items()
    }


def canonical_candidates(out: Path) -> dict[str, Any]:
    """The canonical form of a candidate run's ``<out>/<roe>/<domain>/records.jsonl``."""
    return canonical({
        roe: [json.loads(line) for line in
              (out / roe / DOMAIN / "records.jsonl").read_text().splitlines()]
        for roe in TRAJECTORIES
    })  # fmt: skip


def library_versions() -> dict[str, str]:
    """The builds the golden depends on (USAC, the stub's torch kernels)."""
    return {"opencv": cv2.__version__, "torch": torch.__version__.split("+")[0]}


def run_source_candidates(source_root: Path, work: Path) -> dict[str, Any]:
    """Run the research candidate generation on the tree with the stub model."""
    import contextlib
    import io
    import warnings
    from unittest import mock

    sys.path.insert(0, str(source_root))
    import scripts.evaluate_shirt_auxiliary_flip_gate as source

    shirt_root = write_crop_tree(work / "shirt")
    records_root = write_framewise_records(work / "framewise")
    layout = sf.write_layout(work / "keypoints_3d.json")
    checkpoint = write_source_checkpoint(work / "checkpoint.pth")
    args = source_args(checkpoint, records_root, shirt_root, layout, work / "source_out")
    with mock.patch.object(source, "load_model", make_stub_loader()), \
            contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():  # fmt: skip
        warnings.simplefilter("ignore")
        scored, _ = source.score_and_generate_candidates(args)
    return canonical(scored)


def write_source_golden(source_root: Path, golden: Path) -> None:
    """Write ``source_shirt_candidates.json`` from the research candidate generation."""
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        payload = {**run_source_candidates(source_root, Path(temporary).resolve()),
                   "versions": library_versions()}  # fmt: skip
    (golden / "source_shirt_candidates.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    write_source_golden(Path(sys.argv[1]), Path(__file__).resolve().parent / "golden")
