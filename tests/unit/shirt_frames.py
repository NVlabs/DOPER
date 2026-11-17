# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 5-frame synthetic SHIRT tree, a stub model that returns projected keypoints, and goldens.

The tree follows the public SHIRT layout (``camera.json``, ``<roe>/<roe>.json``,
``<roe>/metadata.json``, ``<roe>/<domain>/images/<filename>``). Its poses, camera, metadata and
eight body landmarks come from ``synthetic_shirt``: frame ``i`` of ``roe1`` is metadata sample
``i`` and frame ``i`` of ``roe2`` is sample ``i + 5``. ``roe1`` labels use the released field
names, ``roe2`` the aliases (``file_name``, ``quaternion``, ``translation``) under ``images``.

Every image is a small uniform PNG whose gray level codes its trajectory and frame. The stub
model's ``infer_image`` reads that code and returns the frame's detection: the true projections
plus noise, with scripted failures on ``roe1``: an outlier (frame 1), a non-finite keypoint and
a low score (frame 2), no detection (frame 3) and only three confident keypoints, so no pose
(frame 4). The stub records the keyword arguments of every inference call.

``golden/source_shirt_framewise.json`` holds what the research framewise evaluation
(``evaluate_shirt_usac_inliers.py`` in the research source snapshot) writes for this tree with the
stub, in the
canonical form of ``canonical_framewise``, with the OpenCV version it was produced with (USAC).
Regenerate it with ``python tests/unit/shirt_frames.py <research-source-dir>``, an extracted copy
of the
research source snapshot. The module needs only numpy, OpenCV, SciPy and ``synthetic_shirt``: the
``shirt_framewise`` comparison loads it by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

FRAMES = 5
TRAJECTORIES = ("roe1", "roe2")
DOMAIN = "lightbox"
IMAGE_SIDE = 16
CODE_TRAJECTORY_STEP = 100
CODE_FRAME_STEP = 10
NOISE_PX = 0.3
OUTLIER_PX = (60.0, -45.0)
LOW_SCORE = 0.1
CAMERA_SIZE = 1024
# Rerank-only placeholder fields of the research ``candidate_record`` (never set by SHIRT).
RERANK_ONLY_FIELDS = ("heatmap_gap", "mode_inliers", "mask_iou")
VOLATILE_SUMMARY_FIELDS = ("inference_latency_ms",)
REGENERATE = (
    "regenerate it with `python tests/unit/shirt_frames.py <research-source-dir>` (the research "
    "framewise evaluation in the research source snapshot)"
)


def synthetic_shirt() -> Any:  # noqa: ANN401
    """The ``synthetic_shirt`` module next to this file (imported by path for the harness)."""
    if "synthetic_shirt" in sys.modules:
        return sys.modules["synthetic_shirt"]
    path = Path(__file__).resolve().parent / "synthetic_shirt.py"
    spec = importlib.util.spec_from_file_location("synthetic_shirt", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["synthetic_shirt"] = module
    spec.loader.exec_module(module)
    return module


def image_code(slot: int, index: int) -> int:
    """Gray level of frame ``index`` of trajectory ``slot``."""
    return CODE_TRAJECTORY_STEP * slot + CODE_FRAME_STEP * (index + 1)


def filename(index: int) -> str:
    """Image file name of frame ``index`` (SHIRT numbers from 1)."""
    return f"img{index + 1:06d}.png"


def speed_quaternion(rvec: np.ndarray) -> np.ndarray:
    """Scalar-first SPEED quaternion whose transposed DCM is the object-to-camera ``rvec``."""
    q_xyzw = Rotation.from_matrix(cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]).as_quat()
    return q_xyzw[[3, 0, 1, 2]]


def _detection(pixels: np.ndarray, scores: np.ndarray, score: float) -> SimpleNamespace:
    low, high = np.nanmin(pixels, axis=0), np.nanmax(pixels, axis=0)
    return SimpleNamespace(
        bbox_xyxy=np.asarray([*low, *high], dtype=np.float64),
        score=score,
        keypoints=pixels,
        keypoint_scores=scores,
    )


def _script_failures(index: int, pixels: np.ndarray, scores: np.ndarray) -> bool:
    """Apply ``roe1``'s scripted failure of frame ``index``; False means no detection."""
    if index == 1:
        pixels[2] += OUTLIER_PX
    elif index == 2:
        pixels[5] = np.nan
        scores[1] = LOW_SCORE
    elif index == 3:
        return False
    elif index == 4:
        scores[3:] = LOW_SCORE
    return True


def trajectory_frames(slot: int) -> list[dict[str, Any]]:
    """Ground truth and stub detection of every frame of trajectory ``slot``."""
    shirt = synthetic_shirt()
    metadata = shirt.make_metadata()
    points = shirt.make_keypoints_3d()
    rng = np.random.default_rng(100 + slot)
    frames = []
    for index in range(FRAMES):
        rvec, tvec = shirt.make_truth_pose(metadata, index + FRAMES * slot)
        pixels = cv2.projectPoints(points, rvec, tvec, shirt.CAMERA_MATRIX, shirt.DISTORTION)[0]
        pixels = pixels.reshape(-1, 2) + rng.normal(scale=NOISE_PX, size=(len(points), 2))
        scores = rng.uniform(0.5, 1.0, size=len(points))
        detected = _script_failures(index, pixels, scores) if slot == 0 else True
        if slot == 1 and index == 1:
            pixels[0] += OUTLIER_PX
        frames.append({
            "code": image_code(slot, index),
            "filename": filename(index),
            "quaternion": speed_quaternion(rvec),
            "translation": np.asarray(tvec, dtype=np.float64).reshape(3),
            "detection": _detection(pixels, scores, 0.9 - 0.05 * index) if detected else None,
        })  # fmt: skip
    return frames


def _label(frame: dict[str, Any], aliases: bool) -> dict[str, Any]:
    keys = ("file_name", "quaternion", "translation") if aliases else (
        "filename", "q_vbs2tango_true", "r_Vo2To_vbs_true")  # fmt: skip
    values = (frame["filename"], frame["quaternion"].tolist(), frame["translation"].tolist())
    return dict(zip(keys, values, strict=True))


def write_shirt_tree(root: Path) -> dict[int, SimpleNamespace | None]:
    """Write the SHIRT tree under ``root``; return the stub detections keyed by image code."""
    shirt = synthetic_shirt()
    root.mkdir(parents=True, exist_ok=True)
    camera = {"cameraMatrix": shirt.CAMERA_MATRIX.tolist(), "distCoeffs": shirt.DISTORTION.tolist(),
              "Nu": CAMERA_SIZE, "Nv": CAMERA_SIZE}  # fmt: skip
    (root / "camera.json").write_text(json.dumps(camera))
    detections: dict[int, SimpleNamespace | None] = {}
    for slot, roe in enumerate(TRAJECTORIES):
        frames = trajectory_frames(slot)
        labels = [_label(frame, aliases=slot == 1) for frame in frames]
        roe_dir = root / roe
        images = roe_dir / DOMAIN / "images"
        images.mkdir(parents=True)
        (roe_dir / f"{roe}.json").write_text(json.dumps({"images": labels} if slot else labels))
        (roe_dir / "metadata.json").write_text(json.dumps(shirt.make_metadata()))
        for frame in frames:
            image = np.full((IMAGE_SIDE, IMAGE_SIDE, 3), frame["code"], dtype=np.uint8)
            cv2.imwrite(str(images / frame["filename"]), image)
            detections[frame["code"]] = frame["detection"]
    return detections


def write_layout(path: Path) -> Path:
    """Write the eight ``synthetic_shirt`` landmarks as a ``keypoints_3d.json`` layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = synthetic_shirt().make_keypoints_3d()
    path.write_text(json.dumps({"keypoints_3d": points.tolist()}))
    return path


class StubModel:
    """Stand-in model: holds the scripted detections and the inference calls it served."""

    def __init__(self, detections: dict[int, SimpleNamespace | None]) -> None:
        self.detections = detections
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> StubModel:
        """Match ``torch.nn.Module.eval``."""
        return self


def make_stub_loader(detections: dict[int, SimpleNamespace | None]) -> Any:  # noqa: ANN401
    """A ``load_model(checkpoint, cfg, device)`` replacement returning one shared stub."""
    model = StubModel(detections)

    def load_model(checkpoint: Any, cfg: Any, device: str) -> StubModel:  # noqa: ANN401
        return model

    load_model.model = model  # type: ignore[attr-defined]
    return load_model


def stub_infer_image(
    model: StubModel,
    image: np.ndarray,
    cfg: Any,
    device: str,
    **kwargs: Any,  # noqa: ANN401
) -> list[SimpleNamespace]:
    """``infer_image`` replacement: the detection of the frame the image's gray level codes."""
    model.calls.append(dict(kwargs))
    detection = model.detections[int(image[0, 0, 0])]
    return [] if detection is None else [detection]


def canonical_framewise(out: Path) -> dict[str, Any]:
    """Per-trajectory records and summaries of a framewise run, without volatile fields.

    The research records carry the rerank-only placeholders ``RERANK_ONLY_FIELDS`` (constant
    for SHIRT); the port does not write them.
    """
    canonical: dict[str, Any] = {}
    for roe in TRAJECTORIES:
        folder = out / roe / DOMAIN
        records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        for record in records:
            for field in RERANK_ONLY_FIELDS:
                (record["baseline"] or {}).pop(field, None)
        summary = json.loads((folder / "summary.json").read_text())
        for field in VOLATILE_SUMMARY_FIELDS:
            summary.pop(field)
        canonical[roe] = {"records": records, "summary": summary}
    return canonical


def library_versions() -> dict[str, str]:
    """The build the golden depends on (USAC)."""
    return {"opencv": cv2.__version__}


def source_argv(checkpoint: Path, shirt_root: Path, layout: Path, out: Path) -> list[str]:
    """Command line of the research framewise evaluation on this tree (lightbox, CPU)."""
    return ["evaluate_shirt_usac_inliers", "--checkpoint", str(checkpoint),
            "--shirt-root", str(shirt_root), "--keypoints-3d", str(layout),
            "--output-dir", str(out), "--domains", DOMAIN,
            "--num-keypoints", str(len(synthetic_shirt().make_keypoints_3d())),
            "--overlay-count", "0", "--device", "cpu"]  # fmt: skip


def run_source_framewise(source_root: Path, work: Path) -> Path:
    """Run the research framewise evaluation on the tree with the stub model."""
    import contextlib
    import io
    from unittest import mock

    sys.path.insert(0, str(source_root))
    import scripts.evaluate_shirt_usac_inliers as source

    shirt_root = work / "shirt"
    loader = make_stub_loader(write_shirt_tree(shirt_root))
    checkpoint = work / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    out = work / "source_out"
    argv = source_argv(checkpoint, shirt_root, write_layout(work / "keypoints_3d.json"), out)
    with mock.patch.object(sys, "argv", argv), mock.patch.object(source, "load_model", loader), \
            mock.patch.object(source, "infer_image", stub_infer_image), \
            contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
        source.main()
    return out


def write_source_golden(source_root: Path, golden: Path) -> None:
    """Write ``source_shirt_framewise.json`` from the research framewise evaluation."""
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        out = run_source_framewise(source_root, Path(temporary).resolve())
        payload = {**canonical_framewise(out), "versions": library_versions()}
    (golden / "source_shirt_framewise.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    write_source_golden(Path(sys.argv[1]), Path(__file__).resolve().parent / "golden")
