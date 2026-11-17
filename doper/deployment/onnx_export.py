# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Freeze PyTorch references on real frames and export the two DOPER branches to ONNX.

A bundle folder holds ``det.onnx`` (fixed batch 1) and ``crop.onnx`` (dynamic batch), one
``ref<frame>_<branch>.npz`` per branch call with its input and outputs, the PyTorch poses
in ``reference_poses.json``, the frames as ``frame<NNN>.png``, ``config.json``,
``camera.json``, ``keypoints.json`` and a ``manifest.json`` hashing all of them. Export
runs on CPU.

Usage::

    python -m doper.deployment.onnx_export --checkpoint best.pth \
        --keypoints-3d keypoints_3d.json --camera camera.json \
        --video take.mp4 --frames 12 --output bundle/

Needs the ``tensorrt`` extra (for ``onnx``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from doper.config import DataConfig, DoperConfig
from doper.console import log
from doper.deployment import bundle
from doper.deployment.parity import detections_json
from doper.infer import infer_image, load_model
from doper.keypoints import load_keypoints_3d
from doper.model import DoperNet

ONNX_OPSET = 17
DEFAULT_SCORE_THRESHOLD = 0.1
DEFAULT_MAX_DETECTIONS = 3
DEFAULT_VIDEO_FRAMES = 12
DISTORTION_COEFFICIENT_COUNT = 5
DISTORTION_KEYS = ("distortion", "dist_coeffs", "distortion_coefficients")
FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
DET_OUTPUTS = ("heatmap", "wh", "offset", "mask")
CROP_OUTPUTS = ("keypoint_heatmap",)
INPUT_NAME = "input"
# Identity fields a cached bundle must match to be reused.
CACHE_KEYS = (
    "export_source_sha256",
    "checkpoint_sha256",
    "keypoints_sha256",
    "camera_sha256",
    "frames_sha256",
    "detector_size",
    "score_thr",
    "max_det",
)


class BranchModule(torch.nn.Module):
    """Expose one ``DoperNet`` branch as a module returning a tuple of tensors for export.

    Attributes:
        output_names: Names of the returned tensors, in order.
    """

    def __init__(self, model: DoperNet, branch: Literal["det", "crop"]) -> None:
        """Wrap ``model.forward_det`` or ``model.forward_crop``.

        Raises:
            ValueError: For a branch other than ``det`` or ``crop``.
        """
        super().__init__()
        if branch not in ("det", "crop"):
            raise ValueError(f"branch must be 'det' or 'crop', got {branch!r}")
        self.model = model
        self.branch = branch
        self.output_names = list(DET_OUTPUTS if branch == "det" else CROP_OUTPUTS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the branch on ``x`` and return its outputs in :attr:`output_names` order."""
        out = self.model.forward_det(x) if self.branch == "det" else self.model.forward_crop(x)
        return tuple(out[name] for name in self.output_names)


class _Recorder:
    """Stand-in model for ``infer_image`` that saves every branch call to an npz file."""

    def __init__(self, model: DoperNet, folder: Path) -> None:
        self.model = model
        self.folder = folder
        self.frame = 0
        self.keypoints_3d_fingerprint = getattr(model, "keypoints_3d_fingerprint", None)

    def _call(self, branch: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = getattr(self.model, f"forward_{branch}")(x)
        path = self.folder / f"ref{self.frame:03d}_{branch}.npz"
        if path.exists():
            raise RuntimeError(f"{branch} branch called twice on frame {self.frame}")
        np.savez(path, input=x.numpy(), **{k: v.numpy() for k, v in y.items()})
        return y

    def forward_det(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._call("det", x)

    def forward_crop(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._call("crop", x)


def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx is required for doper.deployment.onnx_export; install doper[tensorrt]"
        ) from exc


def _load_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``camera_matrix`` and optional distortion (zeros if absent) from JSON."""
    data = json.loads(Path(path).read_text())
    camera_matrix = np.asarray(data["camera_matrix"], np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"{path}: camera_matrix must be 3x3")
    key = next((k for k in DISTORTION_KEYS if k in data), None)
    dist = data[key] if key else [0.0] * DISTORTION_COEFFICIENT_COUNT
    return camera_matrix, np.asarray(dist, np.float64).ravel()


def _read_frame_dir(folder: Path) -> list[np.ndarray]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
    if not paths:
        raise ValueError(f"No frames ({', '.join(FRAME_SUFFIXES)}) in {folder}")
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read frame {path}")
        frames.append(frame)
    return frames


def read_video_frames(video: Path, count: int) -> list[np.ndarray]:
    """Read ``count`` frames spread evenly over a video, first and last included.

    Raises:
        ValueError: If the video has fewer than ``count`` frames or a frame cannot be read.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1 or total < count:
            raise ValueError(f"Not enough source frames in {video}: {total} < {count}")
        frames = []
        for index in np.linspace(0, total - 1, count, dtype=int).tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot read {video}:{index}")
            frames.append(frame)
        return frames
    finally:
        capture.release()


def _frame_sha256(frame: np.ndarray) -> str:
    """Hash a frame's shape, dtype and pixels so a cache is tied to the exact input."""
    array = np.ascontiguousarray(frame)
    header = f"{array.shape}|{array.dtype.str}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _reuse_cache(output: Path, identity: dict[str, Any]) -> bool:
    """Return True for a reusable bundle in ``output``, False for a missing or empty one.

    Raises:
        ValueError: If ``output`` holds anything other than a bundle whose identity and
            every recorded file hash match.
    """
    if not output.exists() or not any(output.iterdir()):
        return False
    if not (output / bundle.MANIFEST_NAME).is_file():
        raise ValueError(f"Output directory is not empty: {output}; use a fresh directory")
    try:
        cached = bundle.verify(output)
    except ValueError as exc:
        raise ValueError(f"Incompatible export cache: {output}: {exc}") from exc
    if any(cached.get(key) != identity[key] for key in CACHE_KEYS):
        raise ValueError(f"Incompatible export cache: {output}; use a fresh output directory")
    return True


def _export_branch(model: DoperNet, branch: Literal["det", "crop"], size: int, path: Path) -> None:
    wrapper = BranchModule(model, branch)
    names = [INPUT_NAME, *wrapper.output_names]
    dynamic = None if branch == "det" else {name: {0: "batch"} for name in names}
    with warnings.catch_warnings():
        # The legacy TorchScript exporter is chosen deliberately (dynamo=False); its
        # deprecation notices and GroupNorm's trace-time batch-size check are expected.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx")
        warnings.filterwarnings(
            "ignore", message="You are using the legacy TorchScript", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore", message="Converting a tensor to a Python boolean", module=r"torch\.nn"
        )
        torch.onnx.export(
            wrapper,
            (torch.zeros(1, 3, size, size),),
            str(path),
            input_names=[INPUT_NAME],
            output_names=wrapper.output_names,
            dynamic_axes=dynamic,
            opset_version=ONNX_OPSET,
            dynamo=False,
        )


def _check_exportable(cfg: DoperConfig) -> None:
    if cfg.model.keypoint_head != "heatmap" or cfg.model.curve_head_enabled:
        raise ValueError(
            "ONNX export supports heatmap keypoint checkpoints without a curve head; got "
            f"keypoint_head={cfg.model.keypoint_head!r}, "
            f"curve_head_enabled={cfg.model.curve_head_enabled}"
        )


def _write_inputs(
    output: Path, cfg: DoperConfig, keypoints_3d: Path, camera_matrix: np.ndarray, dist: np.ndarray
) -> None:
    shutil.copy2(keypoints_3d, output / "keypoints.json")
    camera_json = {"camera_matrix": camera_matrix.tolist(), "distortion": dist.tolist()}
    (output / "camera.json").write_text(json.dumps(camera_json, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def _freeze_references(
    model: DoperNet,
    cfg: DoperConfig,
    frames: Sequence[np.ndarray],
    output: Path,
    infer_kwargs: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    recorder = _Recorder(model, output)
    poses = []
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            if not cv2.imwrite(str(output / f"frame{i:03d}.png"), frame):
                raise OSError(f"Cannot write frame {i} to {output}")
            recorder.frame = i
            preds = infer_image(recorder, frame, cfg, "cpu", **infer_kwargs)  # type: ignore[arg-type]
            poses.append(detections_json(preds))
    return poses


def export_bundle(
    checkpoint: Path,
    keypoints_3d: Path,
    camera: Path,
    frames: Sequence[np.ndarray] | str | Path,
    output: Path,
    *,
    det_size: int | None = None,
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
    max_det: int = DEFAULT_MAX_DETECTIONS,
) -> Path:
    """Export a deployment bundle for one checkpoint on CPU.

    Args:
        checkpoint: Heatmap-head checkpoint without a curve head.
        keypoints_3d: ``keypoints_3d.json`` used for PnP.
        camera: JSON with ``camera_matrix`` and optional ``distortion`` (also accepted as
            ``dist_coeffs`` or ``distortion_coefficients``; zeros if absent).
        frames: BGR frames, or a directory (``str`` or ``Path``) of image files read in sorted
            order.
        output: Bundle directory. A missing or empty directory is filled; an existing bundle
            is reused only when its identity and every recorded file hash match.
        det_size: Detector input size; ``None`` keeps the checkpoint's.
        score_thr: Detection score threshold of the reference pipeline.
        max_det: Maximum detections per frame of the reference pipeline.

    Returns:
        ``output``.

    Raises:
        ImportError: Without the ``onnx`` package (``tensorrt`` extra).
        ValueError: For an unsupported checkpoint, unreadable frames, or an ``output`` that
            is neither empty nor a matching bundle.
    """
    _require_onnx()
    output = Path(output)
    frames = _read_frame_dir(Path(frames)) if isinstance(frames, str | Path) else list(frames)
    if not frames:
        raise ValueError("export_bundle needs at least one frame")
    cfg = DoperConfig(data=DataConfig(root=""))
    model = load_model(checkpoint, cfg, "cpu")
    if det_size is not None:
        cfg.model.det_input_size = det_size
    _check_exportable(cfg)
    identity = {
        "export_source_sha256": bundle.sha256(Path(__file__)),
        "checkpoint_sha256": bundle.sha256(checkpoint),
        "keypoints_sha256": bundle.sha256(keypoints_3d),
        "camera_sha256": bundle.sha256(camera),
        "frames_sha256": [_frame_sha256(frame) for frame in frames],
        "detector_size": cfg.model.det_input_size,
        "score_thr": score_thr,
        "max_det": max_det,
    }
    if _reuse_cache(output, identity):
        return output
    output.mkdir(parents=True, exist_ok=True)
    camera_matrix, dist = _load_camera(camera)
    _write_inputs(output, cfg, keypoints_3d, camera_matrix, dist)
    infer_kwargs = {
        "score_thr": score_thr,
        "max_det": max_det,
        "keypoints_3d": load_keypoints_3d(keypoints_3d).astype(np.float32),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist,
    }
    poses = _freeze_references(model, cfg, frames, output, infer_kwargs)
    (output / "reference_poses.json").write_text(json.dumps(poses))
    _export_branch(model, "det", cfg.model.det_input_size, output / "det.onnx")
    _export_branch(model, "crop", cfg.model.crop_size, output / "crop.onnx")
    entry = {
        **identity,
        "checkpoint": Path(checkpoint).name,
        "crop_size": cfg.model.crop_size,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    bundle.write_manifest(output, entry)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a DOPER checkpoint to an ONNX deployment bundle"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keypoints-3d", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True, help="camera_matrix JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="video to sample frames from")
    source.add_argument("--frames-dir", type=Path, help="directory of frame images")
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_VIDEO_FRAMES,
        help="frames sampled evenly from --video",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-size", type=int, default=None, help="default: checkpoint's")
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DETECTIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bundle export from the command line."""
    args = _build_parser().parse_args(argv)
    frames = (
        read_video_frames(args.video, args.frames) if args.video is not None else args.frames_dir
    )
    torch.manual_seed(0)
    output = export_bundle(
        args.checkpoint,
        args.keypoints_3d,
        args.camera,
        frames,
        args.output,
        det_size=args.det_size,
        score_thr=args.score_thr,
        max_det=args.max_det,
    )
    log(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
