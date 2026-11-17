# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-atomic rendering and annotation checkpoints, independent of Kit imports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code
from doper.scene import labels
from rendering.environment_bundle import validate_applied_readbacks
from rendering.resume import FlushPolicy, load_resume_state, sort_by_image_id
from rendering.shard_io import write_json

PNG_COMPRESS_LEVEL = 3


class Scene(Protocol):
    """What the worker needs from a realized scene (``KitScene`` or a test double).

    ``environment`` (the native replay adapter) is read only when the inputs replay one.
    """

    environment: Any

    def realize(self, record: dict[str, Any]) -> None:
        """Apply every sample-specific part of ``record``."""
        ...

    def set_camera(self, pose: np.ndarray) -> None:
        """Place the render camera at the camera-to-world ``pose``."""
        ...


class Capture(Protocol):
    """What the worker needs from a frame capture (``FrameCapture`` or a test double)."""

    def prime(self) -> None:
        """Initialize the render graph before the first saved pose."""
        ...

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RGB, depth and object mask of one valid frame."""
        ...


class RenderWorker:
    """Commit records only after every view of a sample has been saved."""

    def __init__(self, inputs: Any, scene: Scene, capture: Capture) -> None:  # noqa: ANN401
        """Bind validated ``WorkerInputs`` (or an equivalent namespace), scene and capture."""
        self.inputs, self.scene, self.capture = inputs, scene, capture
        self.args, self.cfg = inputs.args, inputs.cfg
        self.outdir = Path(self.args.outdir)
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []

    @property
    def _environment(self) -> Any:  # noqa: ANN401 - ReplayInputs or None
        return getattr(self.inputs, "environment", None)

    def prepare(self) -> set[int]:
        """Create output folders and, on resume, reload this worker's finished samples.

        Returns:
            Sample indices that are already complete.
        """
        (self.outdir / "images").mkdir(parents=True, exist_ok=True)
        if self._environment is not None:
            (self.outdir / "environment_applied").mkdir(parents=True, exist_ok=True)
        if self.cfg.render_masks:
            (self.outdir / self.cfg.mask_dir).mkdir(parents=True, exist_ok=True)
        if not self.args.resume:
            return set()
        state = load_resume_state(
            str(self.outdir),
            self.args.worker_id,
            start_idx=self.args.start_idx,
            end_idx=self.args.end_idx,
            images_per_sample=self.cfg.images_per_sample,
            render_masks=self.cfg.render_masks,
        )
        if self._environment is not None:
            validate_applied_readbacks(
                self.outdir,
                [self.inputs.scenes[index] for index in state.done_indices],
                self.cfg,
                self._environment.identity,
                bundle=self._environment.bundle,
            )
        self.images, self.annotations = state.images, state.annotations
        return set(state.done_indices)

    def annotate(
        self,
        index: int,
        view: tuple[str, int, np.ndarray],
        object_pose: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        filename: str,
        mask_path: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the COCO image and annotation records of one saved view.

        Args:
            index: Sample index.
            view: View name, camera index and camera-to-world pose.
            object_pose: Object-to-world pose.
            depth: Rendered distance-to-image-plane, used for keypoint visibility.
            mask: Visible-object mask.
            filename: Image file name.
            mask_path: Mask path relative to the dataset, or ``None``.

        Returns:
            The image and annotation records.
        """
        cfg, intr = self.cfg, self.inputs.intr
        name, camera_index, camera_pose = view
        image_id = index * cfg.images_per_sample + camera_index
        uv, kp_depth = G.project_points(
            G.transform_points(object_pose, self.inputs.kp_local), camera_pose, intr
        )
        visibility = G.keypoint_visibility(uv, kp_depth, depth, intr, tol=cfg.occlusion_tol_m)
        bbox = (
            labels.mask_bbox(mask)
            if cfg.bbox_source == "mask" and mask.any()
            else labels.keypoints_bbox(uv, visibility, cfg.resolution, cfg.resolution)
        )
        stereo = (
            {
                "pair_id": index,
                "stereo_baseline_m": round(cfg.stereo_baseline_m, 6),
                "stereo_mode": "parallel",
            }
            if cfg.stereo_enabled
            else None
        )
        image = labels.image_record(
            image_id,
            filename,
            cfg.resolution,
            cfg.resolution,
            pair_id=index if cfg.stereo_enabled else None,
            view=name,
            camera_index=camera_index,
        )
        annotation = labels.annotation_record(
            image_id,
            uv=uv,
            vis=visibility,
            bbox=bbox,
            intr=intr,
            cam_to_world=camera_pose,
            object_pose=object_pose,
            view=name,
            camera_index=camera_index,
            mask_file=mask_path,
            stereo=stereo,
        )
        return image, annotation

    def _views(self, camera: np.ndarray) -> list[tuple[str, int, np.ndarray]]:
        if not self.cfg.stereo_enabled:
            return [("mono", 0, camera)]
        return [
            (name, number, G.stereo_offset(camera, direction * self.cfg.stereo_baseline_m))
            for name, number, direction in (
                (self.cfg.stereo_left_suffix, 0, -0.5),
                (self.cfg.stereo_right_suffix, 1, 0.5),
            )
        ]

    def render_sample(self, record: dict[str, Any]) -> None:
        """Realize ``record``, save every view, and commit its records together."""
        self.scene.realize(record)
        index = record["index"]
        camera = np.asarray(record["camera"]["cam_to_world"])
        obj = np.asarray(record["object"]["pose"])
        settle = (
            self._environment is not None
            or getattr(self.scene, "procedural_environment", None) is not None
        )
        records = []
        for view in self._views(camera):
            name, _, pose = view
            self.scene.set_camera(pose)
            if settle:
                # Match the calibrated environment's discarded settling capture.
                self.capture.capture()
            rgb, depth, mask = self.capture.capture()
            filename = f"{index:06d}_{name}.png" if self.cfg.stereo_enabled else f"{index:06d}.png"
            if self._environment is not None:
                write_json(
                    self.outdir / "environment_applied" / (Path(filename).stem + ".json"),
                    self.scene.environment.verify(record),
                )
            Image.fromarray(rgb).save(
                self.outdir / "images" / filename, compress_level=PNG_COMPRESS_LEVEL
            )
            mask_path = None
            if self.cfg.render_masks:
                mask_path = f"{self.cfg.mask_dir}/{filename}"
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    self.outdir / mask_path, compress_level=PNG_COMPRESS_LEVEL
                )
            records.append(self.annotate(index, view, obj, depth, mask, filename, mask_path))
        # Partial images may survive interruption, but cannot count as a complete sample.
        self.images.extend(image for image, _ in records)
        self.annotations.extend(annotation for _, annotation in records)

    def checkpoint(self, *, partial: bool) -> Path:
        """Write the sorted records as the partial or final worker annotation file."""
        self.images, self.annotations = sort_by_image_id(self.images, self.annotations)
        payload = labels.worker_payload(
            worker_id=self.args.worker_id,
            start_idx=self.args.start_idx,
            end_idx=self.args.end_idx,
            images=self.images,
            annotations=self.annotations,
            render_masks=self.cfg.render_masks,
            mask_dir=self.cfg.mask_dir,
            stereo_enabled=self.cfg.stereo_enabled,
            stereo_baseline_m=self.cfg.stereo_baseline_m,
        )
        stem = self.outdir / f"annotations_worker_{self.args.worker_id:03d}"
        path = stem.with_suffix(".partial.json" if partial else ".json")
        write_json(path, payload)
        if not partial:
            stem.with_suffix(".partial.json").unlink(missing_ok=True)
        return path

    def run(self) -> Path:
        """Render every unfinished sample of the range, checkpointing as configured.

        Returns:
            The final annotation file.
        """
        done = self.prepare()
        policy = FlushPolicy(self.args.flush_every, self.args.flush_every_sec)
        started, rendered = time.monotonic(), 0
        try:
            for index in range(self.args.start_idx, self.args.end_idx):
                if index in done:
                    continue
                if rendered == 0:
                    # Initialize the capture graph before assigning the first saved pose.
                    # Priming after that pose can retain an empty first RGBA buffer in Kit.
                    self.capture.prime()
                self.render_sample(self.inputs.scenes[index])
                rendered += 1
                if policy.should_flush():
                    self.checkpoint(partial=True)
                    policy.flushed()
                print(
                    f"[W{self.args.worker_id}] sample {index} complete ({rendered} new)", flush=True
                )
        except BaseException:
            self.checkpoint(partial=True)
            raise
        path = self.checkpoint(partial=False)
        elapsed = time.monotonic() - started
        print(
            f"[W{self.args.worker_id}] DONE: {rendered} new samples in {elapsed:.2f}s -> {path}",
            flush=True,
        )
        return path
