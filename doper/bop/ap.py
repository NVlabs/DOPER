# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local BOP24 detection AP (MSSD and MSPD) for test sets with public ground truth (T-LESS).

A port of ``bop_toolkit`` ``scripts/eval_bop24_pose.py`` at the pinned revision
(``TOOLKIT_REVISION``) for the two BOP24 errors, run in one process instead of the toolkit's
chain of scripts and JSON files. Every stage keeps the toolkit's order and arithmetic:

1. Estimates (``inout.load_bop_results``): the BOP CSV, at most the 100 best-scoring estimates
   per image (a stable sort by descending score). All estimates of an image must share one
   time (``inout.check_consistent_timings``).
2. Errors (``eval_calc_errors.py`` in detection mode, ``n_top=0``): for every target image the
   objects are re-derived from all of its GT instances (any visibility). Each estimate of such
   an object is compared with every GT instance of the same object: MSSD (infinite when the
   model spheres do not overlap) and MSPD with ``doper.evaluation.bop_metrics``'s ``toolkit_*``
   functions and symmetries (``max_sym_disc_step``). Estimates of objects without GT in the
   image are dropped, as in the toolkit.
3. Scores (``eval_calc_scores.py``): MSSD is divided by the object diameter, MSPD multiplied by
   ``640 / image_width``; per threshold, ``pose_matching.match_poses_scene`` matches estimates
   greedily by score and ``score.calc_pose_detection_scores`` ignores GT below
   ``visib_gt_min`` (and the estimates matched to them) and integrates precision over 101
   recall points (``score.calc_ap``). Both are vendored verbatim in
   ``doper._vendor.bop_toolkit_ap``.
4. Averages (``eval_bop24_pose.py``): per object the mean over the ten thresholds (MSSD 0.05 to
   0.5 of the diameter, MSPD 5 to 50 px), then the mean over objects with at least one valid
   GT; the AP is the mean of the MSSD and MSPD values.

The targets file lists the evaluated images; only its scene and image ids are read (the
detection protocol re-derives the objects from the GT). ``resolve_targets`` picks it: an explicit
file (origin ``user``), else ``test_targets_bop24.json`` (``bop24``), else
``test_targets_bop19.json`` (``derived_from_bop19``): the pinned T-LESS ``tless_base.zip`` ships
only the BOP18 and BOP19 lists, and the BOP19 list names the same 1000 test images, so the AP is
the same as with a BOP24 file over those images.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._vendor.bop_toolkit_ap import pose_matching, score
from ..evaluation.bop_metrics import (
    TOOLKIT_MAX_SYM_DISC_STEP,
    read_ply_vertices,
    toolkit_mspd,
    toolkit_mssd,
    toolkit_symmetry_transforms,
)

TOOLKIT_REVISION = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
TARGETS_NAME = "test_targets_bop24.json"
BOP19_TARGETS_NAME = "test_targets_bop19.json"
TARGETS_ORIGINS = ("user", "bop24", "derived_from_bop19")
# Datasets whose BOP test GT is public, so AP can be computed locally.
PUBLIC_TEST_GT_DATASETS = ("tless",)
# Toolkit constants (eval_bop24_pose.py): thresholds as ``np.arange`` produces them.
MSSD_THRESHOLDS = tuple(float(th) for th in np.arange(0.05, 0.51, 0.05))  # x diameter
MSPD_THRESHOLDS_PX = tuple(float(th) for th in np.arange(5, 51, 5))  # at 640 px width
VISIB_GT_MIN = 0.1
MAX_ESTIMATES_PER_IMAGE = 100
MSPD_REFERENCE_WIDTH_PX = 640.0
# dataset_params.get_split_params: T-LESS test PrimeSense images are 720 x 540. This default is
# specific to the PrimeSense test split; other sensors need their own width (T-LESS test Kinect
# 720, Canon 2560), as does any other dataset. BOP scene_camera.json carries no image size.
TLESS_TEST_IMAGE_WIDTH_PX = 720
TIME_TOLERANCE_S = 0.001
CSV_HEADER = "scene_id,im_id,obj_id,score,R,t,time"
MISSING_TARGETS = (
    "{path}: targets file not found. Local AP reads " + TARGETS_NAME + " or, as the pinned "
    "tless_base.zip ships, " + BOP19_TARGETS_NAME + " (same test images) from the dataset "
    "directory; extract the base part or pass --targets PATH"
)


@dataclass(frozen=True)
class ApResult:
    """BOP24 AP of one result CSV.

    Attributes:
        ap_mssd: Mean MSSD AP over objects with valid targets (``bop24_mAP_mssd``).
        ap_mspd: Mean MSPD AP over the same objects (``bop24_mAP_mspd``).
        ap: Mean of the two (``bop24_mAP``).
        per_object: Object id to ``ap_mssd``, ``ap_mspd``, ``ap`` and ``targets`` (GT instances
            at or above ``visib_gt_min`` in the target images).
        images: Target images scored.
        estimates: Estimates scored (after the per-image cap; on objects with GT in the image).
        average_time_per_image: Mean per-image time of the CSV, or -1 without times.
    """

    ap_mssd: float
    ap_mspd: float
    ap: float
    per_object: dict[int, dict[str, float]]
    images: int
    estimates: int
    average_time_per_image: float

    def to_json(self) -> dict[str, Any]:
        """JSON form with the toolkit revision and thresholds."""
        return {
            "ap_mssd": self.ap_mssd,
            "ap_mspd": self.ap_mspd,
            "ap": self.ap,
            "per_object": {str(obj): values for obj, values in self.per_object.items()},
            "images": self.images,
            "estimates": self.estimates,
            "average_time_per_image": self.average_time_per_image,
            "toolkit_revision": TOOLKIT_REVISION,
            "mssd_thresholds": list(MSSD_THRESHOLDS),
            "mspd_thresholds_px": list(MSPD_THRESHOLDS_PX),
        }


@dataclass
class _Scene:
    """One target scene: its target images in order, GT, GT info and cameras."""

    images: list[int]
    gt: dict[int, list[dict[str, Any]]]
    info: dict[int, list[dict[str, Any]]]
    camera: dict[int, dict[str, Any]]


@dataclass
class _Models:
    """Lazily loaded model points, info and symmetries."""

    root: Path
    max_sym_disc_step: float
    info: dict[int, dict[str, Any]] = field(default_factory=dict)
    points: dict[int, np.ndarray] = field(default_factory=dict)
    symmetries: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def load(self, obj: int) -> tuple[np.ndarray, list[np.ndarray]]:
        if obj not in self.points:
            self.points[obj] = read_ply_vertices(str(self.root / f"obj_{obj:06d}.ply"))
            self.symmetries[obj] = toolkit_symmetry_transforms(
                self.info[obj], max_sym_disc_step=self.max_sym_disc_step
            )
        return self.points[obj], self.symmetries[obj]


def require_public_gt(dataset: str) -> None:
    """Refuse a dataset whose BOP test GT is not public.

    Raises:
        ValueError: ``dataset`` is not in ``PUBLIC_TEST_GT_DATASETS``.
    """
    if dataset not in PUBLIC_TEST_GT_DATASETS:
        raise ValueError(
            f"{dataset}: no public test GT, so its AP comes only from the BOP server; local AP "
            f"supports {list(PUBLIC_TEST_GT_DATASETS)}"
        )


def load_estimates(path: Path) -> list[dict[str, Any]]:
    """Read a BOP19 CSV like ``inout.load_bop_results`` with the per-image detection cap.

    Returns:
        Estimates with ``scene_id``, ``im_id``, ``obj_id``, ``score``, ``R`` (3, 3), ``t``
        (3, 1) and ``time``, grouped by image in first-appearance order, best score first.

    Raises:
        ValueError: A row does not have seven fields.
    """
    results = []
    with Path(path).open() as stream:
        for line_id, line in enumerate(stream, 1):
            if line_id == 1 and CSV_HEADER in line:
                continue
            elems = line.split(",")
            if len(elems) != 7:
                raise ValueError(f"{path}: a line does not have 7 comma-sep. elements: {line}")
            results.append(
                {
                    "scene_id": int(elems[0]),
                    "im_id": int(elems[1]),
                    "obj_id": int(elems[2]),
                    "score": float(elems[3]),
                    "R": np.array(list(map(float, elems[4].split())), np.float64).reshape((3, 3)),
                    "t": np.array(list(map(float, elems[5].split())), np.float64).reshape((3, 1)),
                    "time": float(elems[6]),
                }
            )
    by_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_image[(result["scene_id"], result["im_id"])].append(result)
    capped = []
    for image in by_image.values():
        capped.extend(
            sorted(image, key=lambda x: x["score"], reverse=True)[:MAX_ESTIMATES_PER_IMAGE]
        )
    return capped


def average_time_per_image(estimates: Iterable[Mapping[str, Any]]) -> float:
    """``inout.check_consistent_timings`` plus the toolkit's mean time per image.

    Returns:
        The mean per-image time; -1 when any estimate has no time (negative) or there is no
        estimate (the toolkit returns NaN for an empty CSV).

    Raises:
        ValueError: Two estimates of one image differ in time by more than 1 ms.
    """
    times: dict[tuple[int, int], float] = {}
    available = True
    for est in estimates:
        key = (est["scene_id"], est["im_id"])
        if est["time"] < 0:
            available = False
            continue
        if key in times and abs(times[key] - est["time"]) > TIME_TOLERANCE_S:
            raise ValueError(
                f"the running time for scene {key[0]} and image {key[1]} is not the same for "
                "all estimates"
            )
        times.setdefault(key, est["time"])
    if not available or not times:
        return -1.0
    return float(np.mean(list(times.values())))


def resolve_targets(dataset_dir: Path, explicit: Path | None = None) -> tuple[Path, str]:
    """Pick the targets file and its origin (one of ``TARGETS_ORIGINS``).

    Args:
        dataset_dir: Directory holding the BOP target lists (``<root>/data``).
        explicit: A user-supplied targets file; it must exist.

    Returns:
        ``(path, origin)``: ``explicit`` (``user``), else ``test_targets_bop24.json``
        (``bop24``), else ``test_targets_bop19.json`` (``derived_from_bop19``).

    Raises:
        FileNotFoundError: The explicit file is missing, or neither list exists.
    """
    if explicit is not None:
        if not Path(explicit).is_file():
            raise FileNotFoundError(MISSING_TARGETS.format(path=explicit))
        return Path(explicit), "user"
    for name, origin in ((TARGETS_NAME, "bop24"), (BOP19_TARGETS_NAME, "derived_from_bop19")):
        if (Path(dataset_dir) / name).is_file():
            return Path(dataset_dir) / name, origin
    raise FileNotFoundError(MISSING_TARGETS.format(path=Path(dataset_dir) / TARGETS_NAME))


def load_targets(path: Path) -> dict[int, list[int]]:
    """Target images by scene, in first-appearance order (only scene and image ids are read).

    Raises:
        FileNotFoundError: ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(MISSING_TARGETS.format(path=path))
    organized: dict[int, dict[int, None]] = {}
    for target in json.loads(path.read_text()):
        organized.setdefault(target["scene_id"], {})[target["im_id"]] = None
    return {scene: list(images) for scene, images in organized.items()}


def _int_keys(mapping: Mapping[str, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in mapping.items()}


def _load_scene(gt_root: Path, scene_id: int, images: list[int]) -> _Scene:
    directory = Path(gt_root) / f"{scene_id:06d}"
    names = ("scene_gt.json", "scene_gt_info.json", "scene_camera.json")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"{directory}: {missing} missing; local AP needs public test GT (BOP test sets "
            "without public GT are scored by the BOP server)"
        )
    gt, info, camera = (_int_keys(json.loads((directory / name).read_text())) for name in names)
    for poses in gt.values():
        for pose in poses:
            pose["cam_R_m2c"] = np.array(pose["cam_R_m2c"], np.float64).reshape((3, 3))
            pose["cam_t_m2c"] = np.array(pose["cam_t_m2c"], np.float64).reshape((3, 1))
    for cam in camera.values():
        cam["cam_K"] = np.array(cam["cam_K"], np.float64).reshape((3, 3))
    return _Scene(images=images, gt=gt, info=info, camera=camera)


def _image_errors(
    scene_id: int,
    im_id: int,
    scene: _Scene,
    estimates: Mapping[int, list[dict[str, Any]]],
    models: _Models,
) -> list[dict[str, Any]]:
    """``eval_calc_errors.py`` for one image: raw MSSD and MSPD per (estimate, GT) pair."""
    im_gt = scene.gt[im_id]
    camera_matrix = scene.camera[im_id]["cam_K"]
    errors = []
    for obj_id in dict.fromkeys(gt["obj_id"] for gt in im_gt):
        ranked = sorted(
            enumerate(estimates.get(obj_id, [])), key=lambda x: x[1]["score"], reverse=True
        )
        for est_id, est in ranked:
            errors.append(
                _estimate_errors(
                    est, est_id, obj_id, im_gt, scene.info[im_id], camera_matrix, models
                )
                | {"im_id": im_id, "scene_id": scene_id}
            )
    return errors


def _estimate_errors(
    est: Mapping[str, Any],
    est_id: int,
    obj_id: int,
    im_gt: Sequence[Mapping[str, Any]],
    im_info: Sequence[Mapping[str, Any]],
    camera_matrix: np.ndarray,
    models: _Models,
) -> dict[str, Any]:
    pts, syms = models.load(obj_id)
    diameter = models.info[obj_id]["diameter"]
    mssd, mspd, visib = {}, {}, {}
    for gt_id, gt in enumerate(im_gt):
        if gt["obj_id"] != obj_id:
            continue
        r_gt, t_gt = gt["cam_R_m2c"], gt["cam_t_m2c"]
        overlap = np.linalg.norm(est["t"] - t_gt) < diameter
        mssd[gt_id] = (
            [toolkit_mssd(est["R"], est["t"], r_gt, t_gt, pts, syms)] if overlap else [float("inf")]
        )
        mspd[gt_id] = [toolkit_mspd(est["R"], est["t"], r_gt, t_gt, camera_matrix, pts, syms)]
        visib[gt_id] = im_info[gt_id]["visib_fract"]
    return {
        "obj_id": obj_id,
        "est_id": est_id,
        "score": est["score"],
        "mssd": mssd,
        "mspd": mspd,
        "gt_visib_fracts": visib,
    }


def _normalized(
    errors: Sequence[Mapping[str, Any]], kind: str, scale: Mapping[int, float], divide: bool
) -> list[dict[str, Any]]:
    """Fresh per-estimate records with ``errors`` of ``kind`` normalized as the scorer does."""
    records = []
    for err in errors:
        factor = scale[err["obj_id"]] if divide else scale[err["scene_id"]]
        values = {
            gt_id: [e / factor for e in errs] if divide else [factor * e for e in errs]
            for gt_id, errs in err[kind].items()
        }
        records.append(
            {
                "im_id": err["im_id"],
                "obj_id": err["obj_id"],
                "est_id": err["est_id"],
                "score": err["score"],
                "errors": values,
                "scene_id": err["scene_id"],
                "gt_visib_fracts": err["gt_visib_fracts"],
            }
        )
    return records


def _threshold_scores(
    scenes: Mapping[int, _Scene],
    errors: Mapping[int, list[dict[str, Any]]],
    obj_ids: Sequence[int],
    threshold: float,
    visib_gt_min: float,
) -> dict[str, dict[int, Any]]:
    """``eval_calc_scores.py`` (detection) for one error threshold."""
    matches, estimates = [], []
    for scene_id, scene in scenes.items():
        scene_gt = {im: scene.gt[im] for im in scene.images}
        scene_info = {im: scene.info[im] for im in scene.images}
        valid = {im: [True] * len(scene.gt[im]) for im in scene.images}
        scene_errs = [dict(err) for err in errors[scene_id]]
        matches += pose_matching.match_poses_scene(
            scene_id, scene_gt, scene_info, valid, scene_errs, [threshold], 0
        )
        estimates += scene_errs
    # Objects without a valid GT divide 0 by 0 in the toolkit; their AP is never averaged.
    with np.errstate(divide="ignore", invalid="ignore"):
        return score.calc_pose_detection_scores(
            list(scenes), obj_ids, matches, estimates, visib_gt_min=visib_gt_min, do_print=False
        )


def _error_type_ap(
    scenes: Mapping[int, _Scene],
    errors: Mapping[int, list[dict[str, Any]]],
    obj_ids: Sequence[int],
    thresholds: Sequence[float],
    visib_gt_min: float,
) -> tuple[float, dict[int, float], dict[int, int]]:
    """``eval_bop24_pose.py``'s threshold and object means for one error type."""
    per_threshold = [
        _threshold_scores(scenes, errors, obj_ids, th, visib_gt_min) for th in thresholds
    ]
    targets = per_threshold[-1]["num_targets_per_object"]
    per_object = {
        obj: float(np.mean([s["scores"][obj] for s in per_threshold]))
        for obj in obj_ids
        if targets[obj] > 0
    }
    if not per_object:
        raise ValueError(f"no GT instance with visibility >= {visib_gt_min} in the target images")
    return float(np.mean(list(per_object.values()))), per_object, targets


def bop24_ap(
    csv: Path,
    gt_root: Path,
    targets: Path,
    models: Path,
    *,
    visib_gt_min: float = VISIB_GT_MIN,
    max_sym_disc_step: float = TOOLKIT_MAX_SYM_DISC_STEP,
    image_width: int = TLESS_TEST_IMAGE_WIDTH_PX,
) -> ApResult:
    """BOP24 AP (MSSD, MSPD and their mean) of a result CSV, equal to the pinned toolkit's.

    Args:
        csv: BOP19-format result CSV (``scene_id,im_id,obj_id,score,R,t,time``, mm).
        gt_root: Test split directory with ``<scene:06d>/scene_{gt,gt_info,camera}.json``.
        targets: Targets file (BOP24 or BOP19 format; only the scene and image ids of the
            evaluated images are read). See ``resolve_targets``.
        models: Evaluation models directory: ``models_info.json`` and ``obj_XXXXXX.ply`` (mm).
            Its ``models_info.json`` objects are the objects scored.
        visib_gt_min: GT instances visible below this fraction are ignored with their matches.
        max_sym_disc_step: Continuous-symmetry discretization (``misc.
            get_symmetry_transformations``).
        image_width: Test image width in px; MSPD is scaled by ``640 / image_width``. The
            default is the T-LESS PrimeSense test width (``TLESS_TEST_IMAGE_WIDTH_PX``).

    Returns:
        The AP over the objects with at least one valid GT instance, and per-object values.

    Raises:
        FileNotFoundError: The targets file is missing.
        ValueError: A target scene has no public GT, a CSV row is malformed, an image has
            inconsistent times, or no GT instance is valid.
    """
    target_images = load_targets(targets)
    all_estimates = load_estimates(csv)
    mean_time = average_time_per_image(all_estimates)
    model_set = _Models(root=Path(models), max_sym_disc_step=max_sym_disc_step)
    info = json.loads((Path(models) / "models_info.json").read_text())
    model_set.info = _int_keys(info)
    obj_ids = sorted(model_set.info)
    by_image: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}
    for est in all_estimates:
        image = by_image.setdefault((est["scene_id"], est["im_id"]), {})
        image.setdefault(est["obj_id"], []).append(est)
    scenes, errors = {}, {}
    for scene_id, images in target_images.items():
        scenes[scene_id] = _load_scene(gt_root, scene_id, images)
        errors[scene_id] = [
            err
            for im_id in images
            for err in _image_errors(
                scene_id, im_id, scenes[scene_id], by_image.get((scene_id, im_id), {}), model_set
            )
        ]
    diameters = {obj: float(model_set.info[obj]["diameter"]) for obj in obj_ids}
    widths = {scene_id: MSPD_REFERENCE_WIDTH_PX / image_width for scene_id in scenes}
    mssd_errors = {s: _normalized(e, "mssd", diameters, True) for s, e in errors.items()}
    mspd_errors = {s: _normalized(e, "mspd", widths, False) for s, e in errors.items()}
    ap_mssd, mssd_objects, targets_count = _error_type_ap(
        scenes, mssd_errors, obj_ids, MSSD_THRESHOLDS, visib_gt_min
    )
    ap_mspd, mspd_objects, _ = _error_type_ap(
        scenes, mspd_errors, obj_ids, MSPD_THRESHOLDS_PX, visib_gt_min
    )
    per_object = {
        obj: {
            "ap_mssd": mssd_objects[obj],
            "ap_mspd": mspd_objects[obj],
            "ap": float(np.mean([mssd_objects[obj], mspd_objects[obj]])),
            "targets": targets_count[obj],
        }
        for obj in mssd_objects
    }
    return ApResult(
        ap_mssd=ap_mssd,
        ap_mspd=ap_mspd,
        ap=float(np.mean([ap_mssd, ap_mspd])),
        per_object=per_object,
        images=sum(len(images) for images in target_images.values()),
        estimates=sum(len(e) for e in errors.values()),
        average_time_per_image=mean_time,
    )
