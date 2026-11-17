# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four BOP test variants: single-view, heatmap-refined, multi-view and both.

The variants are a post-pass over each bank model's raw detection journal
(``doper.bop.export.cached_raw_journal``): the same forward pass that the single-view export
uses also stored every kept detection's keypoints, PnP status and raw keypoint heatmaps. Per
model and reference test image, in detection order, every detection with a finite single-view
pose yields one row per variant:

* ``singleview``: the PnP pose.
* ``singleheatmap``: the PnP pose refined against the reference heatmap (always attempted).
* ``multiview``: the fused pose (``doper.bop.multiview.fuse_reference``); the PnP pose when no
  support matched or the joint fit was rejected.
* ``multiheatmap``: only when at least one support detection matched, the fused pose refined
  against the heatmaps of the reference and every matched support view. A matched detection
  whose joint fit was rejected starts from its original pose and still refines over every
  matched view. An unmatched detection keeps the unrefined pose (the ``multiview`` row), never
  the ``singleheatmap`` one.

Every variant keeps the same detections and scores and drops a row whose pose fails the BOP rules
(``doper.bop.export.valid_pose``: in practice ``t_z <= 0``). On a dataset without a multi-view
policy (HANDAL) a detection whose single-view pose is invalid is not refined. Rotations pass through
a ``cv2`` Rodrigues round trip (the fused rotation is converted to a rotation vector and back),
which the bit-exact rows depend on. Variant CSVs carry no runtime (``time = -1``).

The reference implementation is the per-image four-arm composition of the evaluated runners
(``evaluate_frame`` of the T-LESS comparison; the paper's heatmap cells are their Powell
submissions). The later Numba re-evaluation runner differed in three places, which the release
does not reproduce: its T-LESS matched ``multiheatmap`` started from the stored ``multiview``
row's rotation converted to a rotation vector once more (one extra Rodrigues round trip of an
already round-tripped rotation); its HANDAL ``singleheatmap`` started from the ``singleview``
row's rotation converted back to a rotation vector instead of the PnP rotation vector (one
extra round trip); and it always used the parallel Numba preparation (four threads), whereas
the release default is the NumPy preparation (``--preparation parallel`` selects the other;
the two differ by at most ~1e-12 per map).
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from ..recipes.schema import MultiviewSpec, Recipe
from ..refinement import (
    HeatmapObservation,
    LbfgsbAudit,
    PowellAudit,
    RefinementAudit,
    RefineSettings,
    refine_pose,
)
from . import export
from .layout import (
    CHECKPOINT_PATH,
    HEATMAP_VARIANTS,
    KEYPOINTS_PATH,
    MULTIVIEW_VARIANTS,
    VARIANTS,
    Variant,
)
from .multiview import ImageKey, fuse_reference, world_to_camera
from .sources import canonical_sha256, json_roundtrip, publish_json, sha256_file

MANIFEST_NAME = "variants_manifest.json"
OBJECTS_DIR_SUFFIX = ".objects"
HeatmapLoader = Callable[[], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, eq=False)
class JournalDetection:
    """One kept detection of a raw journal (``doper.multiview.KeypointDetection``).

    Attributes:
        index: Its index in the image's kept detections (score order).
        score: Detector score.
        keypoints: ``(K, 2)`` keypoints in image pixels.
        keypoint_scores: ``(K,)`` keypoint confidences.
        rvec: ``(3, 1)`` single-view rotation vector; None without a finite PnP pose.
        tvec: ``(3, 1)`` single-view translation, millimeters; None without a finite pose.
        heatmap: Loads ``(logits, crop_to_image)`` of the same forward pass; None when the
            model kept no heatmaps.
    """

    index: int
    score: float
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    heatmap: HeatmapLoader | None = None

    @classmethod
    def from_raw(
        cls, record: Mapping[str, Any], heatmap: HeatmapLoader | None = None
    ) -> JournalDetection:
        """The journal detection of one raw journal detection (``export.raw_detection``)."""
        posed = record["rvec"] is not None and record["tvec"] is not None
        return cls(
            index=record["index"],
            score=record["score"],
            keypoints=np.asarray(record["keypoints"], dtype=np.float32),
            keypoint_scores=np.asarray(record["keypoint_scores"], dtype=np.float32),
            rvec=np.asarray(record["rvec"], dtype=np.float64) if posed else None,
            tvec=np.asarray(record["tvec"], dtype=np.float64) if posed else None,
            heatmap=heatmap,
        )


@dataclass(frozen=True, eq=False)
class View:
    """One image of the journal: its camera record and its kept detections.

    Attributes:
        camera: ``cam_K`` and, for multi-view, the w2c-named extrinsics.
        detections: Kept detections; ``detections[j].index == j``.
    """

    camera: Mapping[str, Any]
    detections: tuple[JournalDetection, ...]


@dataclass(frozen=True)
class VariantSettings:
    """How the variants are computed.

    Attributes:
        refine: Heatmap optimizer and preparation.
        crop_size: Crop side of the model (``cfg.model.crop_size``).
        beta: Heatmap softmax temperature (``cfg.loss.keypoint_heatmap_beta``).
        variants: Variants to compute (the others are not computed nor returned).
    """

    refine: RefineSettings = field(default_factory=RefineSettings)
    crop_size: int = 512
    beta: float = 1.0
    variants: tuple[Variant, ...] = VARIANTS

    def __post_init__(self) -> None:
        """Reject unknown variants."""
        unknown = [v for v in self.variants if v not in VARIANTS]
        if unknown or not self.variants:
            raise ValueError(f"unknown variants {unknown}; choose from {VARIANTS}")


@dataclass(frozen=True)
class Row:
    """One CSV row of one variant (the object id is the caller's).

    Attributes:
        scene_id: Scene of the reference image.
        im_id: Reference image id.
        detection_index: Index of the reference detection.
        score: Its detector score.
        R: Row-major rotation (9 values).
        t: Translation, millimeters (3 values).
    """

    scene_id: int
    im_id: int
    detection_index: int
    score: float
    R: tuple[float, ...]  # noqa: N815
    t: tuple[float, ...]


def _posed(det: JournalDetection) -> tuple[np.ndarray, np.ndarray] | None:
    if det.rvec is None or det.tvec is None:
        return None
    rvec = np.asarray(det.rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(det.tvec, dtype=np.float64).reshape(3, 1)
    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return None
    return rvec, tvec


def _row(key: ImageKey, det: JournalDetection, rvec: np.ndarray, tvec: Any) -> Row:  # noqa: ANN401
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    return Row(
        scene_id=key.scene_id,
        im_id=key.im_id,
        detection_index=det.index,
        score=det.score,
        R=tuple(float(v) for v in rotation.ravel()),
        t=tuple(float(v) for v in np.asarray(tvec, dtype=np.float64).ravel()),
    )


def _valid(row: Row) -> bool:
    return export.valid_pose({"score": row.score, "R": row.R, "t": row.t})


def _has_extrinsics(camera: Mapping[str, Any]) -> bool:
    return "cam_R_w2c" in camera and "cam_t_w2c" in camera


class _Observer:
    """Builds heatmap observations of one reference image's group."""

    def __init__(
        self,
        journal: Mapping[ImageKey, View],
        reference: ImageKey,
        spec: MultiviewSpec | None,
    ) -> None:
        self.journal, self.reference, self.spec = journal, reference, spec
        camera = journal[reference].camera
        self.reference_inverse = (
            np.linalg.inv(world_to_camera(camera, spec.convention))
            if spec is not None and _has_extrinsics(camera)
            else None
        )

    def relative(self, key: ImageKey) -> np.ndarray:
        """Reference-to-``key`` camera transform (mm)."""
        if self.reference_inverse is None or self.spec is None:
            return np.eye(4)
        camera = self.journal[key].camera
        return world_to_camera(camera, self.spec.convention) @ self.reference_inverse

    def __call__(self, key: ImageKey, index: int) -> HeatmapObservation:
        det = self.journal[key].detections[index]
        if det.heatmap is None:
            raise ValueError(
                f"image {tuple(key)} detection {index} has no keypoint heatmap: the heatmap "
                "variants need a heatmap keypoint head and a raw detection journal"
            )
        logits, crop_to_image = det.heatmap()
        return HeatmapObservation(
            logits=logits,
            crop_to_image=crop_to_image,
            K=np.array(self.journal[key].camera["cam_K"]).reshape(3, 3),
            relative_mm=self.relative(key),
        )


def _check_journal(
    journal: Mapping[ImageKey, View], groups: Mapping[ImageKey, Sequence[ImageKey]]
) -> None:
    for reference, group in groups.items():
        if not group or group[0] != reference:
            raise ValueError(f"group of {tuple(reference)} must start with the reference")
        missing = [tuple(k) for k in group if k not in journal]
        if missing:
            raise ValueError(
                f"images {missing[:3]} of the group of {tuple(reference)} have no journal"
            )
    for key, view in journal.items():
        if [d.index for d in view.detections] != list(range(len(view.detections))):
            raise ValueError(f"image {tuple(key)}: detections must be indexed in order")


def audit_optimizer(audit: RefinementAudit) -> str:
    """The optimizer that produced a refinement audit (``lbfgsb`` or ``powell``).

    Raises:
        TypeError: For an object that is neither audit type.
    """
    if isinstance(audit, LbfgsbAudit):
        return "lbfgsb"
    if isinstance(audit, PowellAudit):
        return "powell"
    raise TypeError(f"not a refinement audit: {type(audit).__name__}")


def _matched(pose: Any) -> list[list[Any]]:  # noqa: ANN401
    return [[k.scene_id, k.im_id, k.stream, j] for k, j in pose.matched]


def compose_variants(
    journal: Mapping[ImageKey, View],
    groups: Mapping[ImageKey, Sequence[ImageKey]],
    keypoints_mm: np.ndarray,
    settings: VariantSettings,
    spec: MultiviewSpec | None,
    *,
    audit: list[dict[str, Any]] | None = None,
) -> dict[Variant, list[Row]]:
    """Compose one object model's test variants (module docstring).

    Args:
        journal: Every reference and support image's view (camera and kept detections).
        groups: Reference image to its group, the reference first (``support_groups``; a
            single-view dataset maps each reference to itself).
        keypoints_mm: ``(K, 3)`` object keypoints, millimeters, in heatmap channel order.
        settings: Variants, optimizer and heatmap geometry.
        spec: The dataset's multi-view policy; None for a dataset without multi-view.
        audit: When given, one entry per composed detection is appended: ``matched``
            supports (``[scene, im, stream, index]``), ``fused_accepted`` and the heatmap
            audits (``single_heatmap``, ``multi_heatmap``; None when not run) and the
            ``optimizer`` that ran them (``audit_optimizer``; None without refinement).

    Returns:
        Rows per requested variant, references in sorted order and detections in score order.

    Raises:
        ValueError: For multi-view variants without ``spec``, a group image missing from the
            journal, or a heatmap variant over a detection without heatmaps.
    """
    names = settings.variants
    fusion = any(v in MULTIVIEW_VARIANTS for v in names)
    if fusion and spec is None:
        raise ValueError("the multi-view variants need the recipe's multi-view policy")
    _check_journal(journal, groups)
    keypoints = np.asarray(keypoints_mm, dtype=np.float64)
    rows: dict[Variant, list[Row]] = {name: [] for name in names}

    def refine(
        rvec: np.ndarray, tvec: np.ndarray, observations: list[HeatmapObservation]
    ) -> tuple[np.ndarray, np.ndarray, RefinementAudit]:
        return refine_pose(
            rvec, tvec, keypoints, observations, settings.crop_size, settings.beta,
            settings.refine,
        )  # fmt: skip

    for reference in sorted(groups):
        group = tuple(groups[reference])
        view = journal[reference]
        observe = _Observer(journal, reference, spec)
        fused = {}
        if fusion:
            assert spec is not None
            poses = fuse_reference(
                view.detections,
                {k: journal[k].detections for k in group[1:]},
                {k: journal[k].camera for k in group},
                keypoints,
                spec,
            )
            fused = {pose.reference_index: pose for pose in poses}
        for det in view.detections:
            initial = _posed(det)
            if initial is None:
                continue
            rvec, tvec = initial
            single = _row(reference, det, rvec, tvec)
            if spec is None and not _valid(single):
                continue
            entry: dict[str, Any] = {
                "scene_id": reference.scene_id,
                "im_id": reference.im_id,
                "stream": reference.stream,
                "detection_index": det.index,
                "matched": None,
                "fused_accepted": None,
                "single_heatmap": None,
                "multi_heatmap": None,
                "optimizer": None,
            }
            out: dict[Variant, Row] = {"singleview": single}
            if "singleheatmap" in names:
                refined_r, refined_t, result = refine(rvec, tvec, [observe(reference, det.index)])
                out["singleheatmap"] = _row(reference, det, refined_r, refined_t)
                entry["single_heatmap"] = result.to_dict()
                entry["optimizer"] = audit_optimizer(result)
            if fusion:
                pose = fused[det.index]
                fused_r = cv2.Rodrigues(pose.R)[0]
                fused_t = np.asarray(pose.t, dtype=np.float64).reshape(3, 1)
                out["multiview"] = _row(reference, det, fused_r, fused_t)
                out["multiheatmap"] = out["multiview"]
                entry["matched"], entry["fused_accepted"] = _matched(pose), pose.accepted
                if "multiheatmap" in names and pose.matched:
                    views = [observe(reference, det.index)]
                    views += [observe(key, j) for key, j in pose.matched]
                    refined_r, refined_t, result = refine(fused_r, fused_t, views)
                    out["multiheatmap"] = _row(reference, det, refined_r, refined_t)
                    entry["multi_heatmap"] = result.to_dict()
                    entry["optimizer"] = audit_optimizer(result)
            for name in names:
                if _valid(out[name]):
                    rows[name].append(out[name])
            if audit is not None:
                audit.append(entry)
    return rows


def journal_views(
    journal: export.RawJournal,
    cameras: Mapping[ImageKey, Mapping[str, Any]],
    keys: Mapping[tuple[int, int], ImageKey] | None = None,
) -> dict[ImageKey, View]:
    """The views of a raw journal; heatmap loaders (hash-checked) only if it has its store.

    Args:
        journal: One model's raw journal over one image population.
        cameras: Camera record of every image (``cam_K``; extrinsics for multi-view).
        keys: Journal key ``(scene_id, im_id)`` to image key (HOT3D streams); default
            ``ImageKey(scene_id, im_id)``.

    Returns:
        Image key to view.

    Raises:
        ValueError: If an image has no camera record.
    """
    views = {}
    for key, record in journal.records.items():
        image = keys[key] if keys is not None else ImageKey(*key)
        if image not in cameras:
            raise ValueError(f"image {tuple(image)} has no camera record")
        heatmaps = journal.has_heatmaps(key)
        detections = tuple(
            JournalDetection.from_raw(
                det, journal.heatmap_loader(key, det["index"]) if heatmaps else None
            )
            for det in record["detections"]
        )
        views[image] = View(camera=cameras[image], detections=detections)
    return views


def groups_sha256(groups: Mapping[ImageKey, Sequence[ImageKey]]) -> str:
    """Identity of the support groups (reference and group, sorted by reference)."""
    return canonical_sha256([[list(ref), [list(k) for k in groups[ref]]] for ref in sorted(groups)])


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    """Versions of the numerical libraries the variants depend on (None: not installed)."""
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "numba": _version("numba"),
    }


def variant_manifest(
    settings: VariantSettings,
    groups: Mapping[ImageKey, Sequence[ImageKey]],
    *,
    variant: Variant,
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """The ``variants_manifest.json`` record of one variant export.

    Records the optimizer and preparation that actually ran (a CSV must never be described
    with another optimizer than the one that produced it), the heatmap geometry, the library
    versions and the support-group hash.

    Args:
        settings: The settings the variant ran with.
        groups: The support groups.
        variant: The exported variant.
        **extra: Further fields (policy, CSV, per-object identities).

    Returns:
        The manifest mapping.
    """
    return {
        "variant": variant,
        "optimizer": settings.refine.optimizer,
        "preparation": settings.refine.preparation,
        "threads": settings.refine.threads,
        "crop_size": settings.crop_size,
        "beta": settings.beta,
        "groups_sha256": groups_sha256(groups),
        "versions": library_versions(),
        **extra,
    }


def csv_method(recipe: Recipe, policy: str, variant: str) -> str:
    """The CSV method token of a variant export: ``<method><policy><variant>``.

    The built-in policies use ``<test.method><policy>``; a fixed policy its ``csv_method``.

    Raises:
        ValueError: For an unknown variant or policy.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if policy in ("base", "selected"):
        return f"{recipe.test.method}{policy}{variant}"
    from .banks import recipe_policy

    return f"{recipe_policy(recipe, policy).csv_method}{variant}"


@dataclass(frozen=True)
class SupportPopulation:
    """One support stream's images (HOT3D), scanned by every bank model.

    Attributes:
        stream: Stream id.
        frames: Its images (staged, rectified; scanned with the pinhole camera).
        population_sha256: Identity of the image list.
    """

    stream: str
    frames: tuple[export.TestFrame, ...]
    population_sha256: str


@dataclass(frozen=True)
class VariantInputs:
    """The images and cameras of a variant export.

    Attributes:
        groups: Reference image to its group (reference first).
        cameras: Camera record of every reference and support image.
        reference_keys: Test journal key ``(scene_id, im_id)`` to its image key.
        supports: Support streams scanned in addition to the test population.
        frames: The test population as ``export.load_test_frames`` loaded it, when the inputs
            were built from it (``run_variant`` and deployment timing then reuse it).
        population: Its identity hash, with ``frames``.
    """

    groups: Mapping[ImageKey, tuple[ImageKey, ...]]
    cameras: Mapping[ImageKey, Mapping[str, Any]]
    reference_keys: Mapping[tuple[int, int], ImageKey]
    supports: tuple[SupportPopulation, ...] = ()
    frames: tuple[export.TestFrame, ...] | None = None
    population: str | None = None


@dataclass(frozen=True)
class VariantSummary:
    """Coverage of one variant export.

    Attributes:
        csv_path: Written CSV.
        csv_sha256: SHA-256 of the CSV.
        variant: The variant.
        images: Test images.
        objects: Objects of the bank.
        missing_objects: Dataset objects absent from a partial bank.
        pose_rows: Rows written.
        detections: Composed detections (single-view poses that are finite).
        single_heatmap_accepted: Accepted single-view heatmap refinements.
        multiview_matched: Detections with at least one matched support.
        multiview_accepted: Accepted joint fits.
        multi_heatmap_accepted: Accepted multi-view heatmap refinements.
        resumed_objects: Objects taken from their saved result.
        journal_generations: Object id to the generations of its test journal and then each
            support stream's journal (the forward passes its rows came from).
    """

    csv_path: Path
    csv_sha256: str
    variant: str
    images: int
    objects: tuple[int, ...]
    missing_objects: tuple[int, ...]
    pose_rows: int
    detections: int
    single_heatmap_accepted: int
    multiview_matched: int
    multiview_accepted: int
    multi_heatmap_accepted: int
    resumed_objects: int
    journal_generations: Mapping[str, list[str]] = field(default_factory=dict)


def _counts(audit: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "detections": len(audit),
        "single_heatmap_accepted": sum(
            bool(a["single_heatmap"] and a["single_heatmap"]["accepted"]) for a in audit
        ),
        "multiview_matched": sum(bool(a["matched"]) for a in audit),
        "multiview_accepted": sum(bool(a["fused_accepted"]) for a in audit),
        "multi_heatmap_accepted": sum(
            bool(a["multi_heatmap"] and a["multi_heatmap"]["accepted"]) for a in audit
        ),
    }


def _model_settings(
    model: Mapping[str, Any] | None, refine: RefineSettings, variant: Variant, run: Path
) -> VariantSettings:
    heatmap_head = model is not None and model.get("keypoint_head") == "heatmap"
    if variant in HEATMAP_VARIANTS and not heatmap_head:
        raise ValueError(f"{run}: the heatmap variants need a model with a heatmap keypoint head")
    if model is None:  # a stand-in model: the default geometry
        return VariantSettings(refine=refine, variants=(variant,))
    return VariantSettings(
        refine=refine,
        crop_size=int(model["crop_size"]),
        beta=float(model["keypoint_heatmap_beta"]),
        variants=(variant,),
    )


def cameras_sha256(cameras: Mapping[ImageKey, Mapping[str, Any]]) -> str:
    """Identity of the camera records (intrinsics and extrinsics) a variant export used."""
    return canonical_sha256([[list(k), dict(cameras[k])] for k in sorted(cameras)])


@dataclass(frozen=True)
class _ExportContext:
    """What every object of one variant export shares."""

    variant: Variant
    frames: Sequence[export.TestFrame]
    population: str
    inputs: VariantInputs
    spec: MultiviewSpec | None
    refine: RefineSettings
    cache: export.JournalCache
    camera: export.CameraName
    device: str
    results: Path
    groups_sha256: str
    cameras_sha256: str


def _object_journals(
    run: Path, obj: int, hashes: tuple[str, str], context: _ExportContext
) -> tuple[list[tuple[export.RawJournal, Mapping[tuple[int, int], ImageKey] | None]], list[str]]:
    """Every journal of one model: the test journal, then each support stream's journal.

    Heatmaps are opened (and their store filled) only for the heatmap variants.

    Returns:
        Each journal with its journal-to-image keys (None for the test journal's own
        references), and each journal's key digest.
    """
    heatmaps = context.variant in HEATMAP_VARIANTS
    key = export.journal_key(*hashes, context.population, camera=context.camera)
    test = export.cached_raw_journal(
        run, context.frames, context.population, context.cache, camera=context.camera,
        device=context.device, obj_id=obj, key=key, heatmaps=heatmaps,
    )  # fmt: skip
    journals: list[tuple[export.RawJournal, Mapping[tuple[int, int], ImageKey] | None]] = [
        (test, context.inputs.reference_keys)
    ]
    digests = [key.digest()]
    for support in context.inputs.supports:
        key = export.journal_key(*hashes, support.population_sha256, camera="pinhole")
        journal = export.cached_raw_journal(
            run, support.frames, support.population_sha256, context.cache, camera="pinhole",
            device=context.device, obj_id=obj, key=key, heatmaps=heatmaps,
        )  # fmt: skip
        keys = {f.key: ImageKey(f.scene_id, f.im_id, support.stream) for f in support.frames}
        journals.append((journal, keys))
        digests.append(key.digest())
    return journals, digests


def _compose_object(run: Path, obj: int, context: _ExportContext) -> tuple[dict[str, Any], bool]:
    """One object's rows and audit counts, resumed from its saved result when identical.

    The journals' generations (the forward pass each holds; a heatmap rescan that differs
    replaces a journal with a new one) enter the identity; their views are built only when the
    saved result does not match it.
    """
    hashes = (sha256_file(run / CHECKPOINT_PATH), sha256_file(run / KEYPOINTS_PATH))
    journals, digests = _object_journals(run, obj, hashes, context)
    test = journals[0][0]
    variant, inputs = context.variant, context.inputs
    settings = _model_settings(test.model, context.refine, variant, run)
    identity = {
        "object_id": obj,
        "variant": variant,
        "checkpoint_sha256": hashes[0],
        "keypoints_sha256": hashes[1],
        "journals": digests,
        "journal_generations": [journal.generation for journal, _ in journals],
        "groups_sha256": context.groups_sha256,
        "cameras_sha256": context.cameras_sha256,
        "multiview": None if context.spec is None else repr(context.spec),
        "settings": variant_manifest(settings, inputs.groups, variant=variant),
    }
    path = context.results / f"obj_{obj:06d}.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["identity"] == json_roundtrip(identity):
            return saved, True
    views: dict[ImageKey, View] = {}
    for journal, keys in journals:
        views.update(journal_views(journal, inputs.cameras, keys))
    audit: list[dict[str, Any]] = []
    rows = compose_variants(
        views,
        inputs.groups,
        export.keypoints_mm(run / KEYPOINTS_PATH),
        settings,
        context.spec,
        audit=audit,
    )[variant]
    result = {
        "identity": identity,
        "rows": [vars(row) | {"R": list(row.R), "t": list(row.t)} for row in rows],
        "counts": _counts(audit),
        "audit": audit,
    }
    publish_json(path, result)
    return json_roundtrip(result), False


def run_variant(
    bank: Mapping[int, Path],
    test_root: Path,
    out_csv: Path,
    *,
    variant: Variant,
    inputs: VariantInputs,
    spec: MultiviewSpec | None,
    refine: RefineSettings,
    cache: export.JournalCache,
    camera: export.CameraName = "pinhole",
    objects: Sequence[int] | None = None,
    allow_partial_bank: bool = False,
    expected_population: export.TestPopulation | None = None,
    device: str = "cuda:0",
    manifest_extra: Mapping[str, Any] | None = None,
) -> VariantSummary:
    """Export one test variant of a bank to a BOP CSV (untimed, ``time = -1``).

    Every model's raw journal over the test population (and the support streams) comes from
    the cache, so the forward passes are shared with the single-view export. Each object's
    composed rows are saved beside the CSV (``<csv stem>.objects/obj_XXXXXX.json``) and reused
    by a rerun with the same identity. ``variants_manifest.json`` is written after the CSV.

    Args:
        bank: Object id to run directory.
        test_root: BOP test split directory or test manifest.
        out_csv: Output path ``<method>_<dataset>-test.csv``.
        variant: The variant to export.
        inputs: Groups, cameras and support populations; its test frames, when present, are
            those of ``test_root`` and ``camera`` (not reloaded).
        spec: The dataset's multi-view policy (required by the multi-view variants).
        refine: Heatmap optimizer and preparation.
        cache: The per-model journal cache.
        camera: Test camera (``pinhole`` or ``native``).
        objects: The dataset's object ids (default: all).
        allow_partial_bank: Export a bank that lacks some objects.
        expected_population: Required test population (default: the frozen one).
        device: Torch device string (used for cache misses).
        manifest_extra: Further manifest fields (policy, bank).

    Returns:
        The export summary.

    Raises:
        ValueError: For a bad CSV name, partial bank, population, journal or heatmap problem,
            or a multi-view variant without ``spec``.
    """
    csv_path, method, dataset = export._resolve_csv(Path(out_csv))
    present, missing = export._bank_objects(bank, dataset, objects, allow_partial_bank)
    if inputs.frames is not None and inputs.population is not None:
        frames, population = list(inputs.frames), inputs.population
    else:
        frames, population = export.load_test_frames(test_root, camera)
    export.require_population(
        frames,
        expected_population
        if expected_population is not None
        else export.frozen_population(dataset),
    )
    references = set(inputs.groups)
    if {inputs.reference_keys[f.key] for f in frames} != references:
        raise ValueError("the variant groups must cover exactly the test population")
    results = csv_path.with_name(f"{csv_path.stem}{OBJECTS_DIR_SUFFIX}")
    records = {f.key: {"seconds": 0.0, "predictions": []} for f in frames}
    per_object, totals, resumed = {}, dict.fromkeys(_counts([]), 0), 0
    context = _ExportContext(
        variant=variant, frames=frames, population=population, inputs=inputs, spec=spec,
        refine=refine, cache=cache, camera=camera, device=device, results=results,
        groups_sha256=groups_sha256(inputs.groups), cameras_sha256=cameras_sha256(inputs.cameras),
    )  # fmt: skip
    for obj in present:
        result, reused = _compose_object(Path(bank[obj]), obj, context)
        resumed += reused
        for row in result["rows"]:
            key = (row["scene_id"], row["im_id"])
            records[key]["predictions"].append(
                {"score": row["score"], "R": row["R"], "t": row["t"], "object_id": obj}
            )
        per_object[str(obj)] = {
            "checkpoint_sha256": result["identity"]["checkpoint_sha256"],
            "crop_size": result["identity"]["settings"]["crop_size"],
            "beta": result["identity"]["settings"]["beta"],
            "journal_generations": result["identity"]["journal_generations"],
            **result["counts"],
        }
        totals = {k: totals[k] + result["counts"][k] for k in totals}
    export.write_csv(csv_path, records, record_time=False)
    summary = VariantSummary(
        csv_path=csv_path,
        csv_sha256=sha256_file(csv_path),
        variant=variant,
        images=len(frames),
        objects=present,
        missing_objects=missing,
        pose_rows=sum(len(r["predictions"]) for r in records.values()),
        resumed_objects=resumed,
        journal_generations={k: v["journal_generations"] for k, v in per_object.items()},
        **totals,
    )
    # Every object's crop size and beta replace the single values in place.
    manifest = variant_manifest(
        VariantSettings(refine=refine, variants=(variant,)),
        inputs.groups,
        variant=variant,
        crop_size=sorted({v["crop_size"] for v in per_object.values()}),
        beta=sorted({v["beta"] for v in per_object.values()}),
        cameras_sha256=context.cameras_sha256,
        method=method,
        dataset=dataset,
        csv=csv_path.name,
        csv_sha256=summary.csv_sha256,
        population_sha256=population,
        multiview=None if spec is None else repr(spec),
        objects=per_object,
    )
    manifest.update(manifest_extra or {})
    publish_json(csv_path.parent / MANIFEST_NAME, manifest)
    return summary
