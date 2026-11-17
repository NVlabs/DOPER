# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed NuRec recipe (``kind: nurec``) and its training stages.

A NuRec recipe adapts a synthetic-trained DOPER model to one real object from NuRec renders of
a reconstruction of real captures. It names the object (mesh, keypoint layout
by its points hash, COCO category), how the capture cameras are seeded for the reconstruction,
the COLMAP flags, the novel-view render, the label qualification, the adaptation split, the two
training stages and the relative-pose evaluation. ``examples/nurec/recipe.yaml`` is a template.

The training stages are ``base`` (the synthetic baseline, from the pretrained backbone) and
``adapt`` (the short full-label adaptation from the base run's ``best.pth``). ``base`` holds
``doper.train`` config keys and launch checks; ``adapt`` holds the keys that differ from
``base``. Dataset paths and ``init_from`` come from the run layout, never from the recipe.

``doper.recipes.load_any_recipe`` parses the recipe and refuses unknown keys. Parsing imports
no torch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..bop.sources import check_keys

SEED_MODES = ("robot_handeye", "colmap_sfm", "doper_pnp")
MASK_MODES = ("sam2", "cad_only")
STAGES = ("base", "adapt")
Stage = Literal["base", "adapt"]
# Train/validation images of the synthetic baseline dataset (``prepare --stage synthetic``),
# split by a deterministic image-id hash.
SYNTHETIC_SPLIT = (9750, 250)
# PnP needs four correspondences.
MIN_PNP_KEYPOINTS = 4
# Launch keys the run layout sets; a recipe stage never names them.
LAYOUT_KEYS = ("init_from", "resume")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class NurecObject:
    """The object the recipe adapts to.

    Attributes:
        mesh: Mesh file name inside the data bundle (meters, object frame).
        mesh_sha256: SHA-256 of that file: every stage that reads the bundle mesh refuses
            another (the capture bundle's identity does not cover the mesh).
        keypoints_points_sha256: SHA-256 of the keypoints as an N x 3 little-endian float64
            array: the layout's identity, independent of the file's names and formatting.
        category: COCO category name of the packaged datasets.
    """

    mesh: str
    mesh_sha256: str
    keypoints_points_sha256: str
    category: str

    def __post_init__(self) -> None:
        """Check the mesh and points hashes.

        Raises:
            ValueError: If one is not a lowercase hex SHA-256.
        """
        for key in ("mesh_sha256", "keypoints_points_sha256"):
            if not _SHA256.fullmatch(str(getattr(self, key))):
                raise ValueError(
                    f"object.{key} must be a lowercase hex SHA-256, got {getattr(self, key)!r}"
                )


@dataclass(frozen=True)
class RenderSpec:
    """Novel-view render of the reconstruction.

    Attributes:
        image_digest: NuRec renderer container image, pinned by digest (``name@sha256:...``).
        count: Number of novel views.
        seed: Sampler seed.
        seed_views: Indices of the source cameras the sampler perturbs, round-robin.
        perturb_deg: Half-range of the uniform per-axis camera-local rotation, degrees.
        perturb_m: Half-range of the uniform per-axis camera-local translation, scene units.
        reject_deg: A view within this rotation AND ``reject_mm`` of a source is rejected.
        reject_mm: Translation half of the rejection test, millimeters.
    """

    image_digest: str
    count: int
    seed: int
    seed_views: tuple[int, ...]
    perturb_deg: float
    perturb_m: float
    reject_deg: float
    reject_mm: float

    def __post_init__(self) -> None:
        """Normalize ``seed_views`` to a tuple and check the digest and counts.

        Raises:
            ValueError: For an image without a ``@sha256:`` digest, a non-positive count or an
                empty or negative seed view list.
        """
        if not _DIGEST.fullmatch(str(self.image_digest)):
            raise ValueError(
                f"render.image_digest must pin the image as NAME@sha256:<64 hex>, got "
                f"{self.image_digest!r}"
            )
        views = self.seed_views
        if isinstance(views, (str, bytes)) or not views:
            raise ValueError(f"render.seed_views must be a non-empty list, got {views!r}")
        if not all(isinstance(v, int) and v >= 0 for v in views):
            raise ValueError(f"render.seed_views must be camera indices, got {views!r}")
        object.__setattr__(self, "seed_views", tuple(views))
        if self.count <= 0:
            raise ValueError(f"render.count must be positive, got {self.count}")


@dataclass(frozen=True)
class HsvExclusion:
    """A color range (OpenCV 8-bit HSV) excluded from keypoint visibility and SAM2 prompts.

    A pixel is excluded when it satisfies every bound that is set; all bounds are strict.

    Attributes:
        h_above: Hue strictly above this.
        h_below: Hue strictly below this.
        s_above: Saturation strictly above this.
        s_below: Saturation strictly below this.
        v_above: Value strictly above this.
        v_below: Value strictly below this.
    """

    h_above: float | None = None
    h_below: float | None = None
    s_above: float | None = None
    s_below: float | None = None
    v_above: float | None = None
    v_below: float | None = None


@dataclass(frozen=True)
class LabelSpec:
    """Per-view label qualification.

    Attributes:
        mask_mode: ``sam2`` (the reference path) or ``cad_only`` (no mask supervision).
        min_visible: Minimum visible keypoints of a qualified view.
        min_coverage: Minimum ``|mask and CAD| / |CAD|`` of a qualified view (``sam2``).
        exclude_hsv: Color excluded from visibility and prompts; None excludes none.
        visibility_tolerance_m: Ray-mesh self-occlusion tolerance, meters.
        erode_px: Side of the square kernel that erodes the mask before the
            keypoint-inside-mask test, pixels (0 or 1 erodes nothing).
    """

    mask_mode: str
    min_visible: int
    min_coverage: float
    exclude_hsv: HsvExclusion | None
    visibility_tolerance_m: float
    erode_px: int

    def __post_init__(self) -> None:
        """Check the mask mode and bounds.

        Raises:
            ValueError: For an unknown mask mode or an out-of-range threshold.
        """
        if self.mask_mode not in MASK_MODES:
            raise ValueError(
                f"labels.mask_mode must be one of {MASK_MODES}, got {self.mask_mode!r}"
            )
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"labels.min_coverage must be in [0, 1], got {self.min_coverage}")
        if self.min_visible < 0 or self.erode_px < 0 or self.visibility_tolerance_m < 0:
            raise ValueError("labels.min_visible, erode_px and visibility_tolerance_m must be >= 0")


@dataclass(frozen=True)
class SplitSpec:
    """Train/validation split of the qualified adaptation views.

    Attributes:
        seed: Split seed.
        train: Training views.
        val: Validation views.
        single_reconstruction_split: Allow a split of views from one reconstruction (one
            capture lineage), which validation then shares with training; recorded as
            ``lineage_split: false``.
    """

    seed: int
    train: int
    val: int
    single_reconstruction_split: bool = False

    def __post_init__(self) -> None:
        """Check the counts.

        Raises:
            ValueError: For a non-positive count.
        """
        if self.train <= 0 or self.val <= 0:
            raise ValueError(f"split.train and split.val must be positive: {self.train}/{self.val}")


@dataclass(frozen=True)
class EvaluationSpec:
    """Relative-pose evaluation on the real captures.

    Attributes:
        score_thr: Detection score threshold.
        exclude: Capture names left out of the metric (recorded in the summary).
    """

    score_thr: float
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize ``exclude`` to a tuple.

        Raises:
            ValueError: For a non-list ``exclude``.
        """
        if isinstance(self.exclude, (str, bytes)) or not all(
            isinstance(e, str) for e in self.exclude
        ):
            raise ValueError(f"evaluation.exclude must be a list of names, got {self.exclude!r}")
        object.__setattr__(self, "exclude", tuple(self.exclude))


def _indices(values: Any, where: str) -> tuple[int, ...]:  # noqa: ANN401
    """Capture indices as a tuple; refuses a non-list or a negative or non-integer entry."""
    if isinstance(values, (str, bytes)) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values
    ):
        raise ValueError(f"{where} must be a list of capture indices, got {values!r}")
    return tuple(values)


@dataclass(frozen=True)
class RestartSpec:
    """Perturbed restarts of the edge refinement (a diagnostic; ``restart_audit``).

    Attributes:
        seed: RNG seed.
        starts: Number of restarts.
        rot_deg: Half-range of the uniform per-component rotation-vector perturbation, degrees.
        trans: Half-range of the uniform per-axis translation perturbation, scene units.
        max_nfev: Function evaluations per restart.
    """

    seed: int = 62
    starts: int = 6
    rot_deg: float = 5.0
    trans: float = 0.002
    max_nfev: int = 150

    def __post_init__(self) -> None:
        """Check the counts.

        Raises:
            ValueError: For a negative restart count, a non-positive evaluation budget or a
                negative range.
        """
        if self.starts < 0 or self.max_nfev <= 0 or self.rot_deg < 0 or self.trans < 0:
            raise ValueError("registration.restarts needs starts >= 0, max_nfev > 0 and "
                             "non-negative ranges")  # fmt: skip


@dataclass(frozen=True)
class RegistrationSpec:
    """CAD-to-scene registration (``prepare --stage register``; ``doper.nurec.registration``).

    Views are capture indices (capture order), like ``render.seed_views``.

    Attributes:
        fit_views: Views whose image edges the refinement fits.
        holdout_views: Views held out of the fit and reported.
        exclude_views: Views left out of the keypoint triangulation.
        max_reproj_px: PnP reprojection error a view's keypoints must stay below to be
            triangulated, pixels.
        canny: Canny thresholds of the edge images.
        boundary_px: Distance to the seed silhouette boundary within which a CAD vertex is a
            refinement point, pixels.
        points: Approximate refinement points per view.
        f_scale: Soft-L1 inlier scale of the refinement, pixels.
        diff_step: Relative finite-difference step of the refinement.
        max_nfev: Function evaluations of the refinement.
        restarts: The restart diagnostic.
    """

    fit_views: tuple[int, ...]
    holdout_views: tuple[int, ...] = ()
    exclude_views: tuple[int, ...] = ()
    max_reproj_px: float = 5.0
    canny: tuple[float, float] = (40.0, 100.0)
    boundary_px: float = 1.5
    points: int = 250
    f_scale: float = 2.0
    diff_step: float = 1e-4
    max_nfev: int = 200
    restarts: RestartSpec = field(default_factory=RestartSpec)

    def __post_init__(self) -> None:
        """Normalize the view lists and check the values.

        Raises:
            ValueError: For malformed or overlapping views, two Canny thresholds that are not
                increasing, or a non-positive limit.
        """
        for name in ("fit_views", "holdout_views", "exclude_views"):
            object.__setattr__(self, name, _indices(getattr(self, name), f"registration.{name}"))
        if not self.fit_views:
            raise ValueError("registration.fit_views must name at least one view")
        if set(self.fit_views) & set(self.holdout_views):
            raise ValueError("registration.fit_views and holdout_views must not overlap")
        canny = tuple(self.canny)
        if len(canny) != 2 or not 0 <= canny[0] < canny[1]:
            raise ValueError(f"registration.canny must be [low, high] with low < high, got "
                             f"{self.canny!r}")  # fmt: skip
        object.__setattr__(self, "canny", canny)
        for name in ("max_reproj_px", "boundary_px", "points", "f_scale", "diff_step",
                     "max_nfev"):  # fmt: skip
            if not getattr(self, name) > 0:
                raise ValueError(f"registration.{name} must be positive, got "
                                 f"{getattr(self, name)}")  # fmt: skip

    def as_record(self) -> dict[str, Any]:
        """The spec as a JSON object."""
        record = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "restarts"}
        return {**record, "restarts": {f.name: getattr(self.restarts, f.name)
                                       for f in fields(self.restarts)}}  # fmt: skip


@dataclass(frozen=True)
class DoperSeedGates:
    """The gates a DOPER pose passes before it seeds COLMAP (``seed_mode: doper_pnp``).

    The recipe's ``doper_pnp`` section (optional; these defaults when absent). Mode (c) is
    experimental.

    Attributes:
        score_thr: Smallest detection score; also the keypoint score a PnP correspondence needs.
        min_keypoints: Fewest keypoints scored at least ``score_thr``.
        max_reproj_px: Largest median PnP reprojection error over those keypoints (capture
            pixels).
        max_consistency_px: Largest median residual of an image's keypoints against the
            keypoints triangulated from every seeded image with the cameras fixed at their
            DOPER poses (capture pixels); also the largest median reprojection error of an
            image's COLMAP features after the first triangulation of the seed (the scene
            evidence, normalized pixels).
        register_leftovers: Register the images the gates left out with ``image_registrator``
            after the seeded refinement.
    """

    score_thr: float = 0.3
    min_keypoints: int = 8
    max_reproj_px: float = 3.0
    max_consistency_px: float = 5.0
    register_leftovers: bool = True

    def __post_init__(self) -> None:
        """Check the values.

        Raises:
            ValueError: For a score outside (0, 1], fewer than four keypoints (PnP needs
                them), a non-positive pixel limit or a non-boolean ``register_leftovers``.
        """
        if not 0.0 < self.score_thr <= 1.0:
            raise ValueError(f"doper_pnp score_thr must be in (0, 1], got {self.score_thr}")
        if not isinstance(self.register_leftovers, bool):
            raise ValueError(f"doper_pnp register_leftovers must be true or false, got "
                             f"{self.register_leftovers!r}")  # fmt: skip
        if (
            not isinstance(self.min_keypoints, int)
            or isinstance(self.min_keypoints, bool)
            or (self.min_keypoints < MIN_PNP_KEYPOINTS)
        ):
            raise ValueError(f"doper_pnp min_keypoints must be at least {MIN_PNP_KEYPOINTS}, "
                             f"got {self.min_keypoints}")  # fmt: skip
        for name in ("max_reproj_px", "max_consistency_px"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"doper_pnp {name} must be positive, got {getattr(self, name)}")

    def as_record(self) -> dict[str, Any]:
        """The gates as a JSON object."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class NurecRecipe:
    """A NuRec adaptation recipe.

    Attributes:
        kind: ``nurec`` (class constant; the YAML must carry it).
        name: Recipe name.
        profile: Profile name. A packaged profile pins its capture bundle's identity, which the
            capture stage and the evaluation check; a name with no packaged profile (a user's own
            capture) pins none.
        object: The object.
        seed_mode: Camera seeding of the reconstruction: ``robot_handeye`` (known camera poses
            from ``robot.jsonl`` and ``handeye.json``), ``colmap_sfm`` or ``doper_pnp``.
        colmap: COLMAP command name to its flags (``--Section.option`` names without dashes).
        render: Novel-view render.
        labels: Label qualification.
        split: Adaptation split.
        base: ``doper.train`` config and launch keys of the synthetic baseline.
        adapt: Keys of the adaptation that differ from ``base``.
        evaluation: Relative-pose evaluation.
        registration: CAD-to-scene registration.
        doper_pnp: The gates of ``seed_mode: doper_pnp`` (mode (c)).
        source: The recipe file, when loaded from one.
    """

    kind: ClassVar[str] = "nurec"
    name: str
    profile: str
    object: NurecObject
    seed_mode: str
    colmap: Mapping[str, Mapping[str, Any]]
    render: RenderSpec
    labels: LabelSpec
    split: SplitSpec
    base: Mapping[str, Any]
    adapt: Mapping[str, Any]
    evaluation: EvaluationSpec
    registration: RegistrationSpec
    doper_pnp: DoperSeedGates = field(default_factory=DoperSeedGates)
    source: Path | None = None

    def stage_config(self, stage: Stage) -> dict[str, Any]:
        """The recipe's ``doper.train`` keys of one stage (``adapt`` applied over ``base``).

        Raises:
            ValueError: For an unknown stage.
        """
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        return {**self.base, **(self.adapt if stage == "adapt" else {})}


def _check_keys(cls: type, data: Any, where: str) -> None:  # noqa: ANN401
    """Refuse a non-mapping section or unknown and missing keys of ``cls``."""
    required = [
        f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
    ]
    check_keys(
        data, [f.name for f in fields(cls)], required, f"nurec recipe {where or 'top level'}",
        not_mapping=f"nurec recipe {where} must be a mapping",
    )  # fmt: skip


def _sub(cls: type, data: Any, where: str) -> Any:  # noqa: ANN401
    """Build a nested section, refusing unknown and missing keys."""
    _check_keys(cls, data, where)
    return cls(**data)


def _check_colmap(colmap: Any) -> dict[str, dict[str, Any]]:  # noqa: ANN401
    """Require a mapping of command name to a mapping of scalar flag values.

    The mapper and image registrator run with the intrinsics fixed whenever the capture gives
    them (``FIXED_MAPPER_FLAGS``), which would override a recipe flag of the same name; a recipe
    that sets one of them to another value is refused rather than silently overridden.

    Raises:
        ValueError: For a malformed mapping or a flag that conflicts with the fixed intrinsics.
    """
    from ..reconstruction.colmap_runner import FIXED_MAPPER_FLAGS

    if not isinstance(colmap, Mapping):
        raise ValueError("nurec recipe colmap must map a command to its flags")
    checked = {}
    for command, flags in colmap.items():
        if not isinstance(flags, Mapping) or not all(
            isinstance(v, (bool, int, float, str)) for v in flags.values()
        ):
            raise ValueError(f"nurec recipe colmap.{command} must map flag names to scalars")
        conflicts = sorted(key for key, value in FIXED_MAPPER_FLAGS.items()
                           if command in ("mapper", "image_registrator")
                           and key in flags and flags[key] != value)  # fmt: skip
        if conflicts:
            raise ValueError(f"nurec recipe colmap.{command} sets {conflicts}, which the "
                             "reconstruction fixes (known intrinsics are never refined); remove "
                             "them")  # fmt: skip
        checked[str(command)] = dict(flags)
    return checked


def _check_training_keys(values: Any, where: str) -> dict[str, Any]:  # noqa: ANN401
    """Require ``doper.train`` keys that are neither dataset paths nor layout launch keys."""
    from ..training.recipe import config_keys
    from .schema import PATH_KEYS

    if not isinstance(values, Mapping):
        raise ValueError(f"nurec recipe {where} must be a mapping of doper.train keys")
    unknown = sorted(set(values) - config_keys())
    if unknown:
        raise ValueError(f"nurec recipe {where} has keys doper.train does not accept: {unknown}")
    misplaced = sorted(set(values) & {*PATH_KEYS, *LAYOUT_KEYS})
    if misplaced:
        raise ValueError(
            f"nurec recipe {where} must not set {misplaced}; they follow from the run layout"
        )
    return dict(values)


def _registration(data: Any) -> RegistrationSpec:  # noqa: ANN401
    """Build the registration section with its nested restarts, refusing unknown keys."""
    _check_keys(RegistrationSpec, data, "registration")
    restarts = _sub(RestartSpec, data.get("restarts", {}), "registration.restarts")
    return RegistrationSpec(**{**data, "restarts": restarts})


def parse_nurec_recipe(data: Mapping[str, Any], source: Path | None = None) -> NurecRecipe:
    """Build a NuRec recipe from its parsed YAML mapping.

    Raises:
        ValueError: For a wrong kind, unknown or missing keys, an unknown seed or mask mode, a
            malformed pin, a training key ``doper.train`` does not accept or that the layout
            owns, or a split without ``single_reconstruction_split`` (every packaged NuRec
            recipe reconstructs one capture set).
    """
    if data.get("kind") != NurecRecipe.kind:
        raise ValueError(f"recipe kind must be 'nurec', got {data.get('kind')!r}")
    values = {key: value for key, value in data.items() if key != "kind"}
    _check_keys(NurecRecipe, values, "")
    labels: Any = values["labels"]
    if isinstance(labels, Mapping) and labels.get("exclude_hsv") is not None:
        exclusion = _sub(HsvExclusion, labels["exclude_hsv"], "labels.exclude_hsv")
        labels = {**labels, "exclude_hsv": exclusion}
    if values["seed_mode"] not in SEED_MODES:
        raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {values['seed_mode']!r}")
    split = _sub(SplitSpec, values["split"], "split")
    if not split.single_reconstruction_split:
        raise ValueError(
            "split.single_reconstruction_split must be true: the recipe's views come from one "
            "reconstruction, so validation shares its scene with training (recorded as "
            "lineage_split: false)"
        )
    return NurecRecipe(
        name=values["name"],
        profile=values["profile"],
        object=_sub(NurecObject, values["object"], "object"),
        seed_mode=values["seed_mode"],
        colmap=_check_colmap(values["colmap"]),
        render=_sub(RenderSpec, values["render"], "render"),
        labels=_sub(LabelSpec, labels, "labels"),
        split=split,
        base=_check_training_keys(values["base"], "base"),
        adapt=_check_training_keys(values["adapt"], "adapt"),
        evaluation=_sub(EvaluationSpec, values["evaluation"], "evaluation"),
        registration=_registration(values["registration"]),
        doper_pnp=_sub(DoperSeedGates, values.get("doper_pnp", {}), "doper_pnp"),
        source=source,
    )


def stage_arguments(recipe: NurecRecipe, stage: Stage, paths: Mapping[str, str]) -> dict[str, Any]:
    """Resolve one training stage into the full ``doper.train`` arguments.

    The release training defaults, then the recipe's stage keys, then the run paths, parsed by
    the ``doper.train`` parser: the namespace ``doper.training.recipe.build_config`` takes.

    Args:
        recipe: The recipe.
        stage: ``base`` or ``adapt``.
        paths: ``data_root``, ``keypoints_3d`` and ``work_dir``; ``adapt`` also ``init_from``
            (the base run's ``best.pth``), which ``base`` must not give.

    Returns:
        Argument name to value, without ``config``.

    Raises:
        ValueError: For an unknown stage or a wrong set of paths.
    """
    from .schema import PATH_KEYS, LaunchSpec, TrainSpec, training_namespace

    config = recipe.stage_config(stage)
    needed = {*PATH_KEYS, "init_from"} if stage == "adapt" else set(PATH_KEYS)
    if set(paths) != needed:
        raise ValueError(f"{stage} stage paths must be {sorted(needed)}, got {sorted(paths)}")
    launch_names = {f.name for f in fields(LaunchSpec)}
    launch = {key: config.pop(key) for key in list(config) if key in launch_names}
    if stage == "adapt":
        launch["init_from"] = str(paths["init_from"])
    spec = TrainSpec(config=config, launch=LaunchSpec(**launch))
    _, namespace = training_namespace(spec, {key: str(paths[key]) for key in PATH_KEYS})
    arguments = dict(vars(namespace))
    arguments.pop("config", None)
    return arguments


def expected_updates(train_count: int, batch: int, accum: int, epochs: int) -> int:
    """Optimizer updates of a training run, as ``doper.train`` counts them.

    The training loader drops the last partial batch, and the last partial accumulation of an
    epoch still steps: ``ceil(floor(train_count / batch) / accum) * epochs``.

    Raises:
        ValueError: For a non-positive argument.
    """
    if min(train_count, batch, accum, epochs) <= 0:
        raise ValueError(
            f"train_count, batch, accum and epochs must be positive: "
            f"{train_count}, {batch}, {accum}, {epochs}"
        )
    return math.ceil((train_count // batch) / accum) * epochs


__all__ = [
    "MASK_MODES",
    "SEED_MODES",
    "STAGES",
    "SYNTHETIC_SPLIT",
    "DoperSeedGates",
    "EvaluationSpec",
    "HsvExclusion",
    "LabelSpec",
    "NurecObject",
    "NurecRecipe",
    "RegistrationSpec",
    "RenderSpec",
    "RestartSpec",
    "SplitSpec",
    "Stage",
    "expected_updates",
    "parse_nurec_recipe",
    "stage_arguments",
]
