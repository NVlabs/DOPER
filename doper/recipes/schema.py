# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed BOP recipes: one YAML file per dataset, validated against ``DoperConfig``.

A recipe names a pinned dataset profile, the objects, how keypoints are selected and PBR is
converted and split, the base training config, and the validation and test protocols. The
training config uses the ``doper.train`` config-file keys; dataset paths (``data_root``,
``keypoints_3d``, ``work_dir``) are filled in by ``resolve_training``, and the launch fields
(``init_from``, ``resume`` and the ``require_*`` checks) live in a typed ``LaunchSpec``.
``load_recipe`` rejects unknown keys at every level and resolves the training config into a
``DoperConfig`` once, so a bad recipe fails before any data is touched. Nothing here imports
torch.

The optional ``adapt`` section is the shared adaptation template and the table of adaptation
methods (``AdaptSpec``, ``MethodSpec``); ``doper.training.adaptation`` resolves it against the
base stage, whose backbone and architecture every method inherits. The optional ``followups``
section names single-object self-training series over that template (``FollowupSpec``,
``doper.training.followups``). ``adapt.multisensor`` (T-LESS) is the three-sensor real TRAIN
adaptation stage (``MultisensorSpec``), a stage of its own rather than a pseudo-label method.
The optional ``multiview`` section is the dataset's multi-view fusion policy (``MultiviewSpec``,
used by ``doper.bop.multiview``); a recipe without it has no multi-view test variants. The
optional ``refine`` section (``RefineSpec``) sets the heatmap refinement defaults of the test
variants (``doper.bop.variants``); ``doper evaluate --optimizer/--preparation`` override it.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from ..association_gate import AssociationGate
from ..bop.layout import RESERVED_RUN_DIRS, round_name
from ..config import DoperConfig
from ..pseudo_labels.gate import GateSpec

ProtocolName = Literal["top1", "top5-greedy"]
CameraName = Literal["pinhole", "native"]
PATH_KEYS = ("data_root", "keypoints_3d", "work_dir")
HOT3D_PANEL = "hot3d_panel"
HOT3D_INDEX = "hot3d_index"
PLACEHOLDER_ROOT = "<data_root>"
MethodSource = Literal["static", "dynamic", "test", "combined", "gt"]
Supervision = Literal["box_mask", "box_mask_keypoints"]
METHOD_SOURCES = ("static", "dynamic", "test", "combined", "gt")
SUPERVISIONS = ("box_mask", "box_mask_keypoints")
PolicyFallback = Literal["none", "skipped", "out_of_scope"]
POLICY_FALLBACKS = ("none", "skipped", "out_of_scope")
# Policies ``doper evaluate`` builds itself: every base model, or the validation-selected one.
BUILTIN_POLICIES = ("base", "selected")
_CSV_TOKEN = re.compile(r"[A-Za-z0-9]+")
# Policy and alias names are directory names under ``<runs>/test``; ``cache`` holds the
# per-model journal cache there.
TEST_CACHE_DIR = "cache"
_NAME_TOKEN = re.compile(r"[a-z0-9_]+")
RESERVED_TEST_NAMES = (TEST_CACHE_DIR,)
COMPONENT_SOURCES = ("static", "dynamic")
# Model keys every adaptation method inherits from the base stage.
ARCHITECTURE_KEYS = (
    "backbone",
    "allow_nonstandard_backbone",
    "num_classes",
    "num_keypoints",
    "det_size",
    "crop_size",
    "keypoint_head",
    "keypoint_heatmap_sigma",
    "letterbox",
)
# Keys the adaptation engine computes from the label pool.
COMPUTED_ADAPT_KEYS = (
    "mix_data_root",
    "mix_per_batch",
    "max_iters_per_epoch",
    "min_mix_keypoint_samples",
    "min_mix_mask_only_samples",
)
BASE_ARM = "base"
# Every adaptation student trains exactly three epochs; follow-up teachers are such students.
ADAPT_EPOCHS = 3
# The T-LESS multisensor stage: its run directory and its mix roots (PrimeSense, Kinect, Canon).
MULTISENSOR_ARM = "multisensor"
MULTISENSOR_SOURCES = 3
# Follow-up rounds are ``roundNN`` directories.
MAX_FOLLOWUP_ROUNDS = 99
# Follow-ups self-train on this onboarding view (the source HANDAL static loop).
FOLLOWUP_SOURCE: MethodSource = "static"


@dataclass(frozen=True)
class KeypointSpec:
    """Hull-anchored keypoint selection on the CAD model (``doper.keypoint_selection``).

    Attributes:
        count: Keypoints per object.
        source_units: Length unit of the CAD model (BOP: ``mm``).
    """

    count: int = 64
    source_units: str = "mm"


SplitPolicy = Literal["scenes", "layout_modulo"]
EmptyMaskPolicy = Literal["fail", "exclude", "keep"]
SPLIT_POLICIES = ("scenes", "layout_modulo")
EMPTY_MASK_POLICIES = ("fail", "exclude", "keep")


@dataclass(frozen=True)
class ConvertSpec:
    """PBR conversion and whole-scene split.

    Attributes:
        models: Directory under ``data/`` holding the training CAD (``models``,
            ``models_cad``).
        val_scenes: PBR scenes held out for the training-time validation split.
        seed: Seed of the held-out scene sample.
        source_view: ``none``, ``exclusions`` (HANDAL: drop documented defective frames) or
            ``hot3d`` (canonical-millimeter view with the pinned depth and mask filters).
        allow_source_empty_masks: Let the projection audit accept empty source masks.
        split: ``scenes`` (a seeded sample of ``val_scenes`` whole scenes) or ``layout_modulo``
            (generated renders: layout ``l`` is validation when ``l % val_modulo ==
            val_residue``, see ``GeneratedSpec``).
        empty_masks: Converted annotations whose source visible pixel count is zero: ``fail``
            refuses them, ``exclude`` drops them (``doper.bop.generated.select_visible``),
            ``keep`` trains on them unchanged (the projection audit then accepts empty masks
            that the source GT info confirms).
    """

    models: str = "models"
    val_scenes: int = 5
    seed: int = 42
    source_view: Literal["none", "exclusions", "hot3d"] = "none"
    allow_source_empty_masks: bool = False
    split: SplitPolicy = "scenes"
    empty_masks: EmptyMaskPolicy = "fail"


@dataclass(frozen=True, kw_only=True)
class GeneratedSpec:
    """The BlenderProc render campaign one generated object trains from.

    The campaign itself is rendered outside ``doper`` (``rendering/blenderproc/campaign.py``);
    ``doper prepare --generated-root`` verifies its receipts against these values and prepares
    the object like an official one, with the ``layout_modulo`` split.

    Attributes:
        render_config: Name of the render config (``rendering/blenderproc/configs/<name>.json``).
        layouts: Physical layouts of the campaign; every one must be published.
        frames_per_layout: Views per layout.
        base_seed: Seed of layout 0; layout ``l`` renders with ``base_seed + l``.
        val_modulo: Layout ``l`` is validation when ``l % val_modulo == val_residue``.
        val_residue: See ``val_modulo``.
        conversion_shards: The conversion shards (shard ``s`` holds the layouts
            ``range(s, layouts, conversion_shards)``); each split lists its images shard by
            shard in that order (HANDAL obj40 16, HOT3D obj33 4).
        keypoints_sha256: SHA-256 of the keypoint selection on the object's official CAD, the
            file the campaign is staged with (``prepare --keypoints-only``); None: unpinned.
        empty_masks: ``fail``, ``exclude`` or ``keep`` annotations whose visible mask is empty
            (``ConvertSpec.empty_masks``).
    """

    render_config: str
    layouts: int = 1000
    frames_per_layout: int = 25
    base_seed: int
    val_modulo: int = 10
    val_residue: int = 9
    conversion_shards: int = 1
    keypoints_sha256: str | None = None
    empty_masks: EmptyMaskPolicy

    def convert(self, convert: ConvertSpec) -> ConvertSpec:
        """The conversion of the object: ``convert`` with the layout split and mask policy."""
        return replace(convert, split="layout_modulo", empty_masks=self.empty_masks)

    def as_record(self) -> dict[str, Any]:
        """The spec as JSON (the prepared object's identity)."""
        return asdict(self)


@dataclass(frozen=True)
class LaunchSpec:
    """Initialization and recipe checks of a training launch (``doper.train`` flags).

    Attributes:
        init_from: Checkpoint to initialize from.
        resume: Checkpoint to resume.
        require_no_init_from: Refuse ``init_from`` and ``resume``.
        require_pretrained_backbone: Refuse a backbone without ImageNet weights.
        require_init_from: Require ``init_from`` and no ``resume``.
    """

    init_from: str | None = None
    resume: str | None = None
    require_no_init_from: bool = False
    require_pretrained_backbone: bool = False
    require_init_from: bool = False


@dataclass(frozen=True)
class TrainSpec:
    """One training stage: ``doper.train`` config-file values and the launch fields.

    Attributes:
        config: Config-file keys (``backbone``, ``epochs``, ...) without dataset paths.
        launch: Initialization and recipe checks.
    """

    config: Mapping[str, Any]
    launch: LaunchSpec = field(default_factory=LaunchSpec)


@dataclass(frozen=True)
class ValidationSpec:
    """Validation AR of each object model (``doper.bop.ar.score_model``).

    Attributes:
        population: Split directory under ``data/`` (``val``), ``hot3d_panel`` (the staged
            HOT3D manifest) or ``none`` (no validation split; selection keeps the base).
        models: Directory under ``data/`` holding the evaluation CAD and ``models_info.json``.
        protocol: ``top1`` or ``top5-greedy``.
        camera: ``pinhole`` or ``native``.
    """

    population: str = "val"
    models: str = "models"
    protocol: ProtocolName = "top1"
    camera: CameraName = "pinhole"


@dataclass(frozen=True)
class Policy:
    """A fixed-policy test bank: one method's endpoint per object, base where it may not exist.

    Attributes:
        name: Policy name (``doper evaluate --policy``).
        method: Run directory of the endpoints under the runs root: a recipe method, or a
            follow-up run path such as ``obj8_loop/round03``.
        csv_method: BOP CSV method token (``[A-Za-z0-9]+``): ``<csv_method>_<dataset>-test.csv``.
        fallback: Where the object's base stands in: ``none`` (never), ``skipped`` (the
            method was skipped below its label minimum) or ``out_of_scope`` (objects outside
            ``objects``).
        objects: The objects an ``out_of_scope`` policy changes; None for every object.
        record_time: The export writes measured per-image seconds (every model rescanned with
            all models resident, no journal cache), whatever ``test.record_time`` says: the
            bank was directly timed in the source (T-LESS ``multisensor``).
    """

    name: str
    method: str
    csv_method: str
    fallback: PolicyFallback = "none"
    objects: tuple[int, ...] | None = None
    record_time: bool = False


@dataclass(frozen=True)
class AliasSpec:
    """A selection candidate kept under its source name (``select_bop_test_models``).

    Attributes:
        run: Run directory under the runs root (a recipe method or a follow-up run path); the
            object's endpoint is ``<runs>/<run>/obj_XXXXXX``. A recipe method named here is a
            candidate only under the alias for ``objects``.
        objects: Objects the alias applies to.
    """

    run: str
    objects: tuple[int, ...]


@dataclass(frozen=True)
class TestSpec:
    """BOP test export (``doper.bop.export.run_bank``).

    Attributes:
        population: Split directory under ``data/`` (``test``) or ``hot3d_index`` (the staged
            HOT3D test manifest).
        camera: ``pinhole`` or ``native``.
        method: Method name of the BOP CSV (``<method><policy>_<dataset>-test.csv``).
        dataset: Dataset name of the BOP CSV.
        policies: Fixed-policy banks besides the built-in ``base`` and ``selected``.
        aliases: Validation-selection candidates kept under a source name.
        record_time: Fixed-policy exports write measured per-image seconds (HOT3D: every
            source bank was directly timed); untimed exports reuse the per-model journal cache.
    """

    __test__ = False  # not a pytest test class

    population: str = "test"
    camera: CameraName = "pinhole"
    method: str = "doper"
    dataset: str = ""
    policies: Mapping[str, Policy] = field(default_factory=dict)
    aliases: Mapping[str, AliasSpec] = field(default_factory=dict)
    record_time: bool = False


@dataclass(frozen=True)
class PrepareSpec:
    """Dataset preparation.

    Attributes:
        parts: Profile parts ``doper prepare`` fetches by default.
    """

    parts: tuple[str, ...] = ("models", "pbr", "val")


@dataclass(frozen=True, kw_only=True)
class MethodSpec:
    """One adaptation method: where its labels come from and when it trains.

    Attributes:
        source: ``static``, ``dynamic`` or ``test`` (a strict-gate pseudo-label pool of that
            view), ``combined`` (the union of ``components``) or ``gt`` (the paired GT pool).
        supervision: Real supervision: ``box_mask``, or ``box_mask_keypoints`` (GT only).
        minimum_labels: Fewer accepted labels skip the method (``adaptation_skipped.json``).
        minimum_overrides: Per-object minimum, replacing ``minimum_labels``.
        objects: Objects the method applies to (None: every recipe object).
        components: The ``static`` and ``dynamic`` methods a ``combined`` method unions.
        transductive: Whether the labels come from the unlabeled test inputs (needs consent).
    """

    source: MethodSource
    supervision: Supervision = "box_mask"
    minimum_labels: int
    minimum_overrides: Mapping[int, int] = field(default_factory=dict)
    objects: tuple[int, ...] | None = None
    components: tuple[str, ...] = ()
    transductive: bool = False

    def minimum_for(self, obj_id: int) -> int:
        """The minimum accepted labels for ``obj_id``."""
        return int(self.minimum_overrides.get(obj_id, self.minimum_labels))

    def applies_to(self, obj_id: int) -> bool:
        """Whether the method runs for ``obj_id``."""
        return self.objects is None or obj_id in self.objects


@dataclass(frozen=True, kw_only=True)
class MultisensorSpec:
    """The T-LESS three-sensor real TRAIN adaptation (``doper train --stage multisensor``).

    The student initializes from the object's epoch-25 base and mixes, per batch, PBR with the
    prepared PrimeSense, Kinect and Canon roots (``doper.bop.multisensor``) in that order. The
    epoch is uncapped (``max_iters_per_epoch: 0``): with all-instance sampling it lasts
    ``max_i ceil(n_i / b_i)`` batches over the four streams, so every retained sample is seen.

    Attributes:
        config: ``doper.train`` config-file keys that replace the base stage's (optimizer,
            budget, precision); the same rules as ``AdaptSpec.config``.
        mix_per_batch: Real samples per batch of PrimeSense, Kinect and Canon; PBR fills the
            rest of ``batch_size``.
        min_mix_keypoint_samples: ``doper.train`` refuses fewer keypoint-supervised real
            samples (GT keypoints: 1).
        min_mix_mask_only_samples: ``doper.train`` refuses fewer mask-only real samples.
    """

    config: Mapping[str, Any]
    mix_per_batch: tuple[int, int, int] = (2, 2, 2)
    min_mix_keypoint_samples: int = 1
    min_mix_mask_only_samples: int = 0


CameraConvention = Literal["w2c", "c2w_named_w2c"]
SupportMode = Literal["nearest", "streams"]
SupportScope = Literal["test_targets", "scene"]
CONVENTIONS: tuple[str, ...] = get_args(CameraConvention)
SUPPORT_MODES: tuple[str, ...] = get_args(SupportMode)
SUPPORT_SCOPES: tuple[str, ...] = get_args(SupportScope)
DEFAULT_GATE = AssociationGate(
    kp_score_thr=0.3,
    min_keypoints=6,
    ransac_px=3.0,
    max_reprojection_px=6.0,
    min_depth=50.0,
    max_depth=3000.0,
)
PREFILTER_PX = 30.0
NEAREST_K = 3


@dataclass(frozen=True, kw_only=True)
class MultiviewSpec:
    """A dataset's multi-view policy (recipe ``multiview`` block).

    Attributes:
        convention: How to read ``cam_R_w2c``/``cam_t_w2c``: ``w2c`` as named, or
            ``c2w_named_w2c`` (camera-to-world stored under the w2c names; inverted).
        support: ``nearest`` camera centers or synchronized ``streams``.
        nearest_k: Support views per reference (``nearest``).
        scope: Candidate supports of ``nearest``: the same scene's ``test_targets`` or every
            frame of the ``scene``.
        scenes: Scenes with multi-view (``nearest``); None for every scene. References in other
            scenes stay single-view.
        streams: Synchronized streams in group order (``streams``).
        skip_nonpositive_support_depth: Skip a support whose transformed original pose has
            depth at most zero.
        prefilter_px: Median keypoint distance below which a support detection is a candidate.
        gate: Association and acceptance thresholds; depths in millimeters.
    """

    convention: CameraConvention
    support: SupportMode
    nearest_k: int = NEAREST_K
    scope: SupportScope = "test_targets"
    scenes: tuple[int, ...] | None = None
    streams: tuple[str, ...] = ()
    skip_nonpositive_support_depth: bool = False
    prefilter_px: float = PREFILTER_PX
    gate: AssociationGate = field(default_factory=lambda: DEFAULT_GATE)

    def __post_init__(self) -> None:
        """Reject values outside the supported policies."""
        if self.convention not in CONVENTIONS:
            raise ValueError(f"multiview convention must be one of {CONVENTIONS}")
        if self.support not in SUPPORT_MODES:
            raise ValueError(f"multiview support must be one of {SUPPORT_MODES}")
        if self.scope not in SUPPORT_SCOPES:
            raise ValueError(f"multiview scope must be one of {SUPPORT_SCOPES}")
        if self.support == "streams" and not self.streams:
            raise ValueError("multiview support 'streams' needs the stream list")
        if self.support == "nearest" and self.streams:
            raise ValueError("multiview streams are for support 'streams' only")
        if self.support == "nearest" and self.nearest_k < 1:
            raise ValueError("multiview nearest_k must be positive")
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("multiview streams must be distinct")
        if not self.prefilter_px > 0:
            raise ValueError("multiview prefilter_px must be positive")


RefineOptimizer = Literal["lbfgsb", "powell"]
RefinePreparation = Literal["numpy", "parallel"]
REFINE_OPTIMIZERS: tuple[str, ...] = get_args(RefineOptimizer)
REFINE_PREPARATIONS: tuple[str, ...] = get_args(RefinePreparation)
REFINE_THREADS = 4


@dataclass(frozen=True, kw_only=True)
class RefineSpec:
    """Heatmap refinement defaults of the test variants (recipe ``refine`` block).

    Mirrors ``doper.refinement.RefineSettings`` without importing it (recipes stay free of
    SciPy and OpenCV).

    Attributes:
        optimizer: ``lbfgsb`` (default; Numba, ``doper[refine]``) or ``powell`` (SciPy only;
            the optimizer of the paper's heatmap rows).
        preparation: ``numpy`` (default) or ``parallel`` (Numba, L-BFGS-B only).
        threads: Upper bound on Numba threads of the parallel preparation.
    """

    optimizer: RefineOptimizer = "lbfgsb"
    preparation: RefinePreparation = "numpy"
    threads: int = REFINE_THREADS

    def __post_init__(self) -> None:
        """Reject unknown values and Powell with the parallel preparation."""
        if self.optimizer not in REFINE_OPTIMIZERS:
            raise ValueError(f"refine optimizer must be one of {REFINE_OPTIMIZERS}")
        if self.preparation not in REFINE_PREPARATIONS:
            raise ValueError(f"refine preparation must be one of {REFINE_PREPARATIONS}")
        if self.optimizer == "powell" and self.preparation != "numpy":
            raise ValueError("Powell uses its original NumPy preparation; use preparation numpy")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("refine threads must be a positive integer")

    @property
    def needs_numba(self) -> bool:
        """Whether the optimizer (L-BFGS-B) or the preparation (parallel) runs Numba kernels."""
        return self.optimizer == "lbfgsb" or self.preparation == "parallel"

    def with_overrides(self, optimizer: str | None, preparation: str | None) -> RefineSpec:
        """This spec with the given optimizer and preparation, where not None (CLI flags)."""
        return replace(
            self,
            optimizer=optimizer or self.optimizer,
            preparation=preparation or self.preparation,
        )

    def to_settings(self) -> Any:  # noqa: ANN401
        """The ``doper.refinement.RefineSettings`` of this spec (imports the refinement)."""
        from ..refinement import RefineSettings

        return RefineSettings(
            optimizer=self.optimizer, preparation=self.preparation, threads=self.threads
        )


def numba_installed() -> bool:
    """Whether Numba (``doper[refine]``) is importable, without importing it."""
    import importlib.util

    return importlib.util.find_spec("numba") is not None


@dataclass(frozen=True, kw_only=True)
class AdaptSpec:
    """The shared adaptation template and the method table of a recipe.

    Attributes:
        config: ``doper.train`` config-file keys that replace the base stage's for adaptation
            (optimizer, budget, precision). Architecture keys come from the base; the mix
            roots, ``max_iters_per_epoch`` and the min-mix checks are computed per pool.
            Required with ``methods``.
        real_per_batch: Real labels per batch (``mix_per_batch``); an epoch is
            ``ceil(labels / real_per_batch)`` iterations.
        gate: The strict pseudo-label gate.
        methods: Method name to method; names are the run directories and selection arms.
        multisensor: The T-LESS three-sensor stage, with its own template; a recipe may have
            it instead of methods.
    """

    config: Mapping[str, Any] = field(default_factory=dict)
    real_per_batch: int = 4
    gate: GateSpec = field(default_factory=GateSpec)
    methods: Mapping[str, MethodSpec] = field(default_factory=dict)
    multisensor: MultisensorSpec | None = None


@dataclass(frozen=True, kw_only=True)
class FollowupSpec:
    """A single-object self-training series over the adaptation template (HANDAL Table 89).

    Every round is one adaptation run on the static onboarding view with the recipe's template,
    ``<runs>/<name>/roundNN/obj_XXXXXX``. Round 1 (the seed round) uses ``teacher``; each later
    round's teacher and initialization is the previous round's student. The series stops when a
    round accepts no labels, when a strict-gate pool reaches ``target_labels``, or after
    ``rounds`` rounds; a labels-only probe then measures the final student's strict-gate yield.

    Attributes:
        obj_id: The object.
        teacher: ``base`` (the completed epoch-25 base) or a recipe method whose completed
            student of ``obj_id`` seeds round 1.
        gate_overrides: ``GateSpec`` fields replaced in round 1 only (object 7's bootstrap
            relaxes ``max_reprojection_px`` to 5.0); later rounds use the strict recipe gate. A
            relaxed round 1 never counts toward ``target_labels``.
        minimum_labels: Fewer accepted labels in a round skip its student.
        rounds: The round cap.
        target_labels: A strict-gate pool of at least this many labels ends the series.
        reuse_labels_from: A static recipe method whose verified pool round 1 trains on instead
            of relabeling (it must share round 1's teacher and gate). Object 8's loop reuses
            the original ``static_pseudo`` pool that was below that method's minimum.
        require_review: Round 1 trains only after a human review of its pool overlays
            (``doper review``, ``REVIEWED.json``) or an explicit, recorded
            ``--accept-unreviewed``.
    """

    obj_id: int
    teacher: str
    gate_overrides: Mapping[str, float] = field(default_factory=dict)
    minimum_labels: int = 1
    rounds: int
    target_labels: int = 20
    reuse_labels_from: str | None = None
    require_review: bool = False

    def method(self) -> MethodSpec:
        """The adaptation method every round runs (static view, box/mask, this minimum)."""
        return MethodSpec(
            source=FOLLOWUP_SOURCE, minimum_labels=self.minimum_labels, objects=(self.obj_id,)
        )

    def as_record(self) -> dict[str, Any]:
        """The spec as JSON (``identity.json`` of the series)."""
        record = asdict(self)
        record["gate_overrides"] = dict(self.gate_overrides)
        return record


@dataclass(frozen=True)
class Recipe:
    """A dataset recipe.

    Attributes:
        name: Recipe name.
        profile: Packaged profile name, or a profile file path (relative to the recipe).
        objects: Every object of the dataset, in BOP id order.
        generated_objects: Objects without official PBR, whose base needs generated renders.
            ``prepare --generated-root`` prepares one from its campaign (``generated``); until
            then it is missing from every bank.
        generated: Render campaign of each generated object (``GeneratedSpec``); a generated
            object without one cannot be prepared.
        prepare: Dataset preparation.
        keypoints: Keypoint selection.
        convert: PBR conversion and split.
        base: Base training stage.
        validation: Validation protocol.
        test: Test export.
        adapt: Adaptation template and methods; None when the recipe has none.
        followups: Single-object self-training series over the adaptation template.
        multiview: Multi-view fusion policy; None when the dataset has no multi-view test.
        refine: Heatmap refinement defaults of the test variants.
        source: The recipe file, when loaded from one.
    """

    name: str
    profile: str
    objects: tuple[int, ...]
    generated_objects: tuple[int, ...]
    prepare: PrepareSpec
    keypoints: KeypointSpec
    convert: ConvertSpec
    base: TrainSpec
    validation: ValidationSpec
    test: TestSpec
    adapt: AdaptSpec | None = None
    followups: Mapping[str, FollowupSpec] = field(default_factory=dict)
    generated: Mapping[int, GeneratedSpec] = field(default_factory=dict)
    multiview: MultiviewSpec | None = None
    refine: RefineSpec = field(default_factory=RefineSpec)
    source: Path | None = None

    @property
    def trainable_objects(self) -> tuple[int, ...]:
        """Objects with official PBR: ``objects`` without ``generated_objects``."""
        return tuple(o for o in self.objects if o not in self.generated_objects)

    def generated_spec(self, obj: int) -> GeneratedSpec:
        """The render campaign of a generated object.

        Raises:
            ValueError: If ``obj`` is not a generated object with a render campaign.
        """
        if obj not in self.generated_objects or obj not in self.generated:
            raise ValueError(
                f"object {obj} is not a generated {self.name} object with a render "
                "campaign (recipe `generated`)"
            )
        return self.generated[obj]

    def convert_for(self, obj: int) -> ConvertSpec:
        """The conversion of one object (generated objects use their campaign's split)."""
        if obj in self.generated_objects and obj in self.generated:
            return self.generated[obj].convert(self.convert)
        return self.convert

    def profile_ref(self) -> str:
        """The profile as ``sources.load_profile`` takes it (a path resolves beside the recipe)."""
        if self.source is not None and self.profile.endswith(".json"):
            return str((self.source.parent / self.profile).resolve())
        return self.profile


def _recipe_directory() -> Path:
    return Path(str(resources.files("doper.recipes")))


def packaged_recipes() -> list[str]:
    """Names of the packaged recipes."""
    return sorted(p.stem for p in _recipe_directory().glob("*.yaml") if p.stem != "train_example")


def _strict(cls: type, data: Any, where: str) -> dict[str, Any]:  # noqa: ANN401
    if not isinstance(data, Mapping):
        raise ValueError(f"recipe {where} must be a mapping")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"recipe {where} has unknown keys {unknown}")
    return dict(data)


def _section(cls: type, data: Any, where: str) -> Any:  # noqa: ANN401
    values = _strict(cls, data if data is not None else {}, where)
    for key in ("parts",):
        if key in values:
            values[key] = tuple(values[key])
    return cls(**values)


def _launch_keys() -> set[str]:
    return {f.name for f in fields(LaunchSpec)}


def training_namespace(
    spec: TrainSpec,
    paths: Mapping[str, str],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Resolve a training stage into its config-file mapping and parsed ``doper.train`` args.

    Args:
        spec: Training stage.
        paths: ``data_root``, ``keypoints_3d`` and ``work_dir``.
        overrides: Config or launch keys that replace the recipe values.

    Returns:
        The resolved config-file mapping (the one ``doper train`` writes) and the arguments
        ``doper.train`` parses from it.

    Raises:
        ValueError: If ``overrides`` sets a dataset path (``PATH_KEYS``); paths come from
            ``paths`` only.
        SystemExit: For a key ``doper.train`` does not accept.
    """
    from ..training.recipe import build_parser

    replaced = sorted(set(overrides or {}) & set(PATH_KEYS))
    if replaced:
        raise ValueError(f"overrides must not set the dataset paths {replaced}")
    launch = {f.name: getattr(spec.launch, f.name) for f in fields(LaunchSpec)}
    resolved = {**dict(paths), **launch, **dict(spec.config), **dict(overrides or {})}
    return resolved, build_parser(resolved).parse_args([])


def resolve_config(
    spec: TrainSpec,
    paths: Mapping[str, str],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[DoperConfig, LaunchSpec]:
    """Resolve a training stage into the ``DoperConfig`` and ``LaunchSpec`` it trains with.

    Raises:
        ValueError: If the backbone is outside the supported set without an explicit opt-out.
    """
    from ..training.recipe import build_config, validate_backbone_policy

    _, args = training_namespace(spec, paths, overrides)
    cfg = build_config(args)
    validate_backbone_policy(cfg.model.backbone, args.allow_nonstandard_backbone)
    launch = LaunchSpec(**{f.name: getattr(args, f.name) for f in fields(LaunchSpec)})
    return cfg, launch


def _train_spec(data: Any, where: str) -> TrainSpec:  # noqa: ANN401
    values = _strict(TrainSpec, data, where)
    config = dict(values.get("config") or {})
    from ..training.recipe import config_keys

    unknown = sorted(set(config) - config_keys())
    misplaced = sorted(set(config) & (set(PATH_KEYS) | _launch_keys()))
    if unknown:
        raise ValueError(f"recipe {where}.config has keys doper.train does not accept: {unknown}")
    if misplaced:
        raise ValueError(f"recipe {where}.config must not set {misplaced}; see LaunchSpec")
    return TrainSpec(config, _section(LaunchSpec, values.get("launch"), f"{where}.launch"))


def _adapt_config(data: Any, where: str = "adapt.config") -> dict[str, Any]:  # noqa: ANN401
    from ..training.recipe import config_keys

    if not isinstance(data, Mapping):
        raise ValueError(f"recipe {where} must be a mapping")
    config = dict(data)
    unknown = sorted(set(config) - config_keys())
    if unknown:
        raise ValueError(f"recipe {where} has keys doper.train does not accept: {unknown}")
    misplaced = sorted(set(config) & (set(PATH_KEYS) | _launch_keys()))
    if misplaced:
        raise ValueError(f"recipe {where} must not set {misplaced}; the engine sets them")
    inherited = sorted(set(config) & set(ARCHITECTURE_KEYS))
    if inherited:
        raise ValueError(f"recipe {where} sets {inherited}; adaptation inherits the base's")
    computed = sorted(set(config) & set(COMPUTED_ADAPT_KEYS))
    if computed:
        raise ValueError(f"recipe {where} sets {computed}; the engine computes them per pool")
    if config.get("epochs") != ADAPT_EPOCHS:
        raise ValueError(f"recipe {where}.epochs must be {ADAPT_EPOCHS} (the fixed budget)")
    return config


def _multisensor_spec(data: Any) -> MultisensorSpec | None:  # noqa: ANN401
    if data is None:
        return None
    where = "adapt.multisensor"
    values = _strict(MultisensorSpec, data, where)
    mix = tuple(int(v) for v in values.get("mix_per_batch", (2, 2, 2)))
    if len(mix) != MULTISENSOR_SOURCES or min(mix) < 1:
        raise ValueError(f"recipe {where}.mix_per_batch needs three positive counts")
    keypoint = int(values.get("min_mix_keypoint_samples", 1))
    mask_only = int(values.get("min_mix_mask_only_samples", 0))
    if min(keypoint, mask_only) < 0:
        raise ValueError(f"recipe {where}: the min-mix sample counts must not be negative")
    return MultisensorSpec(
        config=_adapt_config(values.get("config") or {}, f"{where}.config"),
        mix_per_batch=(mix[0], mix[1], mix[2]),
        min_mix_keypoint_samples=keypoint,
        min_mix_mask_only_samples=mask_only,
    )


def _method_spec(data: Any, where: str, objects: tuple[int, ...]) -> MethodSpec:  # noqa: ANN401
    values = _strict(MethodSpec, data, where)
    for key in ("objects", "components"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    overrides = {int(k): int(v) for k, v in (values.get("minimum_overrides") or {}).items()}
    values["minimum_overrides"] = overrides
    spec = MethodSpec(**values)
    if spec.source not in METHOD_SOURCES:
        raise ValueError(f"recipe {where}.source must be one of {METHOD_SOURCES}")
    if spec.supervision not in SUPERVISIONS:
        raise ValueError(f"recipe {where}.supervision must be one of {SUPERVISIONS}")
    if spec.source != "gt" and spec.supervision != "box_mask":
        raise ValueError(f"recipe {where}: only GT methods carry keypoint supervision")
    if spec.minimum_labels < 1 or any(v < 1 for v in overrides.values()):
        raise ValueError(f"recipe {where}: every minimum must be at least 1 (never zero)")
    scope = set(spec.objects if spec.objects is not None else objects)
    if not scope <= set(objects) or not set(overrides) <= scope:
        raise ValueError(f"recipe {where}: objects and minimum_overrides must be method objects")
    if spec.transductive != (spec.source == "test"):
        raise ValueError(f"recipe {where}: transductive is exactly the test source")
    if (spec.source == "combined") != bool(spec.components):
        raise ValueError(f"recipe {where}: components belong to a combined method only")
    return spec


def _adapt_spec(data: Any, objects: tuple[int, ...]) -> AdaptSpec | None:  # noqa: ANN401
    if data is None:
        return None
    values = _strict(AdaptSpec, data, "adapt")
    raw = values.get("methods") or {}
    multisensor = _multisensor_spec(values.get("multisensor"))
    if not isinstance(raw, Mapping) or not (raw or multisensor):
        raise ValueError(
            "recipe adapt.methods must name at least one method (or adapt.multisensor a stage)"
        )
    if not raw and values.get("config") is not None:
        raise ValueError("recipe adapt.config is the template of adapt.methods; name a method")
    if BASE_ARM in raw or MULTISENSOR_ARM in raw:
        raise ValueError(
            f"recipe adapt.methods must not name the {BASE_ARM!r} or "
            f"{MULTISENSOR_ARM!r} run directories"
        )
    methods = {
        str(name): _method_spec(spec, f"adapt.methods.{name}", objects)
        for name, spec in raw.items()
    }
    for name, spec in methods.items():
        kinds = [methods[c].source if c in methods else None for c in spec.components]
        if spec.components and sorted(map(str, kinds)) != sorted(COMPONENT_SOURCES):
            raise ValueError(
                f"recipe adapt.methods.{name}: components must be one static and one dynamic "
                "method of the table"
            )
    real_per_batch = int(values.get("real_per_batch", 4))
    if real_per_batch < 1:
        raise ValueError("recipe adapt.real_per_batch must be positive")
    return AdaptSpec(
        config=_adapt_config(values.get("config") or {}) if methods else {},
        real_per_batch=real_per_batch,
        gate=_section(GateSpec, values.get("gate"), "adapt.gate"),
        methods=methods,
        multisensor=multisensor,
    )


def _followup_teacher(spec: FollowupSpec, adapt: AdaptSpec, where: str) -> None:
    if spec.teacher == BASE_ARM:
        return
    method = adapt.methods.get(spec.teacher)
    if method is None or not method.applies_to(spec.obj_id) or method.transductive:
        raise ValueError(
            f"recipe {where}.teacher must be {BASE_ARM!r} or a non-transductive recipe method "
            f"of object {spec.obj_id}"
        )


def _followup_reuse(spec: FollowupSpec, adapt: AdaptSpec, where: str) -> None:
    if spec.reuse_labels_from is None:
        return
    method = adapt.methods.get(spec.reuse_labels_from)
    if (
        method is None
        or method.source != FOLLOWUP_SOURCE
        or not method.applies_to(spec.obj_id)
        or spec.teacher != BASE_ARM
        or spec.gate_overrides
    ):
        raise ValueError(
            f"recipe {where}.reuse_labels_from must name a {FOLLOWUP_SOURCE} recipe method of "
            f"object {spec.obj_id}, with the {BASE_ARM!r} teacher and no gate_overrides (the "
            "reused pool was labeled by the base at the recipe gate)"
        )


def _followup(
    name: str,
    data: Any,  # noqa: ANN401
    recipe_objects: tuple[int, ...],
    adapt: AdaptSpec | None,
) -> FollowupSpec:
    where = f"followups.{name}"
    if not _NAME_TOKEN.fullmatch(name):
        raise ValueError(f"recipe {where}: the name must match [a-z0-9_]+")
    if adapt is None or not adapt.methods:
        raise ValueError(
            f"recipe {where} needs adapt.methods: follow-up rounds run the shared adaptation "
            "template (adapt.config)"
        )
    if name in adapt.methods or name == MULTISENSOR_ARM or name in RESERVED_RUN_DIRS:
        raise ValueError(f"recipe {where}: {name!r} is a recipe arm or reserved run directory")
    values = _strict(FollowupSpec, data, where)
    gate = dict(values.get("gate_overrides") or {})
    values["gate_overrides"] = gate
    spec = FollowupSpec(**values)
    if spec.obj_id not in recipe_objects:
        raise ValueError(f"recipe {where}.obj_id must be a recipe object")
    unknown = sorted(set(gate) - {f.name for f in fields(GateSpec)})
    if unknown or not all(isinstance(v, (int, float)) for v in gate.values()):
        raise ValueError(f"recipe {where}.gate_overrides must set numeric GateSpec fields")
    if spec.minimum_labels < 1:
        raise ValueError(f"recipe {where}.minimum_labels must be at least 1 (never zero)")
    if not 1 <= spec.rounds <= MAX_FOLLOWUP_ROUNDS:
        raise ValueError(f"recipe {where}.rounds must be 1..{MAX_FOLLOWUP_ROUNDS}")
    if spec.target_labels < 1:
        raise ValueError(f"recipe {where}.target_labels must be positive")
    _followup_teacher(spec, adapt, where)
    _followup_reuse(spec, adapt, where)
    return spec


def _followups(
    data: Any,  # noqa: ANN401
    objects: tuple[int, ...],
    adapt: AdaptSpec | None,
) -> dict[str, FollowupSpec]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("recipe followups must be a mapping")
    return {str(name): _followup(str(name), spec, objects, adapt) for name, spec in data.items()}


def followup_runs(recipe: Recipe) -> dict[str, FollowupSpec]:
    """Every follow-up round directory (``<followup>/roundNN``) and its follow-up."""
    return {
        f"{name}/{round_name(number)}": spec
        for name, spec in recipe.followups.items()
        for number in range(1, spec.rounds + 1)
    }


def _objects_subset(values: Any, objects: tuple[int, ...], where: str) -> tuple[int, ...]:  # noqa: ANN401
    chosen = tuple(int(v) for v in values)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(objects):
        raise ValueError(f"recipe {where} must list distinct recipe objects")
    return chosen


def _check_name(name: str, where: str) -> None:
    if not _NAME_TOKEN.fullmatch(name):
        raise ValueError(f"recipe {where}: the name must match [a-z0-9_]+")
    if name in RESERVED_TEST_NAMES:
        raise ValueError(f"recipe {where}: {name!r} is a reserved test directory name")


def _policy(name: str, data: Any, objects: tuple[int, ...]) -> Policy:  # noqa: ANN401
    where = f"test.policies.{name}"
    _check_name(name, where)
    values = _strict(Policy, data, where)
    if "name" in values:
        raise ValueError(f"recipe {where} has unknown keys ['name']")
    if values.get("objects") is not None:
        values["objects"] = _objects_subset(values["objects"], objects, f"{where}.objects")
    policy = Policy(name=name, **values)
    if name in BUILTIN_POLICIES:
        raise ValueError(f"recipe {where}: {BUILTIN_POLICIES} are built-in policies")
    if not _CSV_TOKEN.fullmatch(policy.csv_method):
        raise ValueError(f"recipe {where}.csv_method must be alphanumeric (the BOP rule)")
    if policy.fallback not in POLICY_FALLBACKS:
        raise ValueError(f"recipe {where}.fallback must be one of {POLICY_FALLBACKS}")
    if (policy.objects is not None) != (policy.fallback == "out_of_scope"):
        raise ValueError(f"recipe {where}: objects belong to an out_of_scope policy only")
    if not isinstance(policy.record_time, bool):
        raise ValueError(f"recipe {where}.record_time must be true or false")
    return policy


def _alias(name: str, data: Any, objects: tuple[int, ...]) -> AliasSpec:  # noqa: ANN401
    where = f"test.aliases.{name}"
    _check_name(name, where)
    values = _strict(AliasSpec, data, where)
    run = str(values.get("run", ""))
    parts = Path(run).parts
    if not run or Path(run).is_absolute() or ".." in parts:
        raise ValueError(f"recipe {where}.run must be a relative run directory")
    if parts == (BASE_ARM,):
        raise ValueError(f"recipe {where}.run must not claim the {BASE_ARM!r} candidate")
    return AliasSpec(run, _objects_subset(values.get("objects") or (), objects, f"{where}.objects"))


def _test_spec(data: Any, objects: tuple[int, ...]) -> TestSpec:  # noqa: ANN401
    values = _strict(TestSpec, data if data is not None else {}, "test")
    for key in ("policies", "aliases"):
        if not isinstance(values.get(key) or {}, Mapping):
            raise ValueError(f"recipe test.{key} must be a mapping")
    values["policies"] = {
        str(k): _policy(str(k), v, objects) for k, v in (values.get("policies") or {}).items()
    }
    values["aliases"] = {
        str(k): _alias(str(k), v, objects) for k, v in (values.get("aliases") or {}).items()
    }
    return TestSpec(**values)


def _check_test_tables(recipe: Recipe) -> None:
    """Check the policy and alias tables against the recipe's arms and follow-ups.

    Policies name recipe methods or follow-up rounds with unique CSV tokens; a follow-up round
    policy changes only the follow-up's object (out of scope). Aliases do not shadow arms.
    """
    methods = set(recipe.adapt.methods) if recipe.adapt is not None else set()
    if recipe.adapt is not None and recipe.adapt.multisensor is not None:
        methods.add(MULTISENSOR_ARM)
    rounds = followup_runs(recipe)
    tokens = [p.csv_method for p in recipe.test.policies.values()]
    tokens += [f"{recipe.test.method}{name}" for name in BUILTIN_POLICIES]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"recipe test.policies need unique CSV methods: {sorted(tokens)}")
    for policy in recipe.test.policies.values():
        where = f"recipe test.policies.{policy.name}"
        if policy.method in methods:
            continue
        followup = rounds.get(policy.method)
        if followup is None:
            raise ValueError(
                f"{where}: {policy.method!r} is not a recipe method or follow-up round"
            )
        if policy.fallback != "out_of_scope" or policy.objects != (followup.obj_id,):
            raise ValueError(
                f"{where}: a follow-up round policy is out_of_scope for exactly object "
                f"{followup.obj_id}"
            )
    for name in recipe.test.aliases:
        if name in methods or name == BASE_ARM:
            raise ValueError(f"recipe test.aliases.{name} shadows a recipe arm")


def recipe_arms(recipe: Recipe) -> tuple[str, ...]:
    """Every arm of a recipe: ``base`` and its adaptation methods, in table order."""
    return (BASE_ARM, *(recipe.adapt.methods if recipe.adapt is not None else ()))


def _generated(data: Any) -> dict[int, GeneratedSpec]:  # noqa: ANN401
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("recipe generated must map object ids to render campaigns")
    campaigns = {}
    for key, value in data.items():
        where = f"generated.{key}"
        spec = GeneratedSpec(**_strict(GeneratedSpec, value, where))
        if spec.empty_masks not in EMPTY_MASK_POLICIES:
            raise ValueError(f"recipe {where}.empty_masks must be one of {EMPTY_MASK_POLICIES}")
        sizes = (spec.layouts, spec.frames_per_layout, spec.val_modulo, spec.conversion_shards)
        if min(sizes) < 1 or spec.base_seed < 0:
            raise ValueError(
                f"recipe {where}: layouts, views, the modulo and the shards must be positive"
            )
        if not 0 <= spec.val_residue < spec.val_modulo:
            raise ValueError(f"recipe {where}: val_residue must lie in [0, val_modulo)")
        campaigns[int(key)] = spec
    return campaigns


def _check_convert(spec: ConvertSpec, where: str) -> None:
    if spec.split not in SPLIT_POLICIES:
        raise ValueError(f"recipe {where}.split must be one of {SPLIT_POLICIES}")
    if spec.empty_masks not in EMPTY_MASK_POLICIES:
        raise ValueError(f"recipe {where}.empty_masks must be one of {EMPTY_MASK_POLICIES}")
    if spec.split == "layout_modulo":
        raise ValueError(f"recipe {where}.split layout_modulo is for generated objects only")


def _multiview_spec(data: Any) -> MultiviewSpec | None:  # noqa: ANN401
    if data is None:
        return None
    values = _strict(MultiviewSpec, data, "multiview")
    if "gate" in values:
        # A recipe's gate overrides DEFAULT_GATE field by field (e.g. only ``pnp_method``).
        values["gate"] = replace(
            DEFAULT_GATE, **_strict(AssociationGate, values["gate"], "multiview.gate")
        )
    if values.get("scenes") is not None:
        values["scenes"] = tuple(int(v) for v in values["scenes"])
    if "streams" in values:
        values["streams"] = tuple(str(v) for v in values["streams"])
    return MultiviewSpec(**values)


def _objects(values: Any, where: str) -> tuple[int, ...]:  # noqa: ANN401
    objects = tuple(int(v) for v in values)
    if not objects or len(set(objects)) != len(objects) or min(objects) < 1:
        raise ValueError(f"recipe {where} must list distinct positive object ids")
    return objects


def parse_recipe(data: Mapping[str, Any], source: Path | None = None) -> Recipe:
    """Build and validate a recipe from its parsed YAML mapping.

    Args:
        data: Recipe mapping; an explicit ``kind: bop`` is accepted.
        source: The recipe file, if any (profile paths resolve beside it).

    Returns:
        The recipe.

    Raises:
        ValueError: For a non-BOP ``kind``, unknown keys, invalid objects or a training config
            that does not resolve into a ``DoperConfig``.
    """
    kind = data.get("kind", "bop") if isinstance(data, Mapping) else "bop"
    if kind != "bop":
        raise ValueError(
            f"recipe is a {kind!r} recipe, not a BOP recipe; load it with "
            "doper.recipes.load_any_recipe"
        )
    if isinstance(data, Mapping):
        data = {key: value for key, value in data.items() if key != "kind"}
    values = _strict(Recipe, data, "top level")
    if "source" in values:
        raise ValueError("recipe top level has unknown keys ['source']")
    objects = _objects(values["objects"], "objects")
    campaigns = _generated(values.get("generated"))
    listed = values.get("generated_objects")
    generated = tuple(int(v) for v in listed) if listed is not None else tuple(campaigns)
    if listed is not None and campaigns and set(generated) != set(campaigns):
        raise ValueError("recipe generated_objects must name exactly the generated campaigns")
    if not set(generated) <= set(objects):
        raise ValueError("recipe generated_objects must be recipe objects")
    adapt = _adapt_spec(values.get("adapt"), objects)
    recipe = Recipe(
        name=str(values["name"]),
        profile=str(values["profile"]),
        objects=objects,
        generated_objects=generated,
        prepare=_section(PrepareSpec, values.get("prepare"), "prepare"),
        keypoints=_section(KeypointSpec, values.get("keypoints"), "keypoints"),
        convert=_section(ConvertSpec, values.get("convert"), "convert"),
        base=_train_spec(values["base"], "base"),
        validation=_section(ValidationSpec, values.get("validation"), "validation"),
        test=_test_spec(values.get("test"), objects),
        adapt=adapt,
        followups=_followups(values.get("followups"), objects, adapt),
        generated=campaigns,
        multiview=_multiview_spec(values.get("multiview")),
        refine=_section(RefineSpec, values.get("refine"), "refine"),
        source=source,
    )
    _check_test_tables(recipe)
    _check_convert(recipe.convert, "convert")
    if recipe.keypoints.count != recipe.base.config.get("num_keypoints", recipe.keypoints.count):
        raise ValueError("recipe keypoints.count differs from base.config.num_keypoints")
    resolve_config(recipe.base, dict.fromkeys(PATH_KEYS, PLACEHOLDER_ROOT))
    if recipe.adapt is not None:
        from ..training.adaptation import ADAPT_PATH_KEYS, adapt_config

        placeholders: dict[str, str] = dict.fromkeys(ADAPT_PATH_KEYS, PLACEHOLDER_ROOT)
        for method in recipe.adapt.methods:
            adapt_config(recipe, method, 1, placeholders)
        for name, followup in recipe.followups.items():
            adapt_config(recipe, name, 1, placeholders, spec=followup.method())
        if recipe.adapt.multisensor is not None:
            from ..training.adaptation import multisensor_config

            sensors = (PLACEHOLDER_ROOT,) * MULTISENSOR_SOURCES
            multisensor_config(recipe, {**placeholders, "sensors": sensors})
    return recipe


def recipe_path(name_or_path: str | Path) -> Path:
    """Resolve a packaged recipe name or a recipe YAML path to the recipe file.

    Raises:
        FileNotFoundError: If neither a packaged recipe nor a file matches.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = _recipe_directory() / f"{name_or_path}.yaml"
        if not path.is_file() or path.stem == "train_example":
            raise FileNotFoundError(
                f"no recipe {str(name_or_path)!r}; packaged recipes: {packaged_recipes()}"
            )
    return path


def load_recipe(name_or_path: str | Path) -> Recipe:
    """Load a packaged BOP recipe by name (``handal``, ``hopev2``, ``hot3d``, ``tless``) or a file.

    ``doper.recipes.load_any_recipe`` also loads the SPEED+ and SHIRT recipes.

    Args:
        name_or_path: Recipe name or YAML path.

    Returns:
        The validated recipe.

    Raises:
        FileNotFoundError: If neither a packaged recipe nor a file matches.
        ValueError: If the recipe has unknown keys or does not resolve into a ``DoperConfig``.
    """
    path = recipe_path(name_or_path)
    data = yaml.safe_load(path.read_text())
    return parse_recipe(data, source=path.resolve())


def recipe_objects(
    recipe: Recipe, requested: Sequence[int] | None, *, generated: bool = False
) -> tuple[int, ...]:
    """Return the requested objects (default: every trainable one).

    Objects with official PBR are always accepted. A generated object (no official PBR) is
    accepted only with ``generated`` (its data comes from a render root: ``prepare
    --generated-root``, or an object prepared from one) and a render campaign in the recipe.

    Args:
        recipe: Dataset recipe.
        requested: Object ids; None or empty for the default.
        generated: Accept generated objects with a render campaign; the default then also
            includes them.

    Raises:
        ValueError: For an object outside the recipe, or a generated object that is not
            accepted.
    """
    campaigns = [o for o in recipe.generated_objects if o in recipe.generated]
    if not requested:
        accepted = set(recipe.trainable_objects) | (set(campaigns) if generated else set())
        return tuple(o for o in recipe.objects if o in accepted)
    unknown = sorted(set(requested) - set(recipe.objects))
    if unknown:
        raise ValueError(f"objects {unknown} are not {recipe.name} objects")
    refused = sorted(
        o for o in set(requested) & set(recipe.generated_objects)
        if not generated or o not in campaigns
    )  # fmt: skip
    if refused:
        missing = [o for o in refused if o not in campaigns]
        how = (
            f"; {missing} have no render campaign in the recipe"
            if missing
            else "; prepare them from their render root with `doper prepare --generated-root`"
        )
        raise ValueError(
            f"objects {refused} have no official PBR; their base needs generated renders{how}"
        )
    return tuple(sorted(set(requested)))
