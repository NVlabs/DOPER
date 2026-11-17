# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The COLMAP 3.13 command-line runner and the selection of the largest sparse model.

``ColmapCli`` runs the ``colmap`` binary (user-installed; not a Python dependency) once per
step, with the flags of the recipe's ``colmap`` section (``DEFAULT_FLAGS`` are the template
recipe's, ``examples/nurec/recipe.yaml``). Every call's exact argv is recorded in ``calls`` (and
ends up in ``reconstruction.json``); with ``log_dir`` each call's output goes to
``NN-<command>.log``. A failed call raises ``ColmapError`` with the command line and the tail of
its log.

The flag defaults reproduce the reference COLMAP stage and reconstruction: CPU
SIFT (affine shape and domain-size pooling, which COLMAP 3.13 only runs on the CPU) with 16384
features, guided exhaustive matching with 65536 matches, and, whenever the capture gives the
intrinsics, the camera fixed: the extractor gets ``--ImageReader.camera_params``, the
triangulator ``--refine_intrinsics 0``, the bundle adjuster, the mapper and the image
registrator do not refine focal length, principal point or distortion. The bundle adjuster
refines the rig (camera) poses and not the sensor-from-rig transforms.

``FeatureExtraction.num_threads 1``: threaded extraction writes the images to the database in a
nondeterministic order, which changes the reconstruction and its scale from run to run;
single-threaded extraction makes reruns identical. Set the key to -1 (COLMAP's default, every
core) for threaded extraction.

Option names are COLMAP 3.13's (``FeatureExtraction.*``, ``FeatureMatching.*``,
``BundleAdjustment.refine_rig_from_world`` were renamed or added after 3.11), so ``version``
refuses any other release.
"""

from __future__ import annotations

import re
import stat
import struct
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..nurec.capture import PinholeCamera

COLMAP_ENV = "DOPER_COLMAP"
REQUIRED_VERSION = "3.13"
_VERSION = re.compile(r"COLMAP (\d+\.\d+\S*)")
_SUPPORTED = re.compile(r"3\.13(\.\d+)?")
LOG_TAIL_LINES = 20
# Parameter count of the camera models a rectified pinhole maps onto (zero distortion).
PINHOLE_PARAMS = {"PINHOLE": 4, "OPENCV": 8}
DEFAULT_FLAGS: dict[str, dict[str, Any]] = {
    "feature_extractor": {
        "ImageReader.camera_model": "OPENCV",
        "ImageReader.single_camera": 1,
        "FeatureExtraction.use_gpu": 0,
        "FeatureExtraction.num_threads": 1,
        "SiftExtraction.max_num_features": 16384,
        "SiftExtraction.estimate_affine_shape": 1,
        "SiftExtraction.domain_size_pooling": 1,
    },
    "exhaustive_matcher": {
        "FeatureMatching.use_gpu": 1,
        "FeatureMatching.guided_matching": 1,
        "FeatureMatching.max_num_matches": 65536,
    },
    "point_triangulator": {"refine_intrinsics": 0},
    "bundle_adjuster": {
        "BundleAdjustment.refine_focal_length": 0,
        "BundleAdjustment.refine_principal_point": 0,
        "BundleAdjustment.refine_extra_params": 0,
        "BundleAdjustment.refine_rig_from_world": 1,
        "BundleAdjustment.refine_sensor_from_rig": 0,
    },
}
# Added to the mapper's flags when the intrinsics are known (``map(fixed_intrinsics=True)``).
FIXED_MAPPER_FLAGS: dict[str, Any] = {
    "Mapper.ba_refine_focal_length": 0,
    "Mapper.ba_refine_principal_point": 0,
    "Mapper.ba_refine_extra_params": 0,
}


class ColmapError(OSError):
    """A COLMAP call failed, or the binary is missing or the wrong release.

    An ``OSError``: like a missing file, it is a failure of the environment the command runs
    in, which ``doper`` reports as an error message rather than a traceback.
    """


@dataclass(frozen=True)
class ColmapCall:
    """One recorded COLMAP call.

    Attributes:
        command: The COLMAP command (``feature_extractor``, ...).
        argv: The exact argv, the binary first.
        returncode: Exit code.
        log: The call's log file, when the runner writes logs.
    """

    command: str
    argv: tuple[str, ...]
    returncode: int
    log: Path | None = None


def _value(value: Any) -> str:  # noqa: ANN401
    """A flag value as COLMAP reads it (booleans as 0/1)."""
    if isinstance(value, bool):
        return str(int(value))
    return repr(value) if isinstance(value, float) else str(value)


def _flag_args(flags: Mapping[str, Any]) -> list[str]:
    return [item for key, value in flags.items() for item in (f"--{key}", _value(value))]


def _camera_params(model: str, camera: PinholeCamera) -> str:
    """``--ImageReader.camera_params`` of a rectified pinhole (distortion written as 0).

    Raises:
        ValueError: For a camera model a zero-distortion pinhole does not map onto.
    """
    if model not in PINHOLE_PARAMS:
        raise ValueError(f"known intrinsics need camera model {sorted(PINHOLE_PARAMS)}, got "
                         f"{model!r}")  # fmt: skip
    values = [repr(v) for v in (camera.fx, camera.fy, camera.cx, camera.cy)]
    return ",".join(values + ["0"] * (PINHOLE_PARAMS[model] - 4))


def _tail(text: str) -> str:
    return "\n".join(text.rstrip("\n").split("\n")[-LOG_TAIL_LINES:])


class ColmapCli:
    """Runs COLMAP 3.13 commands through its command-line binary.

    Args:
        binary: The ``colmap`` executable (name on ``PATH`` or a path).
        flags: COLMAP command name to its ``--Section.option`` flags (names without dashes):
            the recipe's ``colmap`` section. Commands it does not name run with COLMAP's
            defaults.
        log_dir: Directory for one log per call; None keeps the output in memory only.
    """

    def __init__(
        self,
        binary: str = "colmap",
        flags: Mapping[str, Mapping[str, Any]] | None = None,
        log_dir: Path | None = None,
    ) -> None:
        """Keep the binary, the flags (the defaults when None) and the log directory."""
        self.binary = binary
        self.flags = {k: dict(v) for k, v in (DEFAULT_FLAGS if flags is None else flags).items()}
        self.log_dir = log_dir
        self.calls: list[ColmapCall] = []

    def _command_flags(self, command: str) -> dict[str, Any]:
        return dict(self.flags.get(command, {}))

    def _run(self, command: str, *args: str | Path) -> str:
        """Run one command; record it and return its output.

        Raises:
            ColmapError: For a missing binary or a non-zero exit code.
        """
        argv = (self.binary, command, *(str(a) for a in args))
        log = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = self.log_dir / f"{len(self.calls) + 1:02d}-{command}.log"
        try:
            done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, check=False)  # fmt: skip
        except OSError as error:
            raise ColmapError(
                f"cannot run COLMAP binary {self.binary!r} ({error}): install COLMAP "
                f"{REQUIRED_VERSION} and put it on PATH or set {COLMAP_ENV}"
            ) from error
        if log is not None:
            log.write_text(done.stdout)
        self.calls.append(ColmapCall(command, argv, done.returncode, log))
        if done.returncode != 0:
            where = f" (log {log})" if log is not None else ""
            raise ColmapError(f"{' '.join(argv)} failed with exit code {done.returncode}"
                              f"{where}; log tail:\n{_tail(done.stdout)}")  # fmt: skip
        return done.stdout

    def version(self) -> str:
        """The binary's COLMAP release, which must be 3.13 (``3.13`` or ``3.13.N``).

        Raises:
            ColmapError: When the binary cannot run, reports no version, or is another release.
        """
        output = self._run("help")
        match = _VERSION.search(output)
        if match is None:
            raise ColmapError(f"{self.binary} help reports no COLMAP version:\n{_tail(output)}")
        version = match.group(1)
        if not _SUPPORTED.fullmatch(version):
            raise ColmapError(f"COLMAP {REQUIRED_VERSION}.x is required (its option names and "
                              f"model files), {self.binary} is COLMAP {version}")  # fmt: skip
        return version

    def extract(self, db: Path, images: Path, camera: PinholeCamera | None) -> None:
        """``feature_extractor``; a known camera is fixed as the shared intrinsics.

        Args:
            db: Database to create.
            images: Image directory.
            camera: The images' rectified pinhole, or None to let COLMAP estimate it.
        """
        flags = self._command_flags("feature_extractor")
        if camera is not None:
            model = str(flags.get("ImageReader.camera_model", "SIMPLE_RADIAL"))
            items = list(flags.items())
            reader = [i for i, (key, _) in enumerate(items) if key.startswith("ImageReader.")]
            at = reader[-1] + 1 if reader else 0
            items.insert(at, ("ImageReader.camera_params", _camera_params(model, camera)))
            flags = dict(items)
        self._run("feature_extractor", "--database_path", db, "--image_path", images,
                  *_flag_args(flags))  # fmt: skip

    def match(self, db: Path) -> None:
        """``exhaustive_matcher`` over every image pair of the database."""
        self._run("exhaustive_matcher", "--database_path", db,
                  *_flag_args(self._command_flags("exhaustive_matcher")))  # fmt: skip

    def triangulate(self, db: Path, images: Path, seed: Path, out: Path) -> None:
        """``point_triangulator``: triangulate the matches with the seed model's poses."""
        out.mkdir(parents=True, exist_ok=True)
        self._run("point_triangulator", "--database_path", db, "--image_path", images,
                  "--input_path", seed, "--output_path", out,
                  *_flag_args(self._command_flags("point_triangulator")))  # fmt: skip

    def bundle_adjust(self, model: Path, out: Path) -> None:
        """``bundle_adjuster`` of a model into ``out``."""
        out.mkdir(parents=True, exist_ok=True)
        self._run("bundle_adjuster", "--input_path", model, "--output_path", out,
                  *_flag_args(self._command_flags("bundle_adjuster")))  # fmt: skip

    def register(self, db: Path, model: Path, out: Path, *, fixed_intrinsics: bool = False) -> None:
        """``image_registrator``: register the database's other images into a model.

        Args:
            db: Database with the features and matches.
            model: The model to register into.
            out: Output model directory.
            fixed_intrinsics: Do not refine focal length, principal point or distortion (its
                mapper options, as ``map``).
        """
        out.mkdir(parents=True, exist_ok=True)
        flags = self._command_flags("image_registrator")
        if fixed_intrinsics:
            flags.update(FIXED_MAPPER_FLAGS)
        self._run("image_registrator", "--database_path", db, "--input_path", model,
                  "--output_path", out, *_flag_args(flags))  # fmt: skip

    def map(self, db: Path, images: Path, out: Path, *, fixed_intrinsics: bool = False) -> None:
        """``mapper`` (incremental SfM) into ``out/<n>`` models.

        Args:
            db: Database.
            images: Image directory.
            out: Sparse output directory.
            fixed_intrinsics: Do not refine focal length, principal point or distortion (the
                capture gives them).
        """
        out.mkdir(parents=True, exist_ok=True)
        flags = self._command_flags("mapper")
        if fixed_intrinsics:
            flags.update(FIXED_MAPPER_FLAGS)
        self._run("mapper", "--database_path", db, "--image_path", images, "--output_path", out,
                  *_flag_args(flags))  # fmt: skip

    def to_text(self, model: Path, out: Path) -> None:
        """``model_converter`` of a model to the text format."""
        out.mkdir(parents=True, exist_ok=True)
        self._run("model_converter", "--input_path", model, "--output_path", out,
                  "--output_type", "TXT")  # fmt: skip


def count_registered_images(model: Path) -> int:
    """Registered images of a COLMAP model directory.

    A binary model's ``images.bin`` starts with its registered image count (uint64,
    little-endian); a text model's ``images.txt`` lists the registered images.

    Raises:
        FileNotFoundError: For a directory with neither file.
    """
    binary = model / "images.bin"
    if binary.is_file():
        with binary.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"{binary} is truncated")
        return int(struct.unpack("<Q", header)[0])
    text = model / "images.txt"
    if text.is_file():
        from .colmap_io import read_text_model

        return len(read_text_model(model).images)
    raise FileNotFoundError(f"{model} has neither images.bin nor images.txt")


def select_largest_model(sparse_dir: Path) -> Path:
    """The mapper's model with the most registered images; ties go to the lowest model id.

    ``sparse_dir`` must hold only numbered model directories.

    Raises:
        ValueError: For an unexpected entry, no model, or no registered image.
    """
    models = []
    for entry in sorted(Path(sparse_dir).iterdir()):
        mode = entry.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or not entry.name.isdigit():
            raise ValueError(f"unexpected entry {entry.name!r} in sparse directory {sparse_dir}")
        models.append((-count_registered_images(entry), int(entry.name), entry))
    if not models:
        raise ValueError(f"COLMAP mapper produced no sparse models in {sparse_dir}")
    negative, _, selected = min(models)
    if negative >= 0:
        raise ValueError(f"the largest sparse model in {sparse_dir} has no registered images")
    return selected


__all__ = [
    "COLMAP_ENV",
    "DEFAULT_FLAGS",
    "FIXED_MAPPER_FLAGS",
    "ColmapCall",
    "ColmapCli",
    "ColmapError",
    "count_registered_images",
    "select_largest_model",
]
