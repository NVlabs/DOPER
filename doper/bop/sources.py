# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned BOP dataset sources: profiles, verified download and ``READY.json`` receipts.

A profile pins one public ``bop-benchmark/*`` Hugging Face dataset at a full revision and lists
every archive with its SHA-256. :func:`fetch` downloads the requested parts, :func:`verify`
re-checks them, and :func:`write_ready` records a ``READY.json`` receipt that later steps demand
through :func:`require_ready`. Source defects (zero-byte HANDAL PBR members, the HOT3D depth and
mask defects) are profile data, not code. ``huggingface_hub`` is imported lazily by the default
downloader only, so importing this module stays cheap.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from ..console import log
from ..recipes import profile_directory, profile_path

READY_NAME = "READY.json"
SOURCE_DIR_NAME = "source"
VERIFIED_NAME = "verified.json"
LOCK_NAME = "stage.lock"
# The image files the BOP sources and staged views hold, in listing order (``.jpg`` first).
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".png")
HASH_BLOCK_BYTES = 8 * 1024 * 1024
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_KINDS = {
    "static": "onboarding_static",
    "dynamic": "onboarding_dynamic",
    "test": "unlabelled_test_bop",
}
TRANSDUCTIVE_PARTS = frozenset({"test"})
TEST_NOTICE = (
    "Transductive Test-Time Adaptation: unlabeled BOP test RGB/intrinsics "
    "used for adaptation; not held-out test performance; no test GT or score selection"
)
HOT3D_DEPTH_POLICY = "pinned_hot3d_required_pixel_defects_v1"
HOT3D_MASK_POLICY = "pinned_hot3d_object_mask_defects_v1"
HOT3D_OBJECTS = range(1, 34)
HOT3D_MAX_FRAME = 1000
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class SourceNotReadyError(RuntimeError):
    """Raised when a dataset root has no verified ``READY.json`` receipt for a part."""


class Downloader(Protocol):
    """Fetches one profile file into ``destination`` and returns its local path."""

    def __call__(self, hf_repo: str, filename: str, revision: str, destination: Path) -> Path:
        """Download ``filename`` of ``hf_repo`` at ``revision`` to ``destination / filename``."""
        ...


@dataclass(frozen=True)
class DatasetProfile:
    """One pinned dataset source.

    Attributes:
        name: Profile name, for example ``"hopev2"``.
        hf_repo: Hugging Face dataset repository.
        revision: Full 40-hex commit the files are pinned to.
        files: Relative path of every pinned file mapped to its SHA-256.
        parts: Named groups of ``files`` (``models``, ``pbr``, ``val``, ``test``, ...).
        exclusions: Files or archive members known to be defective. Entries that are also
            keys of ``files`` are not downloaded; every entry is recorded in the receipt.
        defects: Structured defect evidence (HANDAL frames, HOT3D depth and mask records).
        meta: Remaining pinned facts: sizes, counts, layout, resources, camera protocol.
        directory: Directory of the profile file; its ``meta["resources"]`` files resolve
            against it (default: the packaged profile directory).
    """

    name: str
    hf_repo: str
    revision: str
    files: dict[str, str]
    parts: dict[str, list[str]]
    exclusions: list[str] = field(default_factory=list)
    defects: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    directory: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate the pins so a malformed profile fails at load time."""
        if not REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError(f"profile {self.name}: revision must be a full 40-hex commit")
        for relative, digest in self.files.items():
            parts = Path(relative).parts
            if Path(relative).is_absolute() or ".." in parts or not parts:
                raise ValueError(f"profile {self.name}: unsafe file path {relative!r}")
            if not SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"profile {self.name}: bad SHA-256 for {relative}")
        for part, names in self.parts.items():
            unknown = sorted(set(names) - set(self.files))
            if unknown:
                raise ValueError(f"profile {self.name}: part {part} lists unpinned files {unknown}")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: Iterable[str | Path]) -> list[str]:
    """``sha256_file`` of each path, in order, hashed on a thread pool (hashlib frees the GIL).

    Args:
        paths: Files to hash.

    Returns:
        Lowercase hex digests, in the order of ``paths``.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as pool:
        return list(pool.map(sha256_file, paths))


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of a JSON value in canonical (sorted, compact) form.

    Args:
        value: JSON-serializable value.

    Returns:
        Lowercase hex digest.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def publish_json(path: str | Path, value: object) -> None:
    """Write JSON atomically (temporary file, fsync, rename).

    Args:
        path: Destination file; parent directories are created.
        value: JSON-serializable value.
    """
    publish_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def publish_text(path: str | Path, text: str) -> None:
    """Write text atomically (temporary file, fsync, rename), like ``publish_json``.

    The text is written verbatim as UTF-8 (no newline translation).

    Args:
        path: Destination file; parent directories are created.
        text: The file's content.
    """
    publish_bytes(path, text.encode())


def publish_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes atomically (temporary file, fsync, rename).

    Args:
        path: Destination file; parent directories are created.
        data: The file's content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(stream: IO[str], record: object) -> None:
    """Append one JSON line to an open text ``stream`` and make it durable (flush, fsync).

    Args:
        stream: A text stream opened for appending.
        record: JSON-serializable value (no NaN or infinity).
    """
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def json_roundtrip(value: object, *, allow_nan: bool = True) -> Any:  # noqa: ANN401
    """Return ``value`` as it reads back from JSON (tuples as lists, keys as strings).

    Args:
        value: JSON-serializable value.
        allow_nan: Accept NaN and infinity (``json.dumps``).

    Returns:
        The parsed ``json.dumps`` of ``value``.
    """
    return json.loads(json.dumps(value, allow_nan=allow_nan))


def check_keys(
    record: object,
    allowed: Iterable[str],
    required: Iterable[str],
    where: str,
    *,
    not_mapping: str | None = None,
) -> None:
    """Refuse a non-mapping ``record`` or one with unknown or missing keys.

    Args:
        record: Parsed JSON value to check.
        allowed: Every key the record may hold.
        required: Keys the record must hold.
        where: Prefix of the key error, ``f"{where} has unknown keys [...] / missing keys [...]"``.
        not_mapping: Error for a non-mapping record; ``f"{where} must be a JSON object"`` if None.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a JSON object" if not_mapping is None else not_mapping)
    unknown = sorted(set(record) - set(allowed))
    missing = sorted(set(required) - set(record))
    if unknown or missing:
        raise ValueError(f"{where} has unknown keys {unknown} / missing keys {missing}")


def sibling_lock(directory: str | Path) -> Path:
    """The lock file guarding ``directory``: ``.<name>.lock`` beside it.

    A sibling stays in place while the directory is removed or replaced, and its leading dot
    keeps it out of ``obj_*`` listings.
    """
    directory = Path(directory)
    return directory.with_name(f".{directory.name}.lock")


@contextmanager
def file_lock(path: str | Path, *, wait: bool = True, busy: str | None = None) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` (created with its parents).

    Args:
        path: The lock file.
        wait: Block until the lock is free, saying once that it waits; False refuses a held
            lock.
        busy: What a held lock means (default: another process holds ``path``).

    Raises:
        BlockingIOError: If the lock is held and ``wait`` is false.
        OSError: If the lock cannot be taken for another reason (naming ``path``).
    """
    path = Path(path)
    message = busy or f"another process holds {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as err:
                if not wait:
                    raise BlockingIOError(err.errno, message) from None
                log(f"waiting: {message}")
                fcntl.flock(lock, fcntl.LOCK_EX)
        except BlockingIOError:
            raise
        except OSError as err:
            raise OSError(err.errno, f"cannot lock {path}: {err.strerror}") from err
        yield


def link_relative(link: Path, target: Path) -> None:
    """Create ``link`` as a relative symlink to ``target``.

    Both ends are resolved first, so the link stays valid when the tree that holds both is
    moved as a whole, whatever the current directory or the symlinks above them.

    Args:
        link: Symlink to create; its parent must exist.
        target: Existing file or directory it points to.
    """
    target = Path(target).resolve()
    relative = os.path.relpath(target, Path(link).parent.resolve())
    Path(link).symlink_to(relative, target_is_directory=target.is_dir())


def load_profile(name_or_path: str | Path) -> DatasetProfile:
    """Load a packaged profile by name or a profile JSON by path.

    Args:
        name_or_path: A packaged name (``handal``, ``hopev2``, ``hot3d``, ``tless``) or the
            path of a profile file.

    Returns:
        The validated profile.

    Raises:
        FileNotFoundError: If neither a packaged profile nor a file matches.
    """
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = profile_path(str(name_or_path))
    if not candidate.is_file():
        packaged = sorted(p.stem for p in profile_directory().glob("*.json"))
        raise FileNotFoundError(f"no profile {str(name_or_path)!r}; packaged profiles: {packaged}")
    data = json.loads(candidate.read_text())
    return DatasetProfile(
        name=data["name"],
        hf_repo=data["hf_repo"],
        revision=data["revision"],
        files=dict(data["files"]),
        parts={part: list(names) for part, names in data["parts"].items()},
        exclusions=list(data.get("exclusions", [])),
        defects=dict(data.get("defects", {})),
        meta=dict(data.get("meta", {})),
        directory=candidate.parent,
    )


def load_resource(profile: DatasetProfile, key: str) -> dict[str, Any]:
    """Load a frozen staging resource shipped with the profile, checking its pinned SHA-256.

    Args:
        profile: Profile that lists ``key`` under ``meta["resources"]``.
        key: Resource name, for example ``"validation_panel"``.

    Returns:
        The parsed JSON content.

    Raises:
        KeyError: If the profile has no such resource.
        ValueError: If the shipped file does not match its pinned SHA-256.
    """
    entry = profile.meta.get("resources", {})[key]
    path = (profile.directory or profile_directory()) / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(
            f"resource {key} of profile {profile.name} differs from its pinned SHA-256"
        )
    return json.loads(path.read_text())


def source_dir(root: Path) -> Path:
    """Return the directory under ``root`` that holds the downloaded files."""
    return Path(root) / SOURCE_DIR_NAME


def part_files(profile: DatasetProfile, parts: Sequence[str]) -> list[str]:
    """Resolve part names to their files, in profile order without duplicates.

    Args:
        profile: Dataset profile.
        parts: Part names.

    Returns:
        Relative file paths.

    Raises:
        ValueError: If a part is unknown.
    """
    files: dict[str, None] = {}
    for part in parts:
        if part not in profile.parts:
            raise ValueError(
                f"profile {profile.name} has no part {part!r}; has {list(profile.parts)}"
            )
        files.update(dict.fromkeys(profile.parts[part]))
    return list(files)


def _skipped_files(profile: DatasetProfile, names: Sequence[str]) -> list[str]:
    excluded = set(profile.exclusions)
    return [name for name in names if name in excluded]


def require_transductive_consent(flag: bool) -> None:
    """Refuse transductive test adaptation unless the user opted in.

    Args:
        flag: The caller's consent (the ``doper prepare`` flag is ``--allow-test-inputs``).

    Raises:
        ValueError: If ``flag`` is false.
    """
    if not flag:
        raise ValueError(
            "the unlabeled test inputs require explicit consent to transductive use "
            "(allow_transductive=True)"
        )


def default_downloader(hf_repo: str, filename: str, revision: str, destination: Path) -> Path:
    """Download one file of a public dataset with ``huggingface_hub`` at the pinned revision.

    Args:
        hf_repo: Dataset repository.
        filename: Path inside the repository.
        revision: Pinned commit.
        destination: Directory that receives ``filename``.

    Returns:
        Local path of the downloaded file.

    Raises:
        ImportError: If ``huggingface_hub`` is missing; install ``doper[bop]``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            "huggingface_hub is required to download BOP datasets; install doper[bop]"
        ) from err
    return Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            token=False,
        )
    )


def _read_object(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _expected_bytes(profile: DatasetProfile, name: str) -> int | None:
    return profile.meta.get("bytes", {}).get(name)


def _file_problem(
    path: Path, profile: DatasetProfile, name: str, receipts: Mapping[str, Any]
) -> str | None:
    """Return why ``path`` does not match the pin, or None. A matching receipt skips hashing."""
    if not path.is_file():
        return f"missing: {name}"
    stat = path.stat()
    expected = _expected_bytes(profile, name)
    if expected is not None and stat.st_size != expected:
        return f"size mismatch: {name}: expected {expected} bytes, found {stat.st_size}"
    receipt = receipts.get(name)
    if (
        receipt
        and receipt.get("sha256") == profile.files[name]
        and receipt.get("bytes") == stat.st_size
        and receipt.get("mtime_ns") == stat.st_mtime_ns
    ):
        return None
    actual = sha256_file(path)
    if actual != profile.files[name]:
        return f"SHA-256 mismatch: {name}: expected {profile.files[name]}, got {actual}"
    return None


def fetch(
    profile: DatasetProfile,
    root: Path,
    parts: Sequence[str],
    *,
    downloader: Downloader | None = None,
    allow_transductive: bool = False,
) -> Path:
    """Download and SHA-256 check the requested parts under ``root``.

    Files already present and matching their pin are kept; excluded files are not downloaded.

    Args:
        profile: Dataset profile.
        root: Dataset root (always explicit; created if needed).
        parts: Part names to fetch.
        downloader: Download callable; defaults to ``huggingface_hub`` at the pinned revision.
        allow_transductive: Consent to download unlabeled test inputs for adaptation.

    Returns:
        The directory holding the downloaded files (``root/source``).

    Raises:
        ValueError: If a part is unknown, transductive consent is missing, or a downloaded
            file does not match its pinned SHA-256 (the corrupt file is removed).
    """
    names = part_files(profile, parts)
    if TRANSDUCTIVE_PARTS.intersection(parts):
        require_transductive_consent(allow_transductive)
    download = downloader or default_downloader
    root = Path(root)
    destination = source_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    skipped = set(_skipped_files(profile, names))
    with file_lock(root / LOCK_NAME):
        receipts = _read_object(root / VERIFIED_NAME) or {}
        for name in names:
            if name in skipped:
                continue
            target = destination / name
            if target.exists() and _file_problem(target, profile, name, receipts) is not None:
                target.unlink()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                downloaded = Path(download(profile.hf_repo, name, profile.revision, destination))
                if downloaded != target:
                    raise ValueError(f"downloader returned {downloaded}, expected {target}")
            problem = _file_problem(target, profile, name, receipts)
            if problem is not None:
                target.unlink(missing_ok=True)
                raise ValueError(problem)
            stat = target.stat()
            receipts[name] = {
                "sha256": profile.files[name],
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            publish_json(root / VERIFIED_NAME, receipts)
    return destination


def verify(root: Path, profile: DatasetProfile, parts: Sequence[str]) -> list[str]:
    """Check the downloaded files of the given parts against the profile.

    Args:
        root: Dataset root.
        profile: Dataset profile.
        parts: Part names.

    Returns:
        Human-readable problems; empty when every non-excluded file is present and matches.
    """
    root = Path(root)
    names = part_files(profile, parts)
    skipped = set(_skipped_files(profile, names))
    receipts = _read_object(root / VERIFIED_NAME) or {}
    problems = []
    for name in names:
        if name in skipped:
            continue
        problem = _file_problem(source_dir(root) / name, profile, name, receipts)
        if problem is not None:
            problems.append(problem)
    return problems


def write_ready(root: Path, profile: DatasetProfile, parts: Sequence[str]) -> dict[str, Any]:
    """Verify the parts and record them in ``root/READY.json``.

    Receipts accumulate: calling this for another part of the same profile adds to the file.

    Args:
        root: Dataset root.
        profile: Dataset profile.
        parts: Part names to record.

    Returns:
        The written receipt.

    Raises:
        ValueError: If verification finds problems or the root holds another profile or
            revision.
    """
    root = Path(root)
    problems = verify(root, profile, parts)
    if problems:
        raise ValueError("cannot write READY.json: " + "; ".join(problems))
    identity = {
        "dataset": profile.name,
        "hf_repo": profile.hf_repo,
        "revision": profile.revision,
        "profile_sha256": profile_sha256(profile),
    }
    receipt = _read_object(root / READY_NAME)
    if receipt is None:
        receipt = {**identity, "parts": {}}
    elif any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError("source identity changed; use an isolated root for another profile")
    for part in parts:
        names = part_files(profile, [part])
        skipped = _skipped_files(profile, names)
        receipt["parts"][part] = {
            "files": {
                name: {
                    "sha256": profile.files[name],
                    "bytes": (source_dir(root) / name).stat().st_size,
                }
                for name in names
                if name not in skipped
            },
            "skipped": skipped,
        }
    receipt["exclusions"] = sorted(profile.exclusions)
    receipt["defects_sha256"] = canonical_sha256(profile.defects)
    receipt["transductive_parts"] = sorted(TRANSDUCTIVE_PARTS.intersection(receipt["parts"]))
    publish_json(root / READY_NAME, receipt)
    return receipt


def profile_sha256(profile: DatasetProfile) -> str:
    """Return the identity digest that ``READY.json`` records for ``profile``."""
    return canonical_sha256(_profile_identity(profile))


def _profile_identity(profile: DatasetProfile) -> dict[str, Any]:
    return {
        "files": profile.files,
        "parts": profile.parts,
        "exclusions": profile.exclusions,
        "defects": profile.defects,
        "meta": profile.meta,
    }


def require_ready(root: Path, parts: Sequence[str]) -> dict[str, Any]:
    """Return the ``READY.json`` receipt, requiring every given part to be recorded and present.

    Args:
        root: Dataset root.
        parts: Part names the caller needs.

    Returns:
        The receipt.

    Raises:
        SourceNotReadyError: If the receipt is missing, lacks a part, or a recorded file is
            gone or has changed size.
    """
    root = Path(root)
    receipt = _read_object(root / READY_NAME)
    if receipt is None:
        raise SourceNotReadyError(
            f"{root} has no {READY_NAME}; fetch and verify the dataset parts first"
        )
    for part in parts:
        if part not in receipt["parts"]:
            raise SourceNotReadyError(f"{root} is not ready for part {part!r}")
        for name, record in receipt["parts"][part]["files"].items():
            path = source_dir(root) / name
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise SourceNotReadyError(f"{root}: recorded file {name} is missing or changed")
    return receipt


def clean_source_view(
    source: Path, campaign: Path, profile: DatasetProfile, obj_id: int | None = None
) -> Path:
    """Create one immutable view of HANDAL PBR without the corrupt frames in the profile.

    The official ZIP members for these frames have zero bytes. Original assets stay untouched
    (images are symlinked); whole frames, not GT instances, are dropped from the metadata so
    BOP instance and mask indexing stays correct.

    Args:
        source: Extracted ``train_pbr`` directory.
        campaign: Directory that receives the view and its lock.
        profile: HANDAL profile carrying ``exclusions`` and ``defects``.
        obj_id: Object whose frames with missing masks are also dropped, if any.

    Returns:
        The view directory.

    Raises:
        ValueError: If the source no longer matches the documented defects or a view with a
            different identity exists.
    """
    defects = profile.defects
    missing = [row for row in defects.get("missing_required_files", []) if row["obj_id"] == obj_id]
    frames = defects["excluded_frames"] + [
        {"scene": row["scene"], "frame": row["frame"]} for row in missing
    ]
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    identity = {
        # Relative to the campaign directory, so moving the dataset root keeps the identity.
        "source": Path(os.path.relpath(Path(source).resolve(), campaign.resolve())).as_posix(),
        "exclusions_sha256": canonical_sha256([profile.exclusions, defects]),
        "missing_mask_object": obj_id if missing else None,
    }
    view = campaign / (f"source_view_obj{obj_id:06d}" if missing else "source_view")
    with file_lock(campaign / f"{view.name}.lock"):
        if view.exists():
            if json.loads((view / "identity.json").read_text()) != identity:
                raise ValueError(
                    f"source view identity changed: {view} was built from other data or by "
                    "another release; delete it and rerun `doper prepare` to rebuild it"
                )
            return view
        _check_documented_defects(source, profile, missing)
        stage = campaign / f".{view.name}-{os.getpid()}-{time.time_ns()}"
        stage.mkdir()
        excluded = {(row["scene"], str(row["frame"])) for row in frames}
        hashes = _write_filtered_scenes(source, stage, excluded)
        publish_json(stage / "identity.json", identity)
        publish_json(
            stage / "exclusions.json",
            {
                "reason": defects["reason"],
                "zero_byte_required_files": list(profile.exclusions),
                "excluded_frames": frames,
                "missing_required_files": missing,
                "original_metadata_sha256": hashes,
            },
        )
        os.rename(stage, view)
        return view


def _check_documented_defects(
    source: Path, profile: DatasetProfile, missing: Sequence[Mapping[str, Any]]
) -> None:
    for name in profile.exclusions:
        path = source / name.split("/", 1)[1]
        if not path.is_file() or path.stat().st_size != 0:
            raise ValueError(f"source no longer matches documented zero-byte defect: {path}")
    for row in missing:
        if (source / row["path"].split("/", 1)[1]).exists():
            raise ValueError(f"source no longer matches documented missing mask: {row['path']}")


def _write_filtered_scenes(
    source: Path, stage: Path, excluded: set[tuple[str, str]]
) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for scene in sorted(source.iterdir()):
        if not (scene / "scene_gt.json").is_file():
            continue
        destination = stage / scene.name
        destination.mkdir()
        hashes[scene.name] = {}
        for name in ("scene_gt.json", "scene_camera.json"):
            data = json.loads((scene / name).read_text())
            hashes[scene.name][name] = sha256_file(scene / name)
            kept = {k: v for k, v in data.items() if (scene.name, str(int(k))) not in excluded}
            publish_json(destination / name, kept)
        for name in ("rgb", "depth", "mask_visib"):
            link_relative(destination / name, scene / name)
    return hashes


def hot3d_depth_defects(profile: DatasetProfile) -> dict[str, Any]:
    """Return the HOT3D required-pixel (depth) defect manifest after validating it.

    Args:
        profile: HOT3D profile.

    Returns:
        The manifest from ``profile.defects["depth"]``.

    Raises:
        ValueError: If the manifest is not the pinned policy or has unevidenced records.
    """
    data = profile.defects["depth"]
    if (
        data["dataset"] != "hot3d"
        or data["revision"] != profile.revision
        or data["policy"] != HOT3D_DEPTH_POLICY
        or data["excluded_frame_count"] != len(data["defective_frames"])
    ):
        raise ValueError("unknown HOT3D source-defect policy")
    keys = set()
    for row in data["defective_frames"]:
        key = (row["scene"], row["frame"])
        if (
            not re.fullmatch(r"\d{6}", row["scene"])
            or type(row["frame"]) is not int
            or not 0 <= row["frame"] < HOT3D_MAX_FRAME
            or any(type(obj) is not int or obj not in HOT3D_OBJECTS for obj in row["objects"])
            or row["objects"] != sorted(set(row["objects"]))
        ):
            raise ValueError("invalid defect scene/frame/object identifiers")
        if key in keys or not row["objects"] or not row["defects"]:
            raise ValueError("duplicate/empty defect record")
        keys.add(key)
        for defect in row["defects"]:
            expected = f"train_pbr/{row['scene']}/depth/{row['frame']:06d}.png"
            if (
                defect.get("file") != expected
                or defect.get("kind") != "depth"
                or defect.get("reason") != "zero_byte_official_archive_member"
                or defect.get("bytes") != 0
                or defect.get("CRC") != 0
            ):
                raise ValueError("only evidenced empty official depth is authorized")
    return data


def hot3d_defects_sha256(profile: DatasetProfile) -> str:
    """Return the canonical SHA-256 of the HOT3D depth defect manifest."""
    return canonical_sha256(profile.defects["depth"])


def hot3d_affected_objects(profile: DatasetProfile) -> set[int]:
    """Return the HOT3D objects that appear in a frame with a depth defect."""
    rows = hot3d_depth_defects(profile)["defective_frames"]
    return {obj for row in rows for obj in row["objects"]}


def hot3d_mask_records(profile: DatasetProfile, obj: int) -> list[dict[str, Any]]:
    """Return the validated mask-defect records of one HOT3D object.

    Args:
        profile: HOT3D profile.
        obj: Object id.

    Returns:
        Records (empty when the object has no known mask defect).

    Raises:
        ValueError: If the policy is unknown or a record is unevidenced or duplicated.
    """
    data = profile.defects["masks"]
    if (
        data["dataset"] != "hot3d"
        or data["revision"] != profile.revision
        or data["policy"] != HOT3D_MASK_POLICY
    ):
        raise ValueError("unknown HOT3D mask policy")
    records = data["objects"].get(str(obj), [])
    seen = set()
    for row in records:
        if (
            not re.fullmatch(r"\d{6}", row["scene"])
            or type(row["frame"]) is not int
            or not 0 <= row["frame"] < HOT3D_MAX_FRAME
            or type(row["instance"]) is not int
            or row["instance"] < 0
        ):
            raise ValueError("invalid mask defect identifiers")
        expected = (
            f"train_pbr/{row['scene']}/mask_visib/{row['frame']:06d}_{row['instance']:06d}.png"
        )
        empty = (
            row["reason"] == "zero_byte_official_archive_member"
            and row.get("bytes") == 0
            and row.get("CRC") == 0
            and row.get("sha256") == EMPTY_SHA256
        )
        missing = (
            row["reason"] == "missing_official_archive_member"
            and row.get("archive_present") is False
            and row.get("source_exists") is False
        )
        if (
            row["file"] != expected
            or not (empty or missing)
            or (row["scene"], row["frame"], row["instance"]) in seen
        ):
            raise ValueError("unevidenced or duplicate mask defect")
        seen.add((row["scene"], row["frame"], row["instance"]))
    return records


def hot3d_mask_generation(profile: DatasetProfile, obj: int) -> int:
    """Return the mask view generation of an object (1 unless the profile raises it)."""
    value = profile.defects["masks"].get("object_view_generations", {}).get(str(obj), 1)
    if type(value) is not int or value < 1:
        raise ValueError("mask view generation must be a positive integer")
    return value


def hot3d_mask_policy_sha256(profile: DatasetProfile, obj: int) -> str:
    """Return the identity of one object's mask policy.

    Other objects' newly discovered defects cannot change an existing view.
    """
    payload: dict[str, Any] = {
        "dataset": "hot3d",
        "revision": profile.revision,
        "object": obj,
        "records": hot3d_mask_records(profile, obj),
    }
    if hot3d_mask_generation(profile, obj) != 1:
        payload["view_generation"] = hot3d_mask_generation(profile, obj)
    return canonical_sha256(payload)


def hot3d_mask_exclusion_record(profile: DatasetProfile, obj: int) -> dict[str, Any]:
    """Return the exclusion record of an object with known mask defects.

    Raises:
        ValueError: If the object has no evidenced mask defect.
    """
    rows = hot3d_mask_records(profile, obj)
    if not rows:
        raise ValueError("no evidenced mask defect for object")
    affected = obj in hot3d_affected_objects(profile)
    parent = hot3d_depth_defects(profile)["defective_frames"] if affected else []
    generation = hot3d_mask_generation(profile, obj)
    record = {
        "dataset": "hot3d",
        "policy": HOT3D_MASK_POLICY,
        "object": obj,
        "mask_policy_sha256": hot3d_mask_policy_sha256(profile, obj),
        "parent_depth_policy_sha256": hot3d_defects_sha256(profile) if parent else None,
        "excluded_frames": parent + rows,
        "missing_required_files": [
            r["file"] for r in rows if r["reason"] == "missing_official_archive_member"
        ],
        "original_pixels_retained": True,
        "no_mask_fabrication": True,
        "scope": (
            "whole-frame omission ONLY for this object; preserve original BOP instance indices"
        ),
    }
    if generation != 1:
        record["view_generation"] = generation
    return record


def hot3d_exclusion_record(profile: DatasetProfile) -> dict[str, Any]:
    """Return the exclusion record of the shared depth defects."""
    data = hot3d_depth_defects(profile)
    return {
        "dataset": "hot3d",
        "policy": data["policy"],
        "defects_manifest_sha256": hot3d_defects_sha256(profile),
        "excluded_frames": data["defective_frames"],
        "missing_required_files": [],
        "original_pixels_retained": True,
        "no_depth_fabrication": True,
    }


def validate_hot3d_exclusions(
    profile: DatasetProfile, exclusions: Mapping[str, Any], ready: Mapping[str, Any], obj: int
) -> None:
    """Fail closed unless a prepared source carries the approved exclusions for ``obj``.

    Args:
        profile: HOT3D profile.
        exclusions: The source's ``exclusions.json`` content.
        ready: The source's ``READY.json`` content (``identity``, ``permitted_training_objects``).
        obj: Object about to be trained or evaluated.

    Raises:
        ValueError: If the exclusions are unapproved or the wrong filtered view is used.
    """
    if (
        exclusions.get("dataset") != "hot3d"
        or not isinstance(exclusions.get("excluded_frames"), list)
        or not isinstance(exclusions.get("missing_required_files"), list)
    ):
        raise ValueError("wrong source exclusion dataset")
    if hot3d_mask_records(profile, obj):
        if (
            exclusions != hot3d_mask_exclusion_record(profile, obj)
            or ready["identity"].get("mask_policy_sha256") != hot3d_mask_policy_sha256(profile, obj)
            or ready.get("permitted_training_objects") != [obj]
        ):
            raise ValueError("known HOT3D mask defect requires its object-scoped filtered source")
        return
    if ready.get("permitted_training_objects") is not None:
        raise ValueError("object-scoped source cannot be used for another object")
    if not exclusions.get("excluded_frames") and not exclusions.get("missing_required_files"):
        if obj in hot3d_affected_objects(profile):
            raise ValueError("known HOT3D depth defect requires the filtered source")
        return
    if exclusions != hot3d_exclusion_record(profile) or ready["identity"].get(
        "defects_manifest_sha256"
    ) != hot3d_defects_sha256(profile):
        raise ValueError("unapproved HOT3D source exclusions")
