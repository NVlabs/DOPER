# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, portable per-scene Blender environments; no renderer imports.

Publish dependencies and USD first, the scene record last, then seal a complete
collection. Readers bind the sealed manifest digest into render/resume identity.
Bundle files must remain immutable for the duration of a render attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from rendering.shard_io import load_shards, sha256_file

if TYPE_CHECKING:
    from rendering.config import RenderConfig

VERSION = 1
ADAPTER = "blender42-isaac601"


def canonical_bytes(value: object) -> bytes:
    """Serialize ``value`` to canonical JSON bytes (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_hash(value: object) -> str:
    """Return the SHA256 hex digest of the canonical JSON form of ``value``."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _digest(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256 digest")
    return value


def _index(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("environment index must be a nonnegative integer")
    return value


def bundle_path(root: str | Path, name: str) -> Path:
    """Resolve ``name`` inside ``root``; only normalized relative files inside it are accepted."""
    if not isinstance(name, str):
        raise ValueError("invalid bundle path")
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or relative.as_posix() != name
        or ".." in relative.parts
        or "\\" in name
    ):
        raise ValueError(f"invalid bundle path: {name!r}")
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"bundle path escapes root: {name!r}")
    return path


def publish_bytes(path: str | Path, data: bytes) -> None:
    """Atomic no-clobber publication, safe for concurrent identical writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            temporary.chmod(0o644)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != data:
                    raise ValueError(f"existing environment artifact differs: {path}") from None
        finally:
            temporary.unlink(missing_ok=True)


def _source(source: object) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("invalid environment source")
    for name in ("identity_sha256", "usd_sha256", "assets_sha256"):
        _digest(source.get(name))
    if not isinstance(source.get("blender_version"), str) or not source[
        "blender_version"
    ].startswith("4.2."):
        raise ValueError("environment adapter requires Blender 4.2.x")
    return source


def source_from_manifest(manifest: dict[str, Any], blender_version: str) -> dict[str, Any]:
    """Build the bundle source identity from a shard manifest and its Blender version."""
    identity = manifest["render_identity"]
    return _source(
        {
            "identity_sha256": canonical_hash(identity),
            "usd_sha256": identity["object"]["usd_sha256"],
            "assets_sha256": identity["assets"]["content_sha256"],
            "blender_version": blender_version,
        }
    )


def publish_record(
    root: str | Path,
    record: dict[str, Any],
    *,
    source: dict[str, Any],
    layer: str,
    dependencies: list[str],
    light_count: int,
    ground: bool,
) -> dict[str, Any]:
    """Publish the metadata record of one scene after its layer and dependencies."""
    index = _index(record["index"])
    if type(light_count) is not int or light_count < 1 or type(ground) is not bool:
        raise ValueError("invalid realized environment counts")
    if not isinstance(layer, str) or not re.fullmatch(r"layers/[0-9a-f]{64}\.usda", layer):
        raise ValueError("invalid content-addressed environment layer path")
    files = {name: sha256_file(bundle_path(root, name)) for name in [layer, *dependencies]}
    if files[layer] != Path(layer).stem:
        raise ValueError("environment layer content hash mismatch")
    payload = {
        "version": VERSION,
        "scene_sha256": canonical_hash(record),
        "source": _source(source),
        "layer": layer,
        "files": files,
        "light_count": light_count,
        "ground": ground,
    }
    publish_bytes(bundle_path(root, f"records/{index:06d}.json"), canonical_bytes(payload))
    return payload


def _record(
    root: str | Path, index: int, source: dict[str, Any], expected_hash: str | None = None
) -> dict[str, Any]:
    path = bundle_path(root, f"records/{index:06d}.json")
    if not path.is_file():
        raise ValueError(f"missing environment record {index}")
    raw = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"environment metadata hash mismatch: {index}")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != VERSION
    ):
        raise ValueError("unsupported environment record version")
    if value.get("source") != source:
        raise ValueError("environment source mismatch")
    if not isinstance(value.get("layer"), str) or not re.fullmatch(
        r"layers/[0-9a-f]{64}\.usda", value["layer"]
    ):
        raise ValueError("invalid content-addressed environment layer path")
    if not isinstance(value.get("files"), dict) or value["layer"] not in value["files"]:
        raise ValueError("environment files are missing")
    if value["files"][value["layer"]] != Path(value["layer"]).stem:
        raise ValueError("environment layer content hash mismatch")
    _digest(value.get("scene_sha256"))
    if (
        type(value.get("light_count")) is not int
        or value["light_count"] < 1
        or type(value.get("ground")) is not bool
    ):
        raise ValueError("invalid realized environment counts")
    for name, digest in value["files"].items():
        bundle_path(root, name)
        _digest(digest)
    return value


def seal_bundle(
    root: str | Path, records: list[dict[str, Any]], *, source: dict[str, Any] | None
) -> dict[str, Any]:
    """Publish an index only after every requested scene and dependency verifies."""
    source = _source(source)
    entries, seed, verified = {}, None, set()
    for record in records:
        index = _index(record["index"])
        if str(index) in entries:
            raise ValueError(f"duplicate environment index {index}")
        if type(record.get("seed")) is not int:
            raise ValueError("invalid environment seed")
        if seed is not None and seed != record["seed"]:
            raise ValueError("mixed environment seeds")
        seed = record["seed"]
        payload = _record(root, index, source)
        if payload["scene_sha256"] != canonical_hash(record):
            raise ValueError(f"environment scene mismatch: {index}")
        _verify_files(root, payload["files"], verified)
        entries[str(index)] = sha256_file(Path(root) / f"records/{index:06d}.json")
    if not entries:
        raise ValueError("cannot seal an empty environment bundle")
    manifest = {"version": VERSION, "source": source, "seed": seed, "records": entries}
    publish_bytes(bundle_path(root, "manifest.json"), canonical_bytes(manifest))
    return manifest


def _verify_files(root: str | Path, files: dict[str, str], verified: set) -> None:
    for name, digest in files.items():
        if (name, digest) not in verified:
            if sha256_file(bundle_path(root, name)) != digest:
                raise ValueError(f"environment file hash mismatch: {name}")
            verified.add((name, digest))


@dataclass
class BundleReader:
    """Verified read access to a sealed environment bundle."""

    root: Path
    usd_sha256: str
    seed: int
    _verified: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load and validate the sealed manifest against the object digest and seed."""
        self.root = Path(self.root).resolve()
        raw = bundle_path(self.root, "manifest.json").read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        m = self.manifest
        if not isinstance(m, dict) or type(m.get("version")) is not int or m["version"] != VERSION:
            raise ValueError("unsupported environment bundle version")
        self.source = _source(m.get("source"))
        if (
            self.source["usd_sha256"] != self.usd_sha256
            or type(m.get("seed")) is not int
            or m["seed"] != self.seed
        ):
            raise ValueError("environment object or seed mismatch")
        if not isinstance(m.get("records"), dict) or not m["records"]:
            raise ValueError("environment records missing")
        for index, digest in m["records"].items():
            if not isinstance(index, str) or not re.fullmatch("0|[1-9][0-9]*", index):
                raise ValueError("invalid environment index")
            _digest(digest)

    def verify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Verify a scene record and its files against the sealed manifest."""
        index = _index(record["index"])
        digest = self.manifest["records"].get(str(index))
        if digest is None:
            raise ValueError(f"missing environment index {index}")
        value = _record(self.root, index, self.source, digest)
        if value["scene_sha256"] != canonical_hash(record):
            raise ValueError(f"environment scene mismatch: {index}")
        _verify_files(self.root, value["files"], self._verified)
        return value


def replay_options(
    root: str | Path | None, gain: float | None, adapter: str | None
) -> dict[str, Any] | None:
    """Validate explicit replay options; ``None`` when no bundle is requested."""
    if root is None:
        if gain is not None or adapter is not None:
            raise ValueError("environment gain/adapter require --environment-root")
        return None
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not math.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("environment gain must be explicit, finite and positive")
    if adapter != ADAPTER:
        raise ValueError(f"environment adapter must be explicitly {ADAPTER}")
    return {"adapter": adapter, "gain": float(gain)}


def verify_unsealed_record(
    root: str | Path,
    record: dict[str, Any],
    source: dict[str, Any],
    *,
    verified: set | None = None,
) -> dict[str, Any]:
    """Verify a published, not yet sealed, record and return its metadata."""
    payload = _record(root, _index(record["index"]), source)
    if payload["scene_sha256"] != canonical_hash(record):
        raise ValueError(f"environment scene mismatch: {record['index']}")
    _verify_files(root, payload["files"], set() if verified is None else verified)
    return payload


def seal_dataset(dataset: str | Path) -> dict[str, Any]:
    """Seal environment exports from completed, contiguous Blender shards."""
    dataset = Path(dataset)
    shards = sorted(load_shards(dataset), key=lambda item: item["shard"]["start_idx"])
    records, source, end = [], None, 0
    for shard in shards:
        if (
            shard["renderer"] != "blender"
            or shard["status"] != "complete"
            or shard["shard"]["start_idx"] != end
        ):
            raise ValueError(
                "environment sealing requires complete contiguous Blender shards from zero"
            )
        version = shard["render_identity"]["config"]["renderer_settings"]["blender_version"]
        match = re.fullmatch(r"Blender (4\.2\.\d+(?: LTS)?)", version)
        if match is None:
            raise ValueError("environment export requires Blender 4.2.x")
        current = source_from_manifest(shard, match.group(1))
        if source is not None and source != current:
            raise ValueError("environment shards have different source identities")
        source = current
        path = bundle_path(dataset / "scenes", shard["scenes"]["file"])
        if sha256_file(path) != shard["scenes"]["sha256"]:
            raise ValueError("environment scene file hash mismatch")
        batch = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        end = shard["shard"]["end_idx"]
        if sorted(item["index"] for item in batch) != list(range(shard["shard"]["start_idx"], end)):
            raise ValueError("environment scene range is incomplete")
        records.extend(batch)
    return seal_bundle(dataset / "environments", records, source=source)


@dataclass
class ReplayInputs:
    """A verified bundle plus the explicit replay options."""

    bundle: BundleReader
    options: dict

    def validate_assets(self, content_sha256: str) -> None:
        """Require the same asset bytes for paired Blender/Isaac realization."""
        if content_sha256 != self.bundle.source["assets_sha256"]:
            raise ValueError("environment asset content differs from Blender source")

    @property
    def identity(self) -> dict[str, Any]:
        """Identity fields recorded in the render identity."""
        return {"bundle_sha256": self.bundle.sha256, **self.options}


def load_replay(
    args: argparse.Namespace, usd_path: str | Path, *, records: Iterable[dict[str, Any]] = ()
) -> ReplayInputs | None:
    """Build verified replay inputs from CLI arguments, or ``None`` when not requested."""
    options = replay_options(
        getattr(args, "environment_root", None),
        getattr(args, "environment_gain", None),
        getattr(args, "environment_adapter", None),
    )
    if options is None:
        return None
    bundle = BundleReader(args.environment_root, usd_sha256=sha256_file(usd_path), seed=args.seed)
    for record in records:
        bundle.verify_record(record)
    return ReplayInputs(bundle, options)


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the replay options to ``parser``."""
    parser.add_argument("--environment-root", help="Sealed native Blender environment bundle")
    parser.add_argument(
        "--environment-gain", type=float, help="Explicit empirical native-light gain"
    )
    parser.add_argument(
        "--environment-adapter",
        choices=[ADAPTER],
        help="Explicit runtime-specific dome adapter",
    )


def include_replay_identity(
    settings: dict[str, Any], replay: ReplayInputs | None
) -> dict[str, Any]:
    """Record the replay identity in ``settings`` after checking the Isaac Sim version."""
    if replay is not None:
        if not re.match(r"^6\.0\.1(?:$|[-+])", settings["runtime"]["isaacsim"]):
            raise ValueError("environment adapter requires Isaac Sim 6.0.1")
        settings["environment"] = replay.identity
    return settings


def validate_light_readbacks(lights: object, *, expected_count: int | None = None) -> None:
    """Validate serialized post-capture evidence without importing USD."""
    if not isinstance(lights, list) or not lights:
        raise ValueError("live-light evidence missing")
    paths = set()
    domes = 0

    def finite(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(cast(float, value))

    for light in lights:
        if not isinstance(light, dict):
            raise ValueError("invalid light evidence")
        path = light.get("path")
        kind = light.get("type")
        matrix = light.get("world_matrix")
        if (
            not isinstance(path, str)
            or not path.startswith("/World/BlenderEnvironment/")
            or path in paths
            or kind
            not in {
                "DomeLight",
                "DistantLight",
                "RectLight",
                "SphereLight",
                "DiskLight",
            }
            or not finite(light.get("intensity"))
            or not finite(light.get("exposure"))
            or not isinstance(matrix, list)
            or len(matrix) != 4
            or any(
                not isinstance(row, list) or len(row) != 4 or not all(finite(v) for v in row)
                for row in matrix
            )
        ):
            raise ValueError("invalid light fields or duplicate path")
        paths.add(path)
        domes += kind == "DomeLight"
    if domes != 1 or (expected_count is not None and len(lights) != expected_count):
        raise ValueError("live-light inventory mismatch")


def validate_applied_readbacks(
    outdir: str | Path,
    records: list[dict[str, Any]],
    cfg: RenderConfig,
    identity: dict[str, Any],
    *,
    bundle: BundleReader | None = None,
) -> None:
    """A resumed/completed native sample requires its post-capture evidence."""
    suffixes = (
        [f"_{cfg.stereo_left_suffix}", f"_{cfg.stereo_right_suffix}"]
        if cfg.stereo_enabled
        else [""]
    )
    for record in records:
        expected_count = bundle.verify_record(record)["light_count"] if bundle is not None else None
        for suffix in suffixes:
            path = Path(outdir) / "environment_applied" / f"{record['index']:06d}{suffix}.json"
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict) or any(
                    payload.get(key) != value for key, value in identity.items()
                ):
                    raise ValueError("environment identity mismatch")
                if (
                    payload.get("scene_sha256") != canonical_hash(record)
                    or not isinstance(payload.get("lights"), list)
                    or not payload["lights"]
                ):
                    raise ValueError("scene or live-light evidence missing")
                validate_light_readbacks(payload["lights"], expected_count=expected_count)
            except (OSError, ValueError) as exc:
                raise ValueError(f"invalid native environment readback {path.name}: {exc}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seal completed native Blender environment exports"
    )
    parser.add_argument("dataset", type=Path)
    seal_dataset(parser.parse_args().dataset)
