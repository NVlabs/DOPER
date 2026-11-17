# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install helpers for the BlenderProc runtime (separate virtual environment, checkout only).

Setup, outside the DOPER environment::

    uv venv .venv-blenderproc && uv pip install --python .venv-blenderproc blenderproc==2.8.0
    python -m rendering.blenderproc.install pin --site .venv-blenderproc/lib/python3.*/site-packages
    python -m rendering.blenderproc.install blender --dest DIR

BlenderProc installs the BOP toolkit into Blender's Python at first use, from the unpinned
``git+https://github.com/thodan/bop_toolkit``. :func:`pin_bop_toolkit` rewrites exactly those two
URLs (``loader/BopLoader.py`` and ``writer/BopWriterUtility.py``) to the pinned revision and
checks both files against their known SHA-256 before and after; rendering code is unchanged.
:func:`fetch_blender` downloads and verifies the pinned Blender 4.2.1 tarball.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from doper.bop.sources import publish_json, sha256_file
from rendering.blenderproc.assets import http_download, load_manifest

BOP_TOOLKIT_REVISION = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
BLENDERPROC_VERSION = "2.8.0"
UNPINNED_URL = '"git+https://github.com/thodan/bop_toolkit"'
PINNED_URL = f'"git+https://github.com/thodan/bop_toolkit@{BOP_TOOLKIT_REVISION}"'
# SHA-256 of each patched file in the BlenderProc 2.8.0 wheel: (as shipped, after the pin).
PIN_FILES = {
    "loader/BopLoader.py": (
        "a9eefd15607fa313239dddad88eb0a52235f0f091b64d8cccd8b8ccd186f95b3",
        "60c0d9e222ed9915bf9597d99cc0da4431719963dbe76d309945cbee259dea98",
    ),
    "writer/BopWriterUtility.py": (
        "cf49dc4e4cbe12e1f7a209b8f7f7a6b3d6308afeab3b1463f18c1f299d551f2f",
        "33a0355ecb235ecf8a94efb6dcde35fba1b572a010d4dde989a78de467d1f6d1",
    ),
}
RECEIPT_NAME = "blenderproc_dependency_patch.json"


def _check_version(site: Path) -> None:
    metadata = site / f"blenderproc-{BLENDERPROC_VERSION}.dist-info" / "METADATA"
    text = metadata.read_text() if metadata.is_file() else ""
    if not re.search(rf"^Version: {re.escape(BLENDERPROC_VERSION)}$", text, re.MULTILINE):
        raise ValueError(f"{site}: requires blenderproc=={BLENDERPROC_VERSION}")


def pin_bop_toolkit(blenderproc_site: Path, receipt: Path | None = None) -> dict[str, Any]:
    """Pin BlenderProc's implicit BOP toolkit install to ``BOP_TOOLKIT_REVISION`` (idempotent).

    Args:
        blenderproc_site: ``site-packages`` directory that holds ``blenderproc/``.
        receipt: Receipt path; defaults to ``RECEIPT_NAME`` inside ``blenderproc_site``.

    Returns:
        The receipt.

    Raises:
        ValueError: The package is not BlenderProc 2.8.0 or a file differs from the known bytes.
    """
    site = Path(blenderproc_site)
    _check_version(site)
    package = site / "blenderproc" / "python"
    for relative, (shipped, pinned) in PIN_FILES.items():
        path = package / relative
        actual = sha256_file(path)
        if actual == pinned:
            continue
        text = path.read_text()
        if actual != shipped or text.count(UNPINNED_URL) != 1:
            raise ValueError(f"{relative}: unexpected BlenderProc file (SHA-256 {actual})")
        path.write_text(text.replace(UNPINNED_URL, PINNED_URL))
        if sha256_file(path) != pinned:
            raise ValueError(f"{relative}: pinned file has an unexpected SHA-256")
    record = {
        "package": f"blenderproc=={BLENDERPROC_VERSION}",
        "bop_revision": BOP_TOOLKIT_REVISION,
        "change": "dependency bootstrap URLs only; rendering algorithms unchanged",
        "files_sha256": {relative: pinned for relative, (_, pinned) in PIN_FILES.items()},
    }
    publish_json(receipt or site / RECEIPT_NAME, record)
    return record


def fetch_blender(dest: Path, *, http_get: Callable[[str, Path], None] = http_download) -> Path:
    """Download, verify and extract the pinned Blender tarball.

    Args:
        dest: Directory that receives the tarball and ``blender-4.2.1-linux-x64/``.
        http_get: HTTP downloader.

    Returns:
        The Blender installation directory (pass it as ``--blender`` to the campaign).
    """
    pin = load_manifest()["blender"]
    dest.mkdir(parents=True, exist_ok=True)
    install_dir = dest / pin["directory"]
    receipt = dest / "READY.json"
    if receipt.is_file() and json.loads(receipt.read_text()).get("sha256") == pin["sha256"]:
        return install_dir
    archive = dest / pin["filename"]
    if not archive.is_file() or sha256_file(archive) != pin["sha256"]:
        http_get(pin["mirrors"][0] + pin["filename"], archive)
    actual = sha256_file(archive)
    if actual != pin["sha256"]:
        raise ValueError(f"{archive.name}: SHA-256 {actual} != pinned {pin['sha256']}")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != pin["directory"]:
                raise ValueError(f"{archive.name}: unsafe member {member.name!r}")
        if install_dir.exists():
            shutil.rmtree(install_dir)
        bundle.extractall(dest, filter="tar")
    publish_json(
        receipt, {"version": pin["version"], "sha256": pin["sha256"], "directory": pin["directory"]}
    )
    return install_dir


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (``pin`` or ``blender``)."""
    parser = argparse.ArgumentParser(description="BlenderProc runtime install helpers.")
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin", help="pin BlenderProc's BOP toolkit install")
    pin.add_argument("--site", type=Path, required=True, help="site-packages holding blenderproc")
    pin.add_argument("--receipt", type=Path, help="receipt path (default: in --site)")
    pin.add_argument("--dry-run", action="store_true")
    blender = commands.add_parser("blender", help="download and verify Blender 4.2.1")
    blender.add_argument("--dest", type=Path, required=True)
    blender.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "pin":
        if args.dry_run:
            _check_version(args.site)
            print(f"would pin {sorted(PIN_FILES)} under {args.site} to {BOP_TOOLKIT_REVISION}")
            return 0
        print(json.dumps(pin_bop_toolkit(args.site, args.receipt), indent=2))
        return 0
    pinned = load_manifest()["blender"]
    if args.dry_run:
        print(f"would fetch {pinned['mirrors'][0]}{pinned['filename']} ({pinned['sha256']})")
        return 0
    print(fetch_blender(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
