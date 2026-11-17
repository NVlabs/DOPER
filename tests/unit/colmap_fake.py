# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake ``colmap`` binary for the unit tests (COLMAP is not installed in CI).

The fake records each call's argv in ``calls.jsonl`` next to it and writes small outputs:
``feature_extractor`` a database whose image ids run opposite to the file order (so binding
must go by name), ``mapper`` two models (``0`` with one image, ``1`` with every image but the
last; binary ``images.bin`` header plus the TXT files), ``point_triangulator`` its input
model with one 3D point seen by every image (with ``scene.json``, the images it names see the
scene points instead, at their projection through each image's true camera, whatever pose the
model holds), ``image_registrator`` its input model plus every database image it lacks
(identity rotation, center on the z axis), ``model_converter`` and ``bundle_adjuster`` a copy
of the TXT files. The first ``bundle_adjuster`` after
``scale.txt`` is written scales the model's world by its value (and removes the file), the
gauge drift a free-pose bundle adjustment can introduce. ``help`` prints the version in
``version.txt``; a command named in ``fail.txt`` prints a log and exits 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT = r"""
import json, shutil, sqlite3, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
argv = sys.argv[1:]
with (HERE / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(argv) + "\n")
command = argv[0] if argv else ""
options = dict(zip(argv[1::2], argv[2::2]))
fail = HERE / "fail.txt"
if fail.exists() and fail.read_text().strip() == command:
    for i in range(30):
        print(f"fake log line {i}")
    print(f"E fake failure of {command}")
    sys.exit(3)
if command == "help":
    print(f"COLMAP {(HERE / 'version.txt').read_text().strip()} -- Structure-from-Motion "
          "(Commit fake)")
    print("Usage: colmap [command] [options]")
elif command == "feature_extractor":
    names = sorted(p.name for p in Path(options["--image_path"]).iterdir())
    params = options.get("--ImageReader.camera_params", "80,80,47.5,26.5,0,0,0,0")
    db = sqlite3.connect(options["--database_path"])
    db.execute("CREATE TABLE cameras (camera_id INTEGER, params TEXT)")
    db.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
    db.execute("INSERT INTO cameras VALUES (1, ?)", (params,))
    for i, name in enumerate(names):
        db.execute("INSERT INTO images VALUES (?, ?, 1)", (len(names) - i, name))
    db.commit()
elif command == "mapper":
    from doper.reconstruction import colmap_io as io

    db = sqlite3.connect(options["--database_path"])
    params = tuple(float(v) for v in db.execute("SELECT params FROM cameras").fetchone()[0]
                   .split(","))
    rows = db.execute("SELECT image_id, name FROM images ORDER BY name").fetchall()
    camera = io.ColmapCamera(1, "OPENCV", 96, 54, params)
    for model_id, kept in ((0, rows[:1]), (1, rows[:-1])):
        images = {i: io.ColmapImage(i, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, float(i)), 1, name,
                                    ((1.0, 2.0, 1), (3.0, 4.0, -1)))
                  for i, name in kept}
        track = tuple((i, 0) for i, _ in kept)
        points = {1: io.ColmapPoint3D(1, (0.0, 0.0, 1.0), (1, 2, 3), 0.5, track)}
        out = Path(options["--output_path"]) / str(model_id)
        out.mkdir(parents=True, exist_ok=True)
        io.write_text_model(io.ColmapModel({1: camera}, images, points), out)
        (out / "images.bin").write_bytes(struct.pack("<Q", len(images)))
elif command == "point_triangulator" and (Path(options["--input_path"]) / "cameras.txt").exists():
    from dataclasses import replace
    from doper.reconstruction import colmap_io as io

    model = io.read_text_model(options["--input_path"])
    images = {i: replace(image, points2d=((1.0, 2.0, 1),)) for i, image in model.images.items()}
    track = tuple((i, 0) for i in images)
    points = {1: io.ColmapPoint3D(1, (0.0, 0.0, 1.0), (1, 2, 3), 0.5, track)}
    scene = HERE / "scene.json"
    if scene.exists():
        # Features of a static scene: each image observes the scene points at their projection
        # through its TRUE camera, whatever pose the input model holds.
        import numpy as np

        truth = json.loads(scene.read_text())
        xyz = np.array(truth["points"], dtype=float)
        fx, fy, cx, cy = next(iter(model.cameras.values())).params[:4]
        tracks = {j: [] for j in range(len(xyz))}
        for i, image in model.images.items():
            pose = truth["cam_from_world"].get(image.name)
            if pose is None:
                continue
            pose = np.array(pose)
            cam = xyz @ pose[:3, :3].T + pose[:3, 3]
            uv = np.stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy], 1)
            images[i] = replace(image, points2d=tuple((float(u), float(v), j + 2)
                                                      for j, (u, v) in enumerate(uv)))
            for j in range(len(xyz)):
                tracks[j].append((i, j))
        others = tuple((i, 0) for i, image in model.images.items()
                       if image.name not in truth["cam_from_world"])
        points = {1: replace(points[1], track=others)} if others else {}
        for j, point in enumerate(xyz):
            if tracks[j]:
                points[j + 2] = io.ColmapPoint3D(j + 2, tuple(float(v) for v in point),
                                                 (1, 2, 3), 0.5, tuple(tracks[j]))
    out = Path(options["--output_path"])
    out.mkdir(parents=True, exist_ok=True)
    io.write_text_model(io.ColmapModel(model.cameras, images, points), out)
elif command == "bundle_adjuster" and (HERE / "scale.txt").exists():
    from dataclasses import replace
    from doper.reconstruction import colmap_io as io

    scale = float((HERE / "scale.txt").read_text())
    (HERE / "scale.txt").unlink()
    model = io.read_text_model(options["--input_path"])
    images = {i: replace(image, tvec=tuple(scale * v for v in image.tvec))
              for i, image in model.images.items()}
    points = {i: replace(point, xyz=tuple(scale * v for v in point.xyz))
              for i, point in model.points3d.items()}
    out = Path(options["--output_path"])
    out.mkdir(parents=True, exist_ok=True)
    io.write_text_model(io.ColmapModel(model.cameras, images, points), out)
elif command == "image_registrator":
    from doper.reconstruction import colmap_io as io

    model = io.read_text_model(options["--input_path"])
    known = {image.name for image in model.images.values()}
    db = sqlite3.connect(options["--database_path"])
    images = dict(model.images)
    for image_id, name in db.execute("SELECT image_id, name FROM images").fetchall():
        if name not in known:
            images[image_id] = io.ColmapImage(image_id, (1.0, 0.0, 0.0, 0.0),
                                              (0.0, 0.0, float(image_id)), 1, name, ())
    out = Path(options["--output_path"])
    out.mkdir(parents=True, exist_ok=True)
    io.write_text_model(io.ColmapModel(model.cameras, images, model.points3d), out)
elif command in ("model_converter", "point_triangulator", "bundle_adjuster"):
    out = Path(options["--output_path"])
    out.mkdir(parents=True, exist_ok=True)
    for path in Path(options["--input_path"]).glob("*.txt"):  # nothing for a missing input
        shutil.copy(path, out / path.name)
print(f"fake {command} done")
"""


def install_fake_colmap(directory: Path, version: str = "3.13.0") -> Path:
    """Write the fake binary into ``directory``; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "colmap"
    binary.write_text(f"#!{sys.executable}\n{SCRIPT}")
    binary.chmod(0o755)
    (directory / "version.txt").write_text(version)
    return binary


def recorded_calls(binary: Path) -> list[list[str]]:
    """The argv (without the binary) of every call of the fake, in order."""
    path = binary.parent / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]
