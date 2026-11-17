# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N806  (rotation/intrinsics matrices keep their conventional capitalized names)
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from doper.evaluation import bop as bop_eval
from doper.evaluation import bop_metrics as metrics

HANDAL_ROOT = os.environ.get("DOPER_HANDAL_ROOT")
HANDAL_EVAL = Path(HANDAL_ROOT) / "models" / "models_eval" if HANDAL_ROOT else None


def _write_ply(path: Path, pts: np.ndarray, binary: bool) -> None:
    """Write known vertices as a minimal ASCII or binary PLY fixture."""
    header = [
        "ply",
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        if binary:
            f.write(pts.astype("<f4").tobytes())
        else:
            f.write("".join(f"{x} {y} {z}\n" for x, y, z in pts).encode())


def test_read_ply_vertices_ascii_and_binary(tmp_path: Path):
    """Both supported PLY encodings preserve the same model coordinates."""
    pts = np.array([[0, 0, 0], [1.5, -2, 3], [10, 20, 30]], dtype=float)
    _write_ply(tmp_path / "a.ply", pts, binary=False)
    _write_ply(tmp_path / "b.ply", pts, binary=True)
    assert np.allclose(metrics.read_ply_vertices(str(tmp_path / "a.ply")), pts)
    assert np.allclose(metrics.read_ply_vertices(str(tmp_path / "b.ply")), pts)


@pytest.mark.skipif(
    HANDAL_EVAL is None or not (HANDAL_EVAL / "obj_000002.ply").is_file(),
    reason="set DOPER_HANDAL_ROOT to a HANDAL download to run",
)
def test_handal_obj2_eval_model_matches_models_info():
    """Downloaded HANDAL geometry agrees with its published size and symmetry metadata."""
    pts = metrics.read_ply_vertices(str(HANDAL_EVAL / "obj_000002.ply"))
    info = json.loads((HANDAL_EVAL / "models_info.json").read_text())["2"]
    assert pts.shape == (1265, 3)
    size = pts.max(0) - pts.min(0)
    assert np.allclose(size, [info["size_x"], info["size_y"], info["size_z"]], rtol=0.02)
    assert (
        metrics.symmetry_transforms(info) and len(metrics.symmetry_transforms(info)) == 1
    )  # obj2 is not symmetric


def test_mssd_mspd_known_errors():
    """Known translations and symmetric rotations yield analytically expected pose errors."""
    pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=float)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0, 0, 1000.0])
    syms = [np.eye(4)]
    assert (
        metrics.mssd(R, t, R, t, pts, syms) == 0.0 and metrics.mspd(R, t, R, t, K, pts, syms) == 0.0
    )
    # 10 mm along x moves every vertex 10 mm (MSSD) and 500*10/1000 = 5 px (MSPD, ignoring depth)
    assert np.isclose(metrics.mssd(R, t + [10, 0, 0], R, t, pts, syms), 10.0)
    assert np.isclose(metrics.mspd(R, t + [10, 0, 0], R, t, K, pts, syms), 5.0, atol=0.05)
    # a 180 deg flip about z is a large error unless the object is declared symmetric about z
    Rz = np.diag([-1.0, -1.0, 1.0])
    assert metrics.mssd(Rz, t, R, t, pts, syms) == pytest.approx(200.0)
    sym = np.eye(4)
    sym[:3, :3] = Rz
    assert metrics.mssd(Rz, t, R, t, pts, [np.eye(4), sym]) == 0.0
    # continuous symmetry sampling covers a 90 deg rotation about z exactly (36 steps -> 10 deg)
    cont = metrics.symmetry_transforms(
        {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]}
    )
    R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert len(cont) == 36 and metrics.mssd(R90, t, R, t, pts, cont) < 1e-9


def test_average_recall_and_summary():
    """Missed detections remain in the denominator across all recall thresholds."""
    errors = [0.0, 10.0, 60.0, None]  # None = missed detection
    thresholds = [5.0, 20.0, 50.0]
    # <5: 1/4, <20: 2/4, <50: 2/4 -> mean 0.4167
    assert metrics.average_recall(errors, thresholds) == pytest.approx((0.25 + 0.5 + 0.5) / 3)
    frames = [{"mssd_mm": e, "mspd_px640": e} for e in errors]
    s = metrics.summarize(
        frames, diameter_mm=100.0
    )  # MSSD thresholds 5..50 mm == MSPD thresholds 5..50 px
    assert s["frames"] == 4 and s["detections"] == 3 and s["det_rate"] == 0.75
    assert s["ar_mssd"] == pytest.approx(s["ar_mspd"]) and s["ar_mean"] == pytest.approx(
        s["ar_mssd"]
    )
    assert s["mssd_median_mm"] == 10.0
    assert metrics.average_recall([], thresholds) != metrics.average_recall(
        [], thresholds
    )  # nan for no frames


def test_val_frames_picks_frames_with_the_object(tmp_path: Path):
    """Select the requested object and preserve numeric scene-frame ordering."""
    scene = tmp_path / "000003"
    scene.mkdir()
    (scene / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {"obj_id": 1, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0, 0, 1]}
                ],
                "1": [
                    {"obj_id": 2, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0, 0, 2]},
                    {"obj_id": 1, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0, 0, 3]},
                ],
                "10": [
                    {"obj_id": 2, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0, 0, 4]}
                ],
            }
        )
    )
    (scene / "scene_camera.json").write_text(
        json.dumps({k: {"cam_K": [1] * 9} for k in ("0", "1", "10")})
    )
    frames = bop_eval.val_frames(tmp_path, 2)
    assert [(s.name, f, gt["cam_t_m2c"][2]) for s, f, gt, _ in frames] == [
        ("000003", 1, 2),
        ("000003", 10, 4),
    ]


def _eval_frame(tmp_path: Path, frame: int = 0):
    """Create a decodable frame with known object pose and camera intrinsics."""
    import cv2

    scene = tmp_path / "000002"
    rgb = scene / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    path = rgb / f"{frame:06d}.png"
    assert cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))
    gt = {"obj_id": 2, "cam_R_m2c": np.eye(3).ravel().tolist(), "cam_t_m2c": [0, 0, 1000]}
    cam = {"cam_K": [100, 0, 16, 0, 100, 16, 0, 0, 1]}
    return (scene, frame, gt, cam), path


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_prepare_frames_rejects_unreadable_rgb(tmp_path, broken):
    """Missing and corrupt RGB inputs fail before a scoring population is accepted."""
    frame, path = _eval_frame(tmp_path)
    if broken == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="000002.*000000"):
        bop_eval.prepare_frames([frame])


def test_prepare_frames_rejects_empty_population():
    """An empty selection cannot produce a benchmark result."""
    with pytest.raises(ValueError, match=r"no.*frames"):
        bop_eval.prepare_frames([])


def _score_frames(frames, monkeypatch, tmp_path):
    # Replace only model loading/inference: exercise real decoding and scoring.
    """Exercise decoding and metric aggregation with model execution replaced."""

    monkeypatch.setattr(bop_eval, "load_model", lambda *args: object())
    monkeypatch.setattr(bop_eval, "infer_image", lambda *args, **kwargs: [])
    return bop_eval.evaluate_checkpoint(
        "fixture.pth",
        frames,
        obj_id=2,
        kp_mm=np.zeros((4, 3)),
        pts_mm=np.zeros((4, 3)),
        syms=[np.eye(4)],
        diameter_mm=100,
        score_thr=0.15,
        device="cpu",
        csv_path=tmp_path / "result.csv",
        log=lambda _: None,
    )


def test_evaluation_non_detection_counts_as_failure(tmp_path, monkeypatch):
    """A valid image with no prediction contributes zero recall, not a skipped frame."""
    frame, _ = _eval_frame(tmp_path)
    frames = bop_eval.prepare_frames([frame])
    result = _score_frames(frames, monkeypatch, tmp_path)
    assert result["summary"]["frames"] == 1
    assert result["summary"]["detections"] == 0
    assert result["summary"]["ar_mean"] == 0
    assert result["frames"][0]["frame"] == 0


@pytest.mark.parametrize("changed", ["missing", "corrupt", "different_valid_image"])
def test_evaluation_rejects_changed_input_after_preflight(tmp_path, monkeypatch, changed):
    """Scoring rejects changed image content and does not publish a checkpoint CSV."""
    import cv2

    frame, path = _eval_frame(tmp_path)
    frames = bop_eval.prepare_frames([frame])
    if changed == "missing":
        path.unlink()
    elif changed == "corrupt":
        path.write_bytes(b"broken")
    else:
        assert cv2.imwrite(str(path), np.full((32, 32, 3), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="000002.*000000"):
        _score_frames(frames, monkeypatch, tmp_path)
    assert not (tmp_path / "result.csv").exists()


def test_val_frames_rejects_multiple_target_instances(tmp_path):
    """The single-instance evaluator refuses ambiguous target matching."""
    frame, _ = _eval_frame(tmp_path)
    scene, _, gt, cam = frame
    (scene / "scene_gt.json").write_text(json.dumps({"0": [gt, gt]}))
    (scene / "scene_camera.json").write_text(json.dumps({"0": cam}))
    with pytest.raises(ValueError, match="single.instance"):
        bop_eval.val_frames(tmp_path, 2)


@pytest.mark.parametrize("values", [["same=a", "same=b"], ["=file.pth"], ["name="]])
def test_checkpoint_names_reject_ambiguous_inputs(values):
    """Empty or duplicate checkpoint labels cannot silently overwrite results."""
    argv = ["--obj-id", "2", "--val-dir", "v", "--models-dir", "m", "--keypoints", "k"]
    for value in values:
        argv += ["--checkpoint", value]
    with pytest.raises(SystemExit):
        bop_eval.main(argv)


@pytest.mark.parametrize("posed", [True, False])
@pytest.mark.parametrize("include_local_paths", [False, True])
def test_main_scores_fixed_population_for_multiple_checkpoints(
    tmp_path, monkeypatch, posed, include_local_paths
):
    """CLI output preserves scoring inputs and valid JSON with or without detections."""
    import hashlib
    from types import SimpleNamespace

    frame, image_path = _eval_frame(tmp_path / "val")
    scene, _, gt, cam = frame
    (scene / "scene_gt.json").write_text(json.dumps({"0": [gt]}))
    (scene / "scene_camera.json").write_text(json.dumps({"0": cam}))
    models = tmp_path / "models"
    models.mkdir()
    pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=float)
    _write_ply(models / "obj_000002.ply", pts, binary=False)
    (models / "models_info.json").write_text(json.dumps({"2": {"diameter": 150}}))
    keypoints = tmp_path / "keypoints.json"
    keypoints.write_text(json.dumps({"keypoints_3d": (pts / 1000).tolist()}))
    # Only inference is replaced. Exercise CLI parsing, actual image decode,
    # metric computation, per-checkpoint CSVs and the output frame manifest.
    monkeypatch.setattr(bop_eval, "load_model", lambda *args: object())
    detection = SimpleNamespace(
        rvec=np.zeros(3), tvec=np.array([0, 0, 1000]), score=0.9, reproj_error_px=0.0
    )
    keypoints_digest = hashlib.sha256(keypoints.read_bytes()).hexdigest()

    def predict(*args, **kwargs):
        # The manifest must describe the keypoints consumed before inference,
        # even if the source file is replaced while checkpoints are scored.
        """Replace the keypoint source after ingestion to test snapshot provenance."""
        keypoints.write_text('{"keypoints_3d": []}')
        return [detection] if posed else []

    monkeypatch.setattr(bop_eval, "infer_image", predict)
    output = tmp_path / "result.json"
    assert (
        bop_eval.main(
            [
                "--obj-id",
                "2",
                "--val-dir",
                str(tmp_path / "val"),
                "--models-dir",
                str(models),
                "--keypoints",
                str(keypoints),
                "--checkpoint",
                f"first={tmp_path}/a.pth",
                "--checkpoint",
                f"second={tmp_path}/b.pth",
                "--device",
                "cpu",
                "--out-json",
                str(output),
                "--csv-dir",
                str(tmp_path / "csv"),
            ]
            + (["--include-local-paths"] if include_local_paths else [])
        )
        == 0
    )

    def reject_constant(value):
        """Reject non-standard JSON constants that permissive decoders otherwise accept."""
        raise ValueError(f"non-standard JSON constant: {value}")

    result = json.loads(output.read_text(), parse_constant=reject_constant)
    assert result["checkpoints"] == {
        name: str(tmp_path / filename) if include_local_paths else filename
        for name, filename in [("first", "a.pth"), ("second", "b.pth")]
    }
    assert result["val_dir"] == (str(tmp_path / "val") if include_local_paths else "val")
    assert result["evaluation"]["models_dir"] == (str(models) if include_local_paths else "models")
    assert result["evaluation"]["keypoints"] == (
        str(keypoints) if include_local_paths else "keypoints.json"
    )
    if not include_local_paths:
        assert str(tmp_path) not in output.read_text()
    assert result["evaluation"]["keypoints_sha256"] == keypoints_digest
    for name in ("first", "second"):
        summary = result["results"][name]["summary"]
        assert summary["frames"] == 1
        assert summary["detections"] == int(posed)
        assert summary["ar_mean"] == int(posed)
        if not posed:
            assert summary["mssd_median_mm"] is None
            assert summary["mspd_median_px640"] is None
        assert (tmp_path / "csv" / f"{name}_obj000002_val.csv").is_file()
    assert result["evaluation"]["frames"] == [
        {
            "scene": "000002",
            "frame": 0,
            "image_path": str(image_path.resolve()) if include_local_paths else "rgb/000000.png",
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "gt": gt,
            "camera": cam,
        }
    ]


@pytest.mark.parametrize(
    "info",
    [
        {
            "symmetries_discrete": [np.diag([1, -1, -1, 1]).ravel().tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}],
        },
        {"symmetries_continuous": [{"axis": [0, 0, 1]}, {"axis": [1, 0, 0]}]},
    ],
)
def test_symmetry_transforms_reject_unsupported_compositions(info):
    # The extracted evaluator only enumerates transforms; it cannot score
    # groups requiring compositions of continuous and other symmetries.
    """Reject symmetry groups whose composed transforms are not enumerated."""
    with pytest.raises(ValueError, match="unsupported.*symmetr"):
        metrics.symmetry_transforms(info)
