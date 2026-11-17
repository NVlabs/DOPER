# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-gate monocular pseudo-label pools: export, audit and static+dynamic merge."""

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper.bop import sources
from doper.infer import Detection
from doper.pseudo_labels import monocular
from doper.pseudo_labels.monocular import GateSpec

WIDTH, HEIGHT = 100, 50
KEYPOINT_COUNT = 60
FOCAL = 200.0
DEPTH_MM = 500.0
BOX = (30.0, 12.0, 70.0, 38.0)
TEACHER = "ab" * 32
INTRINSICS = [FOCAL, 0.0, WIDTH / 2, 0.0, FOCAL, HEIGHT / 2, 0.0, 0.0, 1.0]
CAMERA_MATRIX = np.array(INTRINSICS).reshape(3, 3)
SEQUENCE = "obj_000001_up"
UNRELATED = frozenset({"0" * 64})


@dataclass(frozen=True)
class Scenario:
    """What the scripted predictor reports for one image (keyed by its flat gray level)."""

    score: float = 0.9
    reproj: float = 1.0
    confident: int = KEYPOINT_COUNT
    mask_rect: tuple[int, int, int, int] = (40, 20, 5, 2)  # x, y, width, height: 10 pixels
    detect: bool = True


class ScriptedPredictor:
    """Deterministic ``Predictor``: the image's gray level selects a ``Scenario``."""

    def __init__(
        self,
        scenarios: dict[int, Scenario],
        keypoints_mm: np.ndarray,
        default: Scenario | None = None,
    ) -> None:
        self.scenarios = scenarios
        self.default = default
        self.keypoints_mm = keypoints_mm
        self.thresholds: list[float] = []

    def __call__(self, image_bgr, camera_matrix, keypoints_mm, score_threshold):
        self.thresholds.append(score_threshold)
        code = int(image_bgr[0, 0, 0])
        scenario = self.scenarios.get(code, self.default) if self.default else self.scenarios[code]
        if not scenario.detect:
            return [], None
        projected = cv2.projectPoints(
            keypoints_mm, np.zeros(3), np.array([0.0, 0.0, DEPTH_MM]), camera_matrix, np.zeros(5)
        )[0].reshape(-1, 2)
        scores = np.where(np.arange(KEYPOINT_COUNT) < scenario.confident, 0.9, 0.1)
        detection = Detection(
            bbox_xyxy=np.array(BOX),
            score=scenario.score,
            keypoints=projected,
            keypoint_scores=scores,
            rvec=np.zeros(3),
            tvec=np.array([0.0, 0.0, DEPTH_MM]),
            reproj_error_px=scenario.reproj,
        )
        x, y, w, h = scenario.mask_rect
        prob = np.zeros((HEIGHT, WIDTH), np.float32)
        prob[y : y + h, x : x + w] = 0.9
        prob[0:4, 0:4] = 0.9  # a similar object far outside the box
        return [detection], prob


def keypoints_mm() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(-30.0, 30.0, (KEYPOINT_COUNT, 3))


def write_keypoints(path: Path) -> Path:
    path.write_text(json.dumps({"keypoints_3d": (keypoints_mm() / 1000.0).tolist()}))
    return path


def write_view(root: Path, kind: str, codes: list[int], sequence: str = SEQUENCE) -> Path:
    """``<root>/<kind>`` with one flat PNG per gray level (so one scripted scenario each)."""
    view = root / kind
    (view / sequence / "rgb").mkdir(parents=True)
    cameras = {}
    for frame, code in enumerate(codes):
        ok, data = cv2.imencode(".png", np.full((HEIGHT, WIDTH, 3), code, np.uint8))
        assert ok
        (view / sequence / "rgb" / f"{frame:06d}.png").write_bytes(data.tobytes())
        cameras[str(frame)] = {"cam_K": INTRINSICS, "width": WIDTH, "height": HEIGHT}
    (view / sequence / "scene_camera.json").write_text(json.dumps(cameras))
    return view


def export(
    tmp_path: Path,
    scenarios: dict[int, Scenario],
    *,
    kind: str = "static",
    codes: list[int] | None = None,
    exclusions: frozenset[str] = UNRELATED,
    gate: GateSpec | None = None,
    name: str = "pool",
    sequence: str = SEQUENCE,
):
    codes = codes if codes is not None else sorted(scenarios)
    view = write_view(tmp_path / f"views_{name}", kind, codes, sequence)
    keypoints = write_keypoints(tmp_path / "keypoints_3d.json")
    summary = monocular.export_pool(
        ScriptedPredictor(scenarios, keypoints_mm()),
        keypoints,
        view,
        kind,
        exclusions,
        gate or GateSpec(),
        tmp_path / name,
        teacher_sha256=TEACHER,
        obj_id=1,
    )
    return summary, view


def read_manifest(pool: Path) -> dict:
    return json.loads((pool / "manifest.json").read_text())


def read_coco(pool: Path) -> dict:
    return json.loads((pool / "train" / "annotations.json").read_text())


def test_gate_defaults_and_manifest_keys() -> None:
    gate = GateSpec()
    assert (gate.detector_threshold, gate.score, gate.keypoint_score) == (0.1, 0.5, 0.3)
    assert (gate.max_reprojection_px, gate.min_inliers, gate.ransac_px) == (3.0, 50, 8.0)
    assert (gate.mask_probability, gate.min_area_fraction, gate.box_pad) == (0.5, 0.002, 0.05)
    assert gate.as_manifest() == {
        "score_gt": 0.5,
        "reprojection_lt_px": 3.0,
        "ransac_inliers_gt": 50,
        "ransac_px": 8.0,
        "kp_score_thr": 0.3,
        "pnp_method": "usac",
    }
    assert GateSpec(pnp_method="epnp").as_manifest() == {
        "score_gt": 0.5,
        "reprojection_lt_px": 3.0,
        "ransac_inliers_gt": 50,
        "ransac_px": 8.0,
        "kp_score_thr": 0.3,
    }
    assert monocular.SOURCE_KINDS == ("static", "dynamic", "test")
    assert monocular.TEST_NOTICE == sources.TEST_NOTICE


GATE_CASES = [
    pytest.param(Scenario(score=0.5), "score", id="score_exactly_threshold_rejected"),
    pytest.param(Scenario(score=0.5001), None, id="score_above_threshold_accepted"),
    pytest.param(Scenario(reproj=3.0), "reproj", id="reproj_exactly_threshold_rejected"),
    pytest.param(Scenario(reproj=2.999), None, id="reproj_below_threshold_accepted"),
    pytest.param(Scenario(confident=50), "inliers", id="inliers_exactly_threshold_rejected"),
    pytest.param(Scenario(confident=51), None, id="inliers_above_threshold_accepted"),
    pytest.param(Scenario(mask_rect=(40, 20, 3, 3)), "mask", id="area_below_threshold_rejected"),
    pytest.param(Scenario(mask_rect=(40, 20, 5, 2)), None, id="area_at_threshold_accepted"),
    pytest.param(Scenario(detect=False), "no_det", id="no_detection"),
    pytest.param(Scenario(score=float("nan")), "score", id="nan_score_rejected"),
]


@pytest.mark.parametrize(("scenario", "reason"), GATE_CASES)
def test_gate_boundaries(tmp_path: Path, scenario: Scenario, reason: str | None) -> None:
    summary, _ = export(tmp_path, {7: scenario})
    reasons = read_manifest(tmp_path / "pool")["reject_reasons"]
    expected = dict.fromkeys(reasons, 0)
    if reason is not None:
        expected[reason] = 1
    assert reasons == expected
    assert (summary.accepted, summary.rejected) == ((0, 1) if reason else (1, 0))
    assert summary.frames_seen == 1


def test_area_threshold_is_exact_for_the_fixture() -> None:
    assert GateSpec().min_area_fraction * HEIGHT * WIDTH == 10.0


def test_mask_is_restricted_to_the_padded_box(tmp_path: Path) -> None:
    export(tmp_path, {7: Scenario()})
    mask = cv2.imread(str(next((tmp_path / "pool/train/masks").glob("*.png"))), 0)
    assert int((mask > 0).sum()) == 10
    assert not mask[0:4, 0:4].any()
    assert read_coco(tmp_path / "pool")["annotations"][0]["area"] == 10


def test_pool_has_no_keypoint_fields(tmp_path: Path) -> None:
    export(tmp_path, {1: Scenario(), 2: Scenario()})
    coco = read_coco(tmp_path / "pool")
    assert len(coco["annotations"]) == 2
    for annotation in coco["annotations"]:
        assert not set(monocular.BANNED_KEYS) & set(annotation)
        assert set(annotation) == {
            "id",
            "image_id",
            "category_id",
            "bbox",
            "area",
            "iscrowd",
            "segmentation",
            "mask_file",
        }
    assert monocular.BANNED_KEYS == ("keypoints", "num_keypoints", "rvec", "tvec", "object_pose")
    record = read_manifest(tmp_path / "pool")["records"][0]
    assert "rvec" in record and "tvec" in record  # the audit trail keeps the teacher pose


def test_export_excludes_validation_hashes(tmp_path: Path) -> None:
    view = write_view(tmp_path / "probe", "static", [1, 2, 3])
    held_out = sources.sha256_file(view / SEQUENCE / "rgb" / "000001.png")
    summary, _ = export(
        tmp_path, dict.fromkeys((1, 2, 3), Scenario()), exclusions=frozenset({held_out})
    )
    manifest = read_manifest(tmp_path / "pool")
    assert manifest["reject_reasons"]["reserved_eval"] == 1
    assert summary.accepted == 2 and summary.rejected == 1
    assert held_out not in {r["source_image_sha256"] for r in manifest["records"]}
    assert not any(
        sources.sha256_file(p) == held_out for p in (tmp_path / "pool/train/images").iterdir()
    )


def test_export_orders_jpg_before_png_and_names_frames(tmp_path: Path) -> None:
    view = write_view(tmp_path / "views_pool", "static", [1, 2])
    ok, data = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), 200, np.uint8))
    assert ok
    (view / SEQUENCE / "rgb" / "000009.jpg").write_bytes(data.tobytes())
    cameras = json.loads((view / SEQUENCE / "scene_camera.json").read_text())
    cameras["9"] = cameras["0"]
    (view / SEQUENCE / "scene_camera.json").write_text(json.dumps(cameras))
    monocular.export_pool(
        ScriptedPredictor({1: Scenario(), 2: Scenario()}, keypoints_mm(), default=Scenario()),
        write_keypoints(tmp_path / "kp.json"),
        view,
        "static",
        UNRELATED,
        GateSpec(),
        tmp_path / "pool",
        teacher_sha256=TEACHER,
        obj_id=1,
    )
    images = [i["file_name"] for i in read_coco(tmp_path / "pool")["images"]]
    assert images == [
        f"scene{SEQUENCE}_frame000009.jpg",
        f"scene{SEQUENCE}_frame000000.png",
        f"scene{SEQUENCE}_frame000001.png",
    ]


def test_detector_threshold_is_passed_to_the_predictor(tmp_path: Path) -> None:
    view = write_view(tmp_path / "views_pool", "static", [1])
    predictor = ScriptedPredictor({1: Scenario()}, keypoints_mm())
    monocular.export_pool(
        predictor,
        write_keypoints(tmp_path / "kp.json"),
        view,
        "static",
        UNRELATED,
        GateSpec(),
        tmp_path / "pool",
        teacher_sha256=TEACHER,
        obj_id=1,
    )
    assert predictor.thresholds == [0.1]


def test_test_kind_records_transductive_flags(tmp_path: Path) -> None:
    export(tmp_path, {1: Scenario()}, kind="test", sequence="000001")
    manifest = read_manifest(tmp_path / "pool")
    assert manifest["source_kind"] == sources.SOURCE_KINDS["test"]
    assert manifest["transductive_test_adaptation"] is True
    assert manifest["test_gt_used"] is False
    assert manifest["test_targets_used"] is False
    assert manifest["transduction"] == sources.TEST_NOTICE
    assert manifest["gt_pose_or_mask_used"] is False
    static = export(tmp_path, {1: Scenario()}, name="pool_static")
    assert static[0].accepted == 1
    flags = read_manifest(tmp_path / "pool_static")
    assert flags["transductive_test_adaptation"] is False
    assert flags["checkpoint_sha256"] == TEACHER
    assert flags["gate"] == GateSpec().as_manifest()
    assert flags["obj_id"] == 1
    assert flags["source_kind"] == "onboarding_static"


def test_export_refuses_existing_output_and_bad_inputs(tmp_path: Path) -> None:
    export(tmp_path, {1: Scenario()})
    with pytest.raises(FileExistsError):
        export(tmp_path, {1: Scenario()}, name="pool")
    with pytest.raises(ValueError, match="kind"):
        export(tmp_path, {1: Scenario()}, kind="val", name="other")


def test_verify_pool_accepts_a_fresh_export(tmp_path: Path) -> None:
    _, view = export(tmp_path, {1: Scenario(), 2: Scenario()})
    assert monocular.verify_pool(tmp_path / "pool", view, UNRELATED) == []


def test_verify_pool_reports_tampered_mask(tmp_path: Path) -> None:
    _, view = export(tmp_path, {1: Scenario()})
    path = next((tmp_path / "pool/train/masks").glob("*.png"))
    mask = cv2.imread(str(path), 0)
    mask[0:4, 0:4] = 255
    assert cv2.imwrite(str(path), mask)
    faults = monocular.verify_pool(tmp_path / "pool", view, UNRELATED)
    assert faults == ["1 masks are empty or fall outside their box"]


def test_verify_pool_reports_contract_violations(tmp_path: Path) -> None:
    _, view = export(tmp_path, {1: Scenario(), 2: Scenario()})
    held_out = sources.sha256_file(view / SEQUENCE / "rgb" / "000000.png")
    pool = tmp_path / "pool"
    coco = read_coco(pool)
    coco["annotations"][0]["keypoints"] = [0, 0, 1]
    (pool / "train/annotations.json").write_text(json.dumps(coco))
    (view / SEQUENCE / "scene_camera.json").write_text("{}")
    faults = monocular.verify_pool(pool, view, frozenset({held_out}))
    assert any("pose or keypoint fields" in f for f in faults)
    assert any("validation image present" in f for f in faults)
    assert "static calibration changed" in faults


def test_export_needs_a_validation_population_and_an_object_for_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exclusion population"):
        export(tmp_path, {1: Scenario()}, exclusions=frozenset())
    view = write_view(tmp_path / "t", "test", [1], "000001")
    with pytest.raises(ValueError, match="obj_id"):
        monocular.export_pool(
            ScriptedPredictor({1: Scenario()}, keypoints_mm()),
            write_keypoints(tmp_path / "kp.json"),
            view,
            "test",
            UNRELATED,
            GateSpec(),
            tmp_path / "pool_t",
            teacher_sha256=TEACHER,
        )


def test_view_with_several_objects_needs_obj_id(tmp_path: Path) -> None:
    view = write_view(tmp_path / "v", "static", [1], "obj_000001_up")
    write_view(tmp_path / "w", "static", [1], "obj_000002_up")
    shutil.copytree(tmp_path / "w/static/obj_000002_up", view / "obj_000002_up")
    args = (
        ScriptedPredictor({1: Scenario()}, keypoints_mm()),
        write_keypoints(tmp_path / "kp.json"),
        view,
        "static",
        UNRELATED,
        GateSpec(),
    )
    with pytest.raises(ValueError, match="obj_id"):
        monocular.export_pool(*args, tmp_path / "p1", teacher_sha256=TEACHER)
    summary = monocular.export_pool(*args, tmp_path / "p2", teacher_sha256=TEACHER, obj_id=2)
    assert summary.frames_seen == 1


def test_verify_pool_requires_manifest(tmp_path: Path) -> None:
    _, view = export(tmp_path, {1: Scenario()})
    (tmp_path / "pool/manifest.json").unlink()
    assert any("manifest" in f for f in monocular.verify_pool(tmp_path / "pool", view, UNRELATED))


def export_pair(tmp_path: Path, **kwargs):
    """Static pool with gray levels 1 and 2, dynamic pool with 2 (same bytes) and 3."""
    both = {1: Scenario(), 2: Scenario(), 3: Scenario()}
    export(tmp_path, both, kind="static", codes=[1, 2], name="static", **kwargs)
    export(tmp_path, both, kind="dynamic", codes=[2, 3], name="dynamic", sequence="obj_000001")
    return {"static": tmp_path / "static", "dynamic": tmp_path / "dynamic"}


def test_merge_dedups_by_image_hash_first_wins(tmp_path: Path) -> None:
    components = export_pair(tmp_path)
    summary = monocular.merge_pools(components, tmp_path / "merged")
    assert (summary.accepted, summary.frames_seen) == (3, 4)
    coco = read_coco(tmp_path / "merged")
    names = [i["file_name"] for i in coco["images"]]
    assert names == [
        f"static__scene{SEQUENCE}_frame000000.png",
        f"static__scene{SEQUENCE}_frame000001.png",
        "dynamic__sceneobj_000001_frame000001.png",
    ]
    assert [i["id"] for i in coco["images"]] == [0, 1, 2]
    assert [a["id"] for a in coco["annotations"]] == [0, 1, 2]
    assert [a["image_id"] for a in coco["annotations"]] == [0, 1, 2]
    assert coco["annotations"][2]["mask_file"] == "masks/dynamic__sceneobj_000001_frame000001.png"
    manifest = read_manifest(tmp_path / "merged")
    assert manifest["source_kind"] == "onboarding_static_dynamic"
    assert manifest["reject_reasons"]["duplicate_image"] == 1
    assert [d["source"] for d in manifest["deduplicated"]] == ["dynamic"]
    assert manifest["source_counts"]["static"]["accepted_for_training"] == 2
    assert manifest["source_counts"]["dynamic"]["accepted_for_training"] == 1
    assert manifest["accepted"] == 3
    assert (tmp_path / "merged/train/images" / names[2]).is_file()
    assert monocular.verify_merged(tmp_path / "merged") == []


def test_merge_requires_both_kinds_and_a_fresh_output(tmp_path: Path) -> None:
    components = export_pair(tmp_path)
    with pytest.raises(ValueError, match="static and dynamic"):
        monocular.merge_pools({"static": components["static"]}, tmp_path / "merged")
    monocular.merge_pools(components, tmp_path / "merged")
    with pytest.raises(FileExistsError):
        monocular.merge_pools(components, tmp_path / "merged")


def test_merge_rejects_a_relaxed_gate(tmp_path: Path) -> None:
    relaxed = replace(GateSpec(), score=0.4)
    export(tmp_path, {1: Scenario()}, kind="static", name="static", gate=relaxed)
    export(tmp_path, {2: Scenario()}, kind="dynamic", name="dynamic", sequence="obj_000001")
    with pytest.raises(ValueError, match="gate"):
        monocular.merge_pools(
            {"static": tmp_path / "static", "dynamic": tmp_path / "dynamic"}, tmp_path / "merged"
        )


def test_verify_merged_rejects_mismatched_gate(tmp_path: Path) -> None:
    components = export_pair(tmp_path)
    monocular.merge_pools(components, tmp_path / "merged")
    manifest_path = components["dynamic"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gate"]["score_gt"] = 0.6
    manifest_path.write_text(json.dumps(manifest))
    faults = monocular.verify_merged(tmp_path / "merged")
    assert faults
    assert any("gate" in f for f in faults)


def test_verify_merged_rejects_mismatched_teacher(tmp_path: Path) -> None:
    components = export_pair(tmp_path)
    monocular.merge_pools(components, tmp_path / "merged")
    manifest_path = components["dynamic"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_sha256"] = "cd" * 32
    manifest_path.write_text(json.dumps(manifest))
    assert any("identity mismatch" in f for f in monocular.verify_merged(tmp_path / "merged"))


def test_verify_merged_reports_changed_files(tmp_path: Path) -> None:
    components = export_pair(tmp_path)
    monocular.merge_pools(components, tmp_path / "merged")
    victim = next((tmp_path / "merged/train/images").iterdir())
    victim.write_bytes(b"changed")
    assert monocular.verify_merged(tmp_path / "merged")


def test_model_predictor_wraps_infer_image(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_infer_image(model, image, cfg, **kwargs):
        calls.update(model=model, cfg=cfg, **kwargs)
        return ["detection"], "mask"

    monkeypatch.setattr("doper.infer.infer_image", fake_infer_image)
    predictor = monocular.ModelPredictor("model", "cfg", "cpu")
    keypoints = keypoints_mm()
    result = predictor(np.zeros((4, 4, 3), np.uint8), CAMERA_MATRIX, keypoints, 0.1)
    assert result == (["detection"], "mask")
    assert calls["score_thr"] == 0.1 and calls["return_mask"] is True
    assert calls["device"] == "cpu" and calls["keypoints_3d"] is keypoints


def test_ready_overlap_is_excluded_even_when_the_caller_passes_other_hashes(
    tmp_path: Path,
) -> None:
    view = write_view(tmp_path / "views_pool", "static", [1, 2, 3])
    overlap = sources.sha256_file(view / SEQUENCE / "rgb" / "000001.png")
    ready = {"validation_overlap": [{"path": f"{SEQUENCE}/rgb/000001.png", "sha256": overlap}]}
    (view / sources.READY_NAME).write_text(json.dumps(ready))
    assert overlap not in UNRELATED
    summary = monocular.export_pool(
        ScriptedPredictor(dict.fromkeys((1, 2, 3), Scenario()), keypoints_mm()),
        write_keypoints(tmp_path / "kp.json"),
        view,
        "static",
        UNRELATED,
        GateSpec(),
        tmp_path / "pool",
        teacher_sha256=TEACHER,
        obj_id=1,
    )
    manifest = read_manifest(tmp_path / "pool")
    assert manifest["reject_reasons"]["reserved_eval"] == 1
    assert summary.accepted == 2
    assert overlap not in {r["source_image_sha256"] for r in manifest["records"]}


def test_verify_merged_survives_moving_the_runs_tree(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    components = export_pair(runs)
    monocular.merge_pools(components, runs / "merged")
    recorded = read_manifest(runs / "merged")["components"]
    assert all(not Path(info["root"]).is_absolute() for info in recorded.values())
    moved = tmp_path / "elsewhere" / "moved"
    moved.parent.mkdir()
    runs.rename(moved)
    assert monocular.verify_merged(moved / "merged") == []
