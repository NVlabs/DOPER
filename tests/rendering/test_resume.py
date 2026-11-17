# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from rendering import resume as R  # noqa: N812
from rendering.dataset_contract import merge_annotations


def _image(idx: int, name: str | None = None) -> dict:
    return {"id": idx, "file_name": name or f"{idx:06d}.png", "width": 8, "height": 8}


def _ann(idx: int, mask: str | None = None) -> dict:
    ann = {"id": idx, "image_id": idx, "keypoints": [1, 2, 2], "bbox": [0, 0, 1, 1]}
    if mask:
        ann["mask_file"] = mask
    return ann


def _write_partial(outdir: Path, worker_id: int, images: list, annotations: list) -> None:
    Path(R.partial_annotations_path(str(outdir), worker_id)).write_text(
        json.dumps({"images": images, "annotations": annotations})
    )


def test_load_resume_state_without_checkpoint(tmp_path: Path):
    state = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=10, images_per_sample=1, render_masks=False
    )
    assert state.done_indices == set() and state.images == [] and state.annotations == []


def test_load_resume_state_mono_requires_image_and_mask_on_disk(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in (3, 4, 6):
        (tmp_path / "images" / f"{i:06d}.png").write_bytes(b"png")
    (tmp_path / "masks" / "000003.png").write_bytes(b"png")
    (tmp_path / "masks" / "000006.png").write_bytes(b"png")
    # 3: complete; 4: mask missing; 5: image missing; 6: outside this worker's range; 7: no image
    _write_partial(
        tmp_path,
        2,
        [_image(3), _image(4), _image(5), _image(6), _image(7)],
        [
            _ann(3, "masks/000003.png"),
            _ann(4, "masks/000004.png"),
            _ann(5, "masks/000005.png"),
            _ann(6, "masks/000006.png"),
            _ann(7, "masks/000007.png"),
        ],
    )
    state = R.load_resume_state(
        str(tmp_path), 2, start_idx=3, end_idx=6, images_per_sample=1, render_masks=True
    )
    assert state.done_indices == {3}
    assert [img["id"] for img in state.images] == [3] and state.annotations[0][
        "mask_file"
    ] == "masks/000003.png"
    # without masks, 4 counts as done too; 5 and 7 still have no image, 6 is out of range
    state = R.load_resume_state(
        str(tmp_path), 2, start_idx=3, end_idx=6, images_per_sample=1, render_masks=False
    )
    assert state.done_indices == {3, 4}


def test_load_resume_state_stereo_needs_both_views(tmp_path: Path):
    (tmp_path / "images").mkdir()
    for name in ("000000_left.png", "000000_right.png", "000001_left.png"):
        (tmp_path / "images" / name).write_bytes(b"png")
    _write_partial(
        tmp_path,
        0,
        [_image(0, "000000_left.png"), _image(1, "000000_right.png"), _image(2, "000001_left.png")],
        [_ann(0), _ann(1), _ann(2)],
    )
    state = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=4, images_per_sample=2, render_masks=False
    )
    assert state.done_indices == {0}  # sample 1 only has its left view checkpointed
    assert [img["id"] for img in state.images] == [0, 1]


def test_sort_by_image_id_keeps_pairs_aligned():
    images = [_image(7), _image(2), _image(5)]
    annotations = [_ann(7), _ann(2), _ann(5)]
    imgs, anns = R.sort_by_image_id(images, annotations)
    assert [i["id"] for i in imgs] == [2, 5, 7] and [a["image_id"] for a in anns] == [2, 5, 7]


def test_cadence_tracker_applies_first_sample_then_group_changes():
    """A resume may start mid-group: the first rendered sample must apply everything."""
    t = R.CadenceTracker()
    assert t.changed("material", 107, 5) is True  # 107 // 5 = 21, nothing applied yet
    assert t.changed("material", 108, 5) is False  # same group
    assert t.changed("material", 110, 5) is True  # group 22
    assert t.changed("ground", 110, 3) is True  # independent field; group 36
    assert t.changed("ground", 111, 3) is True  # group 37
    assert t.changed("ground", 112, 3) is False
    assert t.changed("ground", 113, 3) is False
    assert t.changed("ground", 114, 3) is True  # group 38
    assert t.changed("background", 5, 1) and t.changed("background", 6, 1)  # every sample
    assert t.changed("x", 3, 0) is True  # change_every <= 0 behaves like 1
    assert t.changed("x", 4, 0) is True


def test_cadence_tracker_matches_record_grouping():
    """Consecutive samples share a draw exactly when index // change_every agrees."""
    t = R.CadenceTracker()
    applied = [i for i in range(20) if t.changed("m", i, 5)]
    assert applied == [0, 5, 10, 15]
    t = R.CadenceTracker()
    applied = [i for i in range(7, 20) if t.changed("m", i, 5)]  # resumed at 7
    assert applied == [7, 10, 15]


def test_flush_policy_triggers_on_count_or_time():
    now = [100.0]
    policy = R.FlushPolicy(3, 60.0, clock=lambda: now[0])
    assert [policy.should_flush() for _ in range(2)] == [False, False]
    assert policy.should_flush() is True  # third sample
    policy.flushed()
    assert policy.should_flush() is False
    now[0] += 61.0
    assert policy.should_flush() is True  # time trigger with only two samples since the flush
    policy.flushed()
    # disabled triggers
    never = R.FlushPolicy(0, 0, clock=lambda: now[0])
    now[0] += 1e6
    assert all(not never.should_flush() for _ in range(500))
    count_only = R.FlushPolicy(2, 0, clock=lambda: now[0])
    assert [count_only.should_flush() for _ in range(3)] == [False, True, True]


def test_merge_annotations_ignores_partial_checkpoints_and_tmp_files(tmp_path: Path):
    def worker_file(worker_id: int, ids: list[int]) -> dict:
        return {
            "worker_id": worker_id,
            "start_idx": ids[0],
            "end_idx": ids[-1] + 1,
            "images_per_sample": 1,
            "render_masks": False,
            "mask_dir": None,
            "stereo": {"enabled": False},
            "images": [_image(i) for i in ids],
            "annotations": [_ann(i) for i in ids],
        }

    (tmp_path / "annotations_worker_000.json").write_text(json.dumps(worker_file(0, [0, 1])))
    (tmp_path / "annotations_worker_001.json").write_text(json.dumps(worker_file(1, [2, 3])))
    (tmp_path / "annotations_worker_001.partial.json").write_text(json.dumps(worker_file(1, [2])))
    (tmp_path / "annotations_worker_002.partial.json").write_text(json.dumps(worker_file(2, [4])))
    (tmp_path / "annotations_worker_003.json.tmp").write_text("{")
    counts = merge_annotations(tmp_path, None)
    assert counts["images"] == 4 and counts["annotations"] == 4
    merged = json.loads((tmp_path / "annotations.json").read_text())
    assert sorted(img["id"] for img in merged["images"]) == [0, 1, 2, 3]


def _worker_file(outdir: Path, worker_id: int, ids: list[int], *, partial: bool = False) -> Path:
    path = Path(
        R.partial_annotations_path(str(outdir), worker_id)
        if partial
        else R.final_annotations_path(str(outdir), worker_id)
    )
    path.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "start_idx": ids[0],
                "end_idx": ids[-1] + 1,
                "images_per_sample": 1,
                "images": [_image(i) for i in ids],
                "annotations": [_ann(i) for i in ids],
            }
        )
    )
    return path


def test_consolidate_checkpoints_is_worker_layout_independent(tmp_path: Path):
    """A shard rendered by 4 workers of 5 resumes with 2 workers of 10: nothing is re-rendered."""
    (tmp_path / "images").mkdir()
    # old layout: shard [0, 20) as workers 0..3 x 5 samples; worker 2 died mid-way (partial),
    # worker 3 never ran
    _worker_file(tmp_path, 0, [0, 1, 2, 3, 4])
    _worker_file(tmp_path, 1, [5, 6, 7, 8, 9])
    _worker_file(tmp_path, 2, [10, 11, 12], partial=True)
    for i in list(range(13)) + [16]:  # 16 exists on disk but was never checkpointed -> re-rendered
        (tmp_path / "images" / f"{i:06d}.png").write_bytes(b"png")
    # a neighboring shard's worker file must be left alone
    other = _worker_file(tmp_path, 7, [20, 21])

    info = R.consolidate_checkpoints(str(tmp_path), start_idx=0, end_idx=20, images_per_sample=1)
    assert info["records"] == 13 and len(info["consumed"]) == 3
    assert Path(info["snapshot"]).is_file() and other.is_file()
    assert not (tmp_path / "annotations_worker_000.json").exists()
    assert not (tmp_path / "annotations_worker_002.partial.json").exists()
    consumed_dir = next((tmp_path / "resume" / "consumed").iterdir())
    assert sorted(p.name for p in consumed_dir.iterdir()) == [
        "annotations_worker_000.json",
        "annotations_worker_001.json",
        "annotations_worker_002.partial.json",
    ]

    # new layout: 2 workers of 10; each skips exactly the checkpointed samples in its range
    a = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=10, images_per_sample=1, render_masks=False
    )
    b = R.load_resume_state(
        str(tmp_path), 1, start_idx=10, end_idx=20, images_per_sample=1, render_masks=False
    )
    assert a.done_indices == set(range(10)) and [i["id"] for i in a.images] == list(range(10))
    assert b.done_indices == {10, 11, 12}
    # consolidating again is idempotent (snapshot is a source of itself)
    again = R.consolidate_checkpoints(str(tmp_path), start_idx=0, end_idx=20, images_per_sample=1)
    assert again["records"] == 13 and again["consumed"] == []


def test_consolidate_prefers_final_over_partial_and_skips_unreadable(tmp_path: Path):
    (tmp_path / "images").mkdir()
    for i in range(4):
        (tmp_path / "images" / f"{i:06d}.png").write_bytes(b"png")
    final = _worker_file(tmp_path, 0, [0, 1])
    final_payload = json.loads(final.read_text())
    final_payload["annotations"][0]["bbox"] = [
        9,
        9,
        9,
        9,
    ]  # distinguishable from the partial's copy
    final.write_text(json.dumps(final_payload))
    _worker_file(tmp_path, 0, [0, 1, 2, 3], partial=True)  # stale partial overlapping the final
    Path(R.partial_annotations_path(str(tmp_path), 5)).write_text("{not json")
    info = R.consolidate_checkpoints(str(tmp_path), start_idx=0, end_idx=10, images_per_sample=1)
    snap = json.loads(Path(info["snapshot"]).read_text())
    assert [i["id"] for i in snap["images"]] == [0, 1, 2, 3]
    assert snap["annotations"][0]["bbox"] == [9, 9, 9, 9]  # the final file won for sample 0


def test_load_resume_state_own_partial_beats_snapshot_after_relaunch(tmp_path: Path):
    """After a watchdog relaunch the worker's fresh partial has all the snapshot has, and more."""
    (tmp_path / "images").mkdir()
    for i in range(6):
        (tmp_path / "images" / f"{i:06d}.png").write_bytes(b"png")
    _worker_file(tmp_path, 0, [0, 1], partial=True)
    R.consolidate_checkpoints(
        str(tmp_path), start_idx=0, end_idx=10, images_per_sample=1
    )  # moves the partial
    _worker_file(
        tmp_path, 0, [0, 1, 2, 3, 4, 5], partial=True
    )  # relaunched worker's newer checkpoint
    state = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=10, images_per_sample=1, render_masks=False
    )
    assert state.done_indices == set(range(6))


def test_resume_joins_reordered_annotations_by_image_id(tmp_path):
    (tmp_path / "images").mkdir()
    for i in (0, 1):
        (tmp_path / "images" / f"{i:06d}.png").write_bytes(b"png")
    _write_partial(tmp_path, 0, [_image(0), _image(1)], [_ann(1), _ann(0)])
    state = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=2, images_per_sample=1, render_masks=False
    )
    assert state.done_indices == {0, 1}
    assert [a["image_id"] for a in state.annotations] == [0, 1]


def test_resume_does_not_reuse_an_image_outside_dataset(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    _write_partial(tmp_path, 0, [_image(0, str(outside))], [_ann(0)])
    state = R.load_resume_state(
        str(tmp_path), 0, start_idx=0, end_idx=1, images_per_sample=1, render_masks=False
    )
    assert not state.done_indices


def test_resume_skips_damaged_sources_and_keeps_valid_snapshot(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "000003.png").write_bytes(b"png")
    partial = Path(R.partial_annotations_path(str(tmp_path), 0))
    partial.write_text('{"images": [')
    snapshot = tmp_path / "valid.json"
    snapshot.write_text(json.dumps({"images": [_image(3)], "annotations": [_ann(3)]}))
    unreadable = tmp_path / "directory.json"
    unreadable.mkdir()
    state = R.load_resume_state(
        str(tmp_path),
        0,
        start_idx=0,
        end_idx=5,
        images_per_sample=1,
        render_masks=False,
        snapshots=[str(unreadable), str(snapshot)],
    )
    assert state.done_indices == {3}
    assert state.images == [_image(3)]


def test_count_shard_images_uses_full_sample_prefix_for_stereo(tmp_path: Path):
    from rendering.synth_parallel import count_shard_images

    images = tmp_path / "images"
    images.mkdir()
    for sample in (999999, 1000000, 1000001, 2000000):
        for view in ("left", "right"):
            (images / f"{sample:06d}_{view}.png").touch()
    (images / "1000000junk.png").touch()
    (images / "preview.png").touch()
    assert count_shard_images(tmp_path, 1000000, 1000001) == 2
    assert count_shard_images(tmp_path, 999999, 1000001) == 4


def test_count_shard_images_mono(tmp_path: Path):
    from rendering.synth_parallel import count_shard_images

    images = tmp_path / "images"
    images.mkdir()
    for sample in (4, 5, 6):
        (images / f"{sample:06d}.png").touch()
    assert count_shard_images(tmp_path, 5, 6) == 1
