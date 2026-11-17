# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from tiny_coco import make_mini_coco, tiny_model_config
from torch.utils.data import DataLoader

from doper.config import AugConfig, DataConfig, DoperConfig, LossConfig
from doper.dataset import DoperDataset
from doper.infer import load_model
from doper.losses import compute_losses
from doper.model import DoperNet
from doper.rotation_aux import annotation_quaternion, require_matching_rotation_metadata
from doper.targets import crop_camera_roll_matrix
from doper.training.checkpoints import load_init_weights
from doper.training.loop import evaluate
from doper.training.recipe import build_config, parse_args, validate_config


def config(root, enabled=True):
    return DoperConfig(
        data=DataConfig(root=str(root)),
        model=tiny_model_config(
            det_input_size=64,
            crop_size=64,
            neck_channels=8,
            head_channels=8,
            rotation_aux_enabled=enabled,
            rotation_num_bins=8,
            rotation_pool_size=2,
        ),
        aug=AugConfig(enabled=False),
    )


def test_rotation_joint_backward_checkpoint_and_inference(tmp_path):
    make_mini_coco(tmp_path, n_images=2, size=64)
    cfg = config(tmp_path)
    dataset = DoperDataset(cfg, "train")
    # Exercise a batch containing one labeled and one unlabeled crop.
    dataset.primary_annotation(0)["object_pose"] = np.eye(4).tolist()
    dataset.primary_annotation(1).pop("object_pose", None)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["rotation_weight"].tolist() == [1.0, 0.0]
    model = DoperNet(cfg.model).eval()
    out = model(batch["det_image"], batch["crop_image"])
    out["rotation_logits"].retain_grad()
    loss_cfg = LossConfig(
        heatmap_weight=0.0, wh_weight=0.0, offset_weight=0.0, mask_weight=0.0, keypoint_weight=0.0
    )
    loss, logs = compute_losses(out, batch, loss_cfg)
    loss.backward()
    assert logs["rotation"] > 0
    assert model.rotation_head.weight.grad.abs().sum() > 0
    assert next(model.backbone.parameters()).grad.abs().sum() > 0
    assert next(model.kpt_head.neck.parameters()).grad.abs().sum() > 0
    assert torch.count_nonzero(out["rotation_logits"].grad[1]) == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.rotation_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.rotation_head.weight)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "cfg": {"model": asdict(cfg.model)}}, checkpoint)
    restored = load_model(checkpoint, config(tmp_path, enabled=False), "cpu")
    with torch.no_grad():
        expected = model.forward_crop(batch["crop_image"])
        actual = restored.forward_crop(batch["crop_image"])
    torch.testing.assert_close(actual["rotation_logits"], expected["rotation_logits"])
    torch.testing.assert_close(actual["rotation_bins"], expected["rotation_bins"])
    metrics = evaluate(model, DataLoader(dataset, batch_size=2), cfg, "cpu")
    assert np.isfinite(metrics["rotation_top1_deg"])


def test_rotation_disabled_and_init_from_base(tmp_path):
    cfg = config(tmp_path, enabled=False)
    base = DoperNet(cfg.model).eval()
    assert not any(k.startswith("rotation") for k in base.state_dict())
    path = tmp_path / "base.pth"
    torch.save({"model": base.state_dict(), "cfg": {"model": asdict(cfg.model)}}, path)
    enabled = config(tmp_path)
    model = DoperNet(enabled.model).eval()
    load_init_weights(model, enabled, str(path), "cpu")
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(base.forward_kpt(image), model.forward_kpt(image))
    # Caller settings must not accidentally enable an absent head on inference load.
    loaded = load_model(path, enabled, "cpu")
    assert not loaded.cfg.rotation_aux_enabled


def test_rotation_pose_conventions_and_crop_roll(tmp_path):
    rotation = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    q = annotation_quaternion({"object_pose": pose})
    np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), rotation, atol=1e-6)
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("y", 35, degrees=True).as_matrix()
    world_pose = camera @ np.diag([1.0, -1.0, -1.0, 1.0]) @ pose
    np.testing.assert_allclose(
        annotation_quaternion({"object_pose": world_pose, "camera_extrinsics": camera}),
        q,
        atol=1e-6,
    )
    make_mini_coco(tmp_path, n_images=2, size=64)
    cfg = config(tmp_path)
    cfg.aug.enabled = True
    cfg.aug.crop_rot90_prob = 1.0
    cfg.aug.crop_rotate_prob = 0.0
    dataset = DoperDataset(cfg, "train")
    dataset.primary_annotation(0)["object_pose"] = pose.tolist()
    sample = dataset[0]
    actual = sample["rotation_quaternion"].numpy()
    expected = crop_camera_roll_matrix(float(sample["crop_rotation_deg"])) @ rotation
    np.testing.assert_allclose(
        Rotation.from_quat(actual[[1, 2, 3, 0]]).as_matrix(), expected, atol=1e-6
    )
    assert annotation_quaternion({}) is None
    with pytest.raises(ValueError, match="proper orthonormal"):
        annotation_quaternion({"object_pose": np.diag([2.0, 1.0, 1.0, 1.0])})


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--keypoint-head", "simcc"], "requires"),
        (["--rotation-num-bins", "1"], "bins"),
        (["--rotation-loss-weight", "nan"], "finite"),
    ],
)
def test_rotation_cli_rejects_invalid_settings(extra, match):
    args = parse_args(["--data-root", "unused", "--rotation-aux", *extra])
    with pytest.raises(ValueError, match=match):
        validate_config(build_config(args), args)


def test_rotation_resume_metadata_guard(tmp_path):
    cfg = config(tmp_path)
    metadata = asdict(cfg.model)
    require_matching_rotation_metadata(metadata, cfg.model)
    cfg.model.rotation_logit_scale = 12.0
    with pytest.raises(ValueError, match="rotation_logit_scale"):
        require_matching_rotation_metadata(metadata, cfg.model)


def test_rotation_train_loop_resume_and_missing_labels(tmp_path):
    import json

    from doper.checkpoint_io import load_checkpoint
    from doper.train import main

    root = tmp_path / "data"
    make_mini_coco(root, n_images=2, size=64)
    run = tmp_path / "run"
    arguments = [
        "--data-root",
        str(root),
        "--work-dir",
        str(run),
        "--backbone",
        "resnet18",
        "--allow-nonstandard-backbone",
        "--no-pretrained",
        "--no-require-pretrained-backbone",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--max-iters-per-epoch",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--det-size",
        "64",
        "--crop-size",
        "64",
        "--rotation-aux",
        "--rotation-num-bins",
        "8",
        "--no-aug",
    ]
    main([*arguments, "--dry-run"])
    assert not run.exists()
    main(arguments)
    first = load_checkpoint(run / "last.pth", "cpu")
    assert first["cfg"]["model"]["rotation_aux_enabled"]
    assert np.isfinite(first["metrics"]["rotation_top1_deg"])
    main([*arguments, "--epochs", "2", "--resume", str(run / "last.pth")])
    resumed = load_checkpoint(run / "last.pth", "cpu")
    assert resumed["epoch"] == 1
    torch.testing.assert_close(resumed["model"]["rotation_bins"], first["model"]["rotation_bins"])
    assert not torch.equal(
        resumed["model"]["rotation_head.weight"], first["model"]["rotation_head.weight"]
    )
    for split in ("train", "val"):
        path = root / split / "annotations.json"
        contents = json.loads(path.read_text())
        for annotation in contents["annotations"]:
            annotation.pop("object_pose", None)
        path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="requires training annotations"):
        main([*arguments, "--work-dir", str(tmp_path / "missing")])


def test_rotation_unlabeled_batch_has_zero_loss_and_gradient():
    from doper.rotation_aux import rotation_aux_loss, uniform_rotation_bins

    bins = uniform_rotation_bins(8, 42)
    torch.testing.assert_close(bins, uniform_rotation_bins(8, 42))
    logits = torch.randn(2, 8, requires_grad=True)
    loss = rotation_aux_loss(
        logits, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2), bins, sample_weight=torch.zeros(2)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_rotation_background_sample_has_no_rotation_supervision(tmp_path):
    from test_background_data import _cfg

    from doper.background_data import BackgroundNegativeDataset

    cfg = _cfg(tmp_path)
    cfg.model.rotation_aux_enabled = True
    sample = BackgroundNegativeDataset(cfg)[0]
    assert sample["rotation_weight"].item() == 0
    torch.testing.assert_close(sample["rotation_quaternion"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
