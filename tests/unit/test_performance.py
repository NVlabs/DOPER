# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in performance settings preserve historical runtime and tensor contracts."""

import pytest
import torch

from doper.config import TrainConfig
from doper.performance import configure_runtime, image_layout, loader_options


def test_default_backend_settings_are_untouched(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    assert configure_runtime(TrainConfig(), "cpu") == torch.float16
    assert torch.backends.cudnn.benchmark
    assert torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_bf16_checks_support_without_affecting_legacy_fp16(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="unsupported"):
        configure_runtime(TrainConfig(amp_dtype="bfloat16"), "cuda")
    assert configure_runtime(TrainConfig(), "cuda") == torch.float16
    assert configure_runtime(TrainConfig(amp=False, amp_dtype="bfloat16"), "cuda") == torch.bfloat16


def test_layout_preserves_values_and_targets():
    batch = dict(
        det_image=torch.randn(2, 3, 32, 32),
        crop_image=torch.randn(2, 3, 64, 64),
        keypoint_heatmap=torch.randn(2, 64, 16, 16),
        img_id=torch.tensor([1, 2]),
    )
    result = image_layout(batch, True)
    for key in ("det_image", "crop_image"):
        assert torch.equal(result[key], batch[key])
        assert result[key].is_contiguous(memory_format=torch.channels_last)
    assert result["keypoint_heatmap"] is batch["keypoint_heatmap"]
    assert result["img_id"] is batch["img_id"]
    assert image_layout(batch, False) is batch


def test_worker_options_are_valid_for_sync_and_multiprocess():
    assert loader_options(TrainConfig(num_workers=0)) == {}
    options = loader_options(TrainConfig(num_workers=2, prefetch_factor=4, opencv_threads=1))
    assert options["prefetch_factor"] == 4
    # Test actual loader initialization, collation and worker callback pickling.
    batches = list(
        torch.utils.data.DataLoader(
            torch.arange(8),
            batch_size=4,
            num_workers=2,
            timeout=20,
            multiprocessing_context="spawn",
            **options,
        )
    )
    assert torch.equal(torch.cat(batches), torch.arange(8))
    with pytest.raises(ValueError, match="prefetch"):
        loader_options(TrainConfig(prefetch_factor=0))


def test_training_counts_partial_accumulation_and_selects_synthetic_best(tmp_path, monkeypatch):
    """Exercise the real loop on a tiny CPU fixture, including the final partial update."""
    import json
    import sys

    from doper import train
    from doper.training import loop

    class Samples(torch.utils.data.Dataset):
        ids = list(range(10))
        anns_by_img = {i: [dict(keypoints=[0.0] * 192)] for i in range(10)}

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return dict(det_image=torch.ones(3, 4, 4), crop_image=torch.ones(3, 4, 4))

    class TinyModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, det, crop):
            return self.weight * det.mean()

    optimizer_holder = []
    observed_lrs = []
    original_build_optimizer = train.build_optimizer

    def build_optimizer(model, cfg):
        optimizer = original_build_optimizer(model, cfg)
        optimizer_holder.append(optimizer)
        return optimizer

    def losses(out, *args, **kwargs):
        observed_lrs.append([g["lr"] for g in optimizer_holder[0].param_groups])
        return out.square(), {}

    monkeypatch.setattr(train, "build_optimizer", build_optimizer)
    monkeypatch.setattr(train, "build_train_set", lambda *args: Samples())
    monkeypatch.setattr(train, "DoperDataset", lambda *args: Samples())
    monkeypatch.setattr(train, "DoperNet", TinyModel)
    monkeypatch.setattr(loop, "compute_losses", losses)
    monkeypatch.setattr(train, "evaluate", lambda *args, **kw: {"kp_err_px": 2.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data-root",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "2",
            "--num-workers",
            "0",
            "--no-pretrained",
            "--channels-last",
            "--no-amp",
        ],
    )
    train.main()
    timing = json.loads((tmp_path / "work/epoch_timing.jsonl").read_text())
    assert timing["images"] == 10 and timing["microbatches"] == 5
    assert timing["optimizer_updates"] == 3 and timing["scaler_skipped_updates"] == 0
    assert timing["global_scheduler_step"] == 3
    # Two accumulated microbatches share an LR; the partial final update uses step2.
    expected_steps = (0, 0, 1, 1, 2)
    assert len(observed_lrs) == len(expected_steps)
    for observed, step in zip(observed_lrs, expected_steps, strict=True):
        expected = train.lr_at(step, 3, 0.002, 500)
        assert observed == pytest.approx([expected * 0.1, expected])

    checkpoint = torch.load(tmp_path / "work/best.pth", weights_only=False)
    assert checkpoint["best_kp"] == 2.0
    assert checkpoint["training_provenance"]["resolved_config"]["train"]["channels_last"]


def test_bf16_support_is_checked_on_requested_device(monkeypatch):
    from contextlib import contextmanager

    active = ["cuda:0"]

    @contextmanager
    def selected(device):
        original = active[0]
        active[0] = device
        try:
            yield
        finally:
            active[0] = original

    monkeypatch.setattr(torch.cuda, "device", selected)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: active[0] == "cuda:1")
    assert configure_runtime(TrainConfig(amp_dtype="bfloat16"), "cuda:1") == torch.bfloat16
    assert active[0] == "cuda:0"
