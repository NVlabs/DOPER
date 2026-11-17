# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure launch contract shared by service jobs and scheduled shards."""

from pathlib import Path

import pytest

from rendering.invocation import RenderInvocation, build_render_command


@pytest.mark.parametrize("renderer", ["blender", "isaac"])
def test_launch_is_pure_and_preserves_explicit_layout(renderer, monkeypatch):
    """Build both launcher commands without filesystem access or worker-layout changes."""

    def forbidden(*args, **kwargs):
        """Fail if the operation performs a prohibited preparation or filesystem action."""
        raise AssertionError("command construction must not inspect or prepare assets")

    monkeypatch.setattr(Path, "exists", forbidden)
    cmd = build_render_command(
        RenderInvocation(
            renderer=renderer,
            repo_root=Path("/code"),
            python_bin="/python",
            config=Path("/config.yaml"),
            outdir=Path("/output"),
            num_images=7,
            seed=99,
            blend=Path("/part.blend"),
            usd=Path("/part.usda"),
            keypoints=Path("/kp.json"),
            workers=3,
            gpus=(0, 1),
            workers_per_gpu=2,
            start_idx=10,
            worker_id_offset=4,
            no_merge=True,
            resume=True,
        )
    )
    assert cmd[:2] == ["/python", "-u"]
    for flag, value in [
        ("--num-images", "7"),
        ("--seed", "99"),
        ("--start-idx", "10"),
        ("--worker-id-offset", "4"),
        ("--keypoints-json", "/kp.json"),
    ]:
        assert cmd[cmd.index(flag) + 1] == value
    assert "--no-merge" in cmd and "--resume" in cmd
    if renderer == "isaac":
        from rendering.isaac.isaac_parallel import parse_args

        args = parse_args(cmd[3:])
        assert args.gpus == [0, 1] and args.workers_per_gpu == 2
        assert args.num_images == 7 and args.isaac_python == "/python"
        assert args.no_merge and args.resume and args.keypoints_json == "/kp.json"
    else:
        from rendering.synth_parallel import build_parser

        args = build_parser().parse_args(cmd[3:])
        assert args.num_workers == 3 and args.blend_file == "/part.blend"
        assert args.num_images == 7 and args.seed == 99 and args.start_idx == 10
        assert args.worker_id_offset == 4 and args.keypoints_json == "/kp.json"
        assert args.no_merge and args.resume and args.blender_bin is None
        assert "--blender-bin" not in cmd


def test_blender_binary_is_forwarded_only_when_given():
    """An unset binary leaves DOPER_BLENDER and PATH lookup to the launcher."""
    kwargs = dict(
        renderer="blender",
        repo_root=Path("/code"),
        python_bin="/python",
        config=Path("/config.yaml"),
        outdir=Path("/output"),
        num_images=2,
        blend=Path("/part.blend"),
    )
    assert "--blender-bin" not in build_render_command(RenderInvocation(**kwargs))
    cmd = build_render_command(RenderInvocation(**kwargs, blender_bin="/opt/blender"))
    assert cmd[cmd.index("--blender-bin") + 1] == "/opt/blender"


@pytest.mark.parametrize("renderer,kwargs", [("other", {}), ("isaac", {}), ("blender", {})])
def test_invalid_renderer_or_missing_required_object_fails(renderer, kwargs):
    """Reject unknown backends and launches missing their required prepared object."""
    with pytest.raises(ValueError):
        build_render_command(
            RenderInvocation(
                renderer=renderer,
                repo_root=Path("/code"),
                python_bin="/python",
                config=Path("/cfg"),
                outdir=Path("/out"),
                num_images=2,
                **kwargs,
            )
        )
