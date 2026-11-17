# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NuRec template recipe's whole command chain plans under ``--dry-run``, stage to stage.

The robot_handeye path, in the order docs/nurec.md gives: prepare capture -> reconstruct ->
synthetic -> train base -> evaluate base (the registration keypoints) -> register -> accept the
registration -> render -> labels -> train adapt -> evaluate adapted -> report. Every command
plans without reading or writing data, and each input a stage derives itself is the output an
earlier plan named.
"""

from __future__ import annotations

import re
from pathlib import Path

from doper import cli
from tests.unit.nurec_fixtures import NUREC_TEMPLATE, write_bundle

PlanStep = dict[str, dict[str, str]]
_STEP = re.compile(r"^\[doper\] step \d+/\d+: ")
_VALUE = re.compile(r"^\[doper\]\s+(?P<kind>in|out) (?P<key>[^:]+): (?P<value>.*)$")


def _plan(argv: list[str], capsys) -> list[PlanStep]:  # noqa: ANN001
    """Run one command with ``--dry-run``; return its steps' inputs and outputs as printed."""
    assert cli.main([*argv, "--dry-run"]) == 0, argv
    steps: list[PlanStep] = []
    for line in capsys.readouterr().out.splitlines():
        if _STEP.match(line):
            steps.append({"in": {}, "out": {}})
        elif (value := _VALUE.match(line)) is not None:
            steps[-1][value["kind"]][value["key"]] = value["value"]
    assert steps, argv
    return steps


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_dry_run_chain_nurec(tmp_path, capsys):
    # Arrange: a fixture capture bundle and empty data and run roots.
    bundle, root, runs = tmp_path / "bundle", tmp_path / "root", tmp_path / "runs"
    write_bundle(bundle)
    root.mkdir()
    before = _tree(tmp_path)
    artifact = tmp_path / "scene" / "last.usdz"  # the NuRec scene, trained outside DOPER
    assets = [f"--{name}-dir={tmp_path / name}" for name in ("hdri", "texture", "distractor")]
    paths = ["--root", str(root), "--runs", str(runs)]
    prepare = ["prepare", str(NUREC_TEMPLATE), "--root", str(root), "--stage"]
    evaluate = ["evaluate", str(NUREC_TEMPLATE), *paths, "--bundle", str(bundle)]
    # Act: plan every stage in order. Paths a user passes on the command line (the checkpoint,
    # keypoint frames, evaluation directories) are taken from the earlier plan's printed
    # outputs; they are hand-offs by argv, so they are not asserted below.
    (capture,) = _plan([*prepare, "capture", "--bundle", str(bundle)], capsys)
    (reconstruct,) = _plan([*prepare, "reconstruct"], capsys)
    synthetic = _plan([*prepare, "synthetic", *assets], capsys)
    (base,) = _plan(["train", str(NUREC_TEMPLATE), *paths, "--stage", "base"], capsys)
    (base_eval,) = _plan([*evaluate, "--checkpoint", base["out"]["best"],
                          "--out", str(runs / "eval" / "base")], capsys)  # fmt: skip
    (register,) = _plan([*prepare, "register", "--artifact", str(artifact),
                         "--keypoint-frames", base_eval["out"]["frames"]], capsys)  # fmt: skip
    (accept,) = _plan([*prepare, "register", "--accept-registration", "--reviewer-note",
                       "overlays checked"], capsys)  # fmt: skip
    render_plan, render = _plan([*prepare, "render", "--artifact", str(artifact)], capsys)
    geometry, sam, qualify, package = _plan(
        [*prepare, "labels", "--artifact", str(artifact), "--sam2-weights", "sam2.pt"], capsys
    )
    (adapt,) = _plan(["train", str(NUREC_TEMPLATE), *paths, "--stage", "adapt"], capsys)
    (adapt_eval,) = _plan([*evaluate, "--checkpoint", adapt["out"]["best"],
                           "--out", str(runs / "eval" / "adapt")], capsys)  # fmt: skip
    _plan(["report", str(NUREC_TEMPLATE), "--runs", str(runs),
           "--baseline", str(Path(base_eval["out"]["summary"]).parent),
           "--adapted", str(Path(adapt_eval["out"]["summary"]).parent)], capsys)  # fmt: skip
    # Assert: each input a stage derives from --root/--runs is an earlier plan's output, and
    # nothing was written.
    assert reconstruct["in"]["capture"] == capture["out"]["receipt"]
    assert reconstruct["in"]["frames"] == capture["out"]["frames"]
    reconstruction = str(Path(reconstruct["out"]["metadata"]).parent)
    assert {str(Path(v).parent) for v in reconstruct["out"].values()} == {reconstruction}
    assets_step, synth_render, audit, split = synthetic
    assert synth_render["in"]["assets"] == assets_step["out"]["assets"]
    assert audit["in"]["render"] == synth_render["out"]["render"]
    assert audit["in"]["config"] == synth_render["out"]["config"]
    assert audit["in"]["assets"] == assets_step["out"]["receipt"]
    assert split["in"]["render"] == synth_render["out"]["render"]
    assert split["in"]["audit"] == audit["out"]["audit"]
    assert base["in"]["dataset"] == split["out"]["dataset"]
    assert register["in"]["reconstruction"] == reconstruction
    assert register["in"]["capture"] == str(Path(capture["out"]["receipt"]).parent)
    assert accept["in"]["candidate"] == register["out"]["candidate"]
    assert render_plan["in"]["reconstruction"] == reconstruction
    assert render_plan["in"]["frames"] == capture["out"]["frames"]
    assert render["in"]["trajectory"] == render_plan["out"]["trajectory"]
    assert geometry["in"]["views"] == render_plan["out"]["views"]
    assert geometry["in"]["registration"].startswith(accept["out"]["registration"] + " ")
    assert sam["in"]["frames"] == render["out"]["frames"]
    assert qualify["in"]["frames"] == render["out"]["frames"]
    assert qualify["in"]["labels"] == geometry["out"]["labels"]
    assert qualify["in"]["cad masks"] == geometry["out"]["masks"]
    assert qualify["in"]["masks"] == sam["out"]["masks"]
    assert package["in"]["qualified"] == qualify["out"]["qualified"]
    assert adapt["in"]["dataset"] == package["out"]["dataset"]
    assert adapt["in"]["init from"] == base["out"]["best"]
    assert _tree(tmp_path) == before
