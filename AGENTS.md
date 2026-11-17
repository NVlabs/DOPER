# DOPER agent guide

DOPER renders synthetic COCO-style data from a CAD model with Blender, trains a keypoint-based
6-DoF pose model, reproduces the paper's BOP and SPEED+ results, and adapts to a real object from
NuRec renders. [README.md](README.md) is the user documentation; this file is the short operating
guide for a coding agent.

## Setup

- Python 3.11 or newer, managed with [uv](https://docs.astral.sh/uv/). The first step in a fresh
  checkout is `uv sync` (plus any extras the task needs); run it without asking, because it only
  installs the project's locked dependencies into `.venv`. Then run commands inside that
  environment, with `uv run ...` or after `source .venv/bin/activate`. Use `uv`, not `pip`.
- Extras are opt-in (`uv sync --extra bop`; repeat `--extra` for several). `uv sync --all-extras
  --group dev` gives the full development environment. The extras are listed in the README.
- Rendering needs Blender 4.2 LTS or newer: `blender` on `PATH`, or `export
  DOPER_BLENDER=/path/to/blender`.
- Training with the default recipe downloads the pretrained DINOv3 ConvNeXt backbone, so it needs
  network access. Offline, the smoke train in `examples/minimal/README.md` skips it.

## Core workflows

- "The CAD example dry run" (or "dry run of the CAD example") means steps 1-4 of
  `examples/minimal/README.md` run for real, with real renders of the small splits (16 train, 8
  val, one worker), and then `python -m doper.train ... --dry-run` against those splits. Only
  training is a dry run; `synth_parallel.py --dry-run` is optional preflight, not a substitute
  for rendering.
- Every `doper` verb (`prepare`, `train`, `review`, `evaluate`, `report`) takes `--dry-run`, which
  prints the ordered steps and touches no data. Run it first, show the user, then run for real.
  `--root` (dataset) and `--runs` (outputs) are always explicit.
- Reproduce one paper table: pick a recipe (`handal`, `hopev2`, `hot3d`, `tless`, `speedplus`,
  `shirt`) and follow its section of the README, for example
  `doper prepare handal --root data/bop --dry-run`.
- Train on the user's own CAD model: use the skill `skills/doper-cad-workflow/SKILL.md`.
  `doper train` serves the paper recipes only; a CAD model of your own trains with
  `python -m doper.train --config FILE`, starting from `doper/recipes/train_example.yaml`.
  `examples/minimal/README.md` runs the whole chain on a bundled model.

## Where things are

- Packaged recipes: `doper/recipes/*.yaml`; dataset profiles (pinned sources and checksums):
  `doper/recipes/profiles/*.json`.
- Starter render recipe: `configs/synth_example.yaml`. Rendering code: `rendering/`. Training
  entry point: `doper/train.py`. Docs: `docs/`.
- Required paths are never guessed: ask the user for the dataset root, the output directory, the
  CAD file and its units (`--source-units`), the checkpoint and the camera files.

## Guardrails

- Consent flags record a human decision. Never add one on your own; ask the user, naming what
  the flag allows: `--accept-license`, `--accept-new-layout`, `--accept-non-paper-masks`,
  `--accept-registration`, `--accept-unreviewed`, `--allow-test-inputs`, `--allow-partial-bank`,
  `--allow-unpinned-checkpoint`, `--allow-nonstandard-backbone`.
- `--force` rebuilds or rescores existing outputs and `--force-from STEP` (NuRec `render`,
  `labels` and `synthetic` only) rebuilds from one step onward; both overwrite results. Use them
  only when the user asks for a rebuild.
- Never set `DOPER_ALLOW_BAD_DATA` or `SYNTH_ALLOW_MISSING_ASSETS` without the user's approval.
  They let a run continue past bad data or missing assets.
- No silent overrides: do not change a recipe, a pinned checksum or a training default to make a
  command pass. If a command refuses, report the message and the options to the user.

## Rendering guidance

- The starter recipes randomize object pose, camera distance and framing, and procedural ground
  and world backgrounds only.
- Stronger domain randomization is opt-in. Suggest it when the deployment needs it; do not enable
  it on your own: HDRI and texture backgrounds (`scene.hdri_dir`, `scene.texture_dir`),
  distractor objects (`scene.distractor_dir`), truncation and occlusion (`robustness:`), and
  glare and shadow views (`robust_appearance:`). The "Domain randomization" notes in
  `examples/minimal/README.md` give the keys and defaults.
- Keep `num_keypoints` equal across the render config, the keypoint selection and training.
- On a machine without a usable GPU, set `render.engine: CYCLES` in a copy of the render config.

## Verification

- `make check` (ruff and pyright) and `make test` (everything not marked `gpu` or `isaac`).
  `make format` fixes formatting.
- Tests marked `blender` need Blender 4.2 or newer and skip when it is absent; `gpu` tests need
  CUDA and TensorRT; `isaac` tests need the Isaac Sim 6.0.1 container; some tests skip without an
  optional extra (`onnx`, `usd`, `numba`).
- Run `make format && make check && make test` before reporting a change as done.
