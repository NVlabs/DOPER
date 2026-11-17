---
name: doper-cad-workflow
description: Train DOPER, a keypoint-based 6-DoF pose model, on the user's own CAD model, from keypoint selection and Blender rendering through training, evaluation, ONNX export and a TensorRT benchmark.
---

# Train DOPER on your own CAD model

Run every command from the repository root with the environment active. If `.venv` is missing,
run `uv sync` first (no need to ask), then `source .venv/bin/activate`. Blender 4.2 LTS or newer
is required: `blender` on `PATH`, or `export DOPER_BLENDER=/path/to/blender`.
`examples/minimal/README.md` runs this same chain on a bundled model; read it first. "The CAD
example dry run" means steps 1-4 run for real, including real renders of the small splits (16
train, 8 val, one worker), and then step 5, the training `--dry-run`, against those splits: only
training is a dry run. Ask the user for the CAD file and its units (`m`, `mm`, `cm`, `in`).

```bash
BLENDER="${DOPER_BLENDER:-blender}"
WORK=work/my_object && mkdir -p "$WORK"
CAD=path/to/model.obj   # the user's CAD file
```

## Pairing rules

- `--center-geometry` (step 1 and 3) goes with `--object-frame aabb_centered` (step 2): the
  keypoints are selected in the frame of the centered mesh.
- The keypoint prefix is `KP_<mesh_name>_` and must equal `object.keypoint_prefix` in the render
  config; `object.mesh_name` equals `--object-name`. The example uses `object` and `KP_object_`.
- `num_keypoints` is the same in the render config (`object.num_keypoints`), the selector
  (`--num-keypoints`) and training (`--num-keypoints`, or `num_keypoints` in the config).

## Steps

1. Import the CAD, center it on its bounding box and export the centered mesh. The keypoints this
   run samples are not used; only the centered mesh is.

```bash
"$BLENDER" --background --factory-startup --python rendering/mesh_setup.py -- \
  --input "$CAD" --output "$WORK/draft.blend" \
  --source-units m --center-geometry --centered-mesh-output "$WORK/centered.stl"
```

2. Select the keypoints on the centered mesh.

```bash
python -m doper.keypoint_selection --mesh "$WORK/centered.stl" --source-units m \
  --object-frame aabb_centered --out-json "$WORK/keypoints.json" \
  --keypoint-prefix KP_object_ --num-keypoints 64 --object-name object
```

3. Author the keypoints into the `.blend` and a USD copy of the object.

```bash
"$BLENDER" --background --factory-startup --python rendering/mesh_setup.py -- \
  --input "$CAD" --output "$WORK/object.blend" \
  --source-units m --center-geometry --keypoints-input "$WORK/keypoints.json" \
  --usd-output "$WORK/object.usda" --keypoints-output "$WORK/keypoints_3d.json"
```

4. Render the train and val splits with a render config: a copy of `configs/synth_example.yaml`
   (or `examples/minimal/render.yaml`) with `object.mesh_name`, `object.keypoint_prefix` and
   `object.num_keypoints` set as above. `--dry-run` is an optional preflight: it validates the
   config and inputs and starts no Blender, and the real render must still follow. Each output
   directory is a COCO split (`images/`, `masks/`, `annotations.json`).

```bash
python rendering/synth_parallel.py --blend-file "$WORK/object.blend" --usd "$WORK/object.usda" \
  --config examples/minimal/render.yaml --outdir "$WORK/data/train" \
  --num-images 16 --num-workers 1 --seed 0
python rendering/synth_parallel.py --blend-file "$WORK/object.blend" --usd "$WORK/object.usda" \
  --config examples/minimal/render.yaml --outdir "$WORK/data/val" \
  --num-images 8 --num-workers 1 --seed 1
```

   Those counts are the example's smoke size; ask the user how many images a real run needs.
   The renders use EEVEE, which needs OpenGL. Without a usable GPU, set `render.engine: CYCLES`
   (CPU) in the render config copy.
5. Validate the training setup against the real splits. The dry run checks that both splits and
   the keypoint file exist and that `--num-keypoints` matches the file, prints the plan and exits;
   it builds no model and downloads nothing.

```bash
python -m doper.train --data-root "$WORK/data" --keypoints-3d "$WORK/keypoints_3d.json" \
  --num-keypoints 64 --work-dir "$WORK/run" --dry-run
```

   `--config FILE` takes the same keys; copy `doper/recipes/train_example.yaml`, set the paths
   and keep its architecture fields. CLI flags override the file.

## Next steps (documented, not run here)

- Train (GPU, network for the pretrained backbone): the command from step 5 without `--dry-run`,
  or `python -m doper.train --config FILE`. `doper train` is for the paper recipes, not this.
  Without a GPU, only the README's two-iteration CPU smoke train is practical.
- Evaluate on an image: `python -m doper.infer --checkpoint CKPT --image IMG
  --keypoints-3d "$WORK/keypoints_3d.json" --intrinsics camera_intrinsics.json`.
- Export to ONNX (needs the `tensorrt` extra): `python -m doper.deployment.onnx_export
  --checkpoint CKPT --keypoints-3d "$WORK/keypoints_3d.json" --camera CAMERA_JSON
  --frames-dir FRAMES --output BUNDLE`.
- Benchmark the bundle with TensorRT (GPU): `python -m doper.deployment.benchmark --bundle BUNDLE
  --output RESULT_DIR`.

## Do not

- Do not enable domain randomization (HDRIs, textures, distractors, `robustness:`,
  `robust_appearance:`) unasked; suggest it and point at the "Domain randomization" notes in
  `examples/minimal/README.md`.
- Do not set `SYNTH_ALLOW_MISSING_ASSETS` or `DOPER_ALLOW_BAD_DATA` without the user's approval.
