# Minimal example: 3DBenchy

Renders a small synthetic dataset for the [#3DBenchy](https://www.3dbenchy.com) model (about
60 x 31 x 48 mm, an STL in millimeters) and loads it with the DOPER dataset class. 3DBenchy by
Creative Tools is in the public domain (CC0 1.0); DOPER does not ship it, and the first step
downloads it.

Requirements: Blender 4.2 LTS or newer (`blender` on `PATH`, or `export DOPER_BLENDER=/path/to/blender`)
and the `doper` package installed in the active Python environment. Run every command from the
repository root; the first one picks the Blender binary.

```bash
BLENDER="${DOPER_BLENDER:-blender}"
WORK=work/minimal && mkdir -p "$WORK"
curl -L -o "$WORK/3DBenchy.stl" \
  https://raw.githubusercontent.com/CreativeTools/3DBenchy/b542785f9642ab39db8a778a0ccc934b3ebd829a/Single-part/3DBenchy.stl
echo "6ab57f1c3f8e86bc3cbd302c6fa6270acf06277c6335454e922419c25d42e97e  $WORK/3DBenchy.stl" | sha256sum -c
```

"Dry run of this example" means steps 1-4 run for real (the renders included) and step 6 is the
training `--dry-run`; only training is a dry run.

1. Import the CAD, center it on its bounding box and export the centered mesh. The sampled
   keypoints this run also writes (`object_keypoints_3d.json`) are not used; only the centered
   mesh is.

```bash
"$BLENDER" --background --factory-startup --python rendering/mesh_setup.py -- \
  --input "$WORK/3DBenchy.stl" --output "$WORK/draft.blend" \
  --source-units mm --center-geometry --centered-mesh-output "$WORK/centered.stl"
```

2. Select 64 keypoints on the centered mesh (hull-anchored selector). The prefix must be
   `KP_object_` to match `object.keypoint_prefix` in `render.yaml`.

```bash
python -m doper.keypoint_selection --mesh "$WORK/centered.stl" --source-units m \
  --object-frame aabb_centered --out-json "$WORK/keypoints.json" \
  --keypoint-prefix KP_object_ --num-keypoints 64 --object-name object
```

3. Author the selected keypoints into the `.blend` and a USD copy of the object.

```bash
"$BLENDER" --background --factory-startup --python rendering/mesh_setup.py -- \
  --input "$WORK/3DBenchy.stl" --output "$WORK/object.blend" \
  --source-units mm --center-geometry --keypoints-input "$WORK/keypoints.json" \
  --usd-output "$WORK/object.usda" --keypoints-output "$WORK/keypoints_3d.json"
```

4. Render the train and val splits. Each output directory is a COCO split (`images/`, `masks/`,
   `annotations.json`). To check a render config before a long render, add `--dry-run` to the
   command: it validates the config and inputs, prints the worker sample ranges and starts no
   Blender and writes nothing.

```bash
python rendering/synth_parallel.py --blend-file "$WORK/object.blend" --usd "$WORK/object.usda" \
  --config examples/minimal/render.yaml --outdir "$WORK/data/train" \
  --num-images 16 --num-workers 1 --seed 0
python rendering/synth_parallel.py --blend-file "$WORK/object.blend" --usd "$WORK/object.usda" \
  --config examples/minimal/render.yaml --outdir "$WORK/data/val" \
  --num-images 8 --num-workers 1 --seed 1
```

5. Load the split with the DOPER dataset.

```bash
python - "$WORK/data" <<'PY'
import sys
from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.dataset import DoperDataset

cfg = DoperConfig(data=DataConfig(root=sys.argv[1]), model=ModelConfig(num_keypoints=64))
dataset = DoperDataset(cfg, "train")
sample = dataset[0]
print(len(dataset), "samples;", int(sample["kp_weight"].numel()), "keypoints per sample")
PY
```

6. Validate the training config against the real splits. The dry run checks that both splits and
   the keypoint file exist and that `--num-keypoints` matches the file, prints the plan and exits.
   It builds no model, downloads no weights and creates no work directory.

```bash
python -m doper.train --data-root "$WORK/data" --keypoints-3d "$WORK/keypoints_3d.json" \
  --num-keypoints 64 --work-dir "$WORK/run" --dry-run
```

7. Optional CPU smoke train: two iterations of one epoch, only to prove the training loop runs
   end to end. It is not a production recipe and its checkpoint is not usable. The two flags
   `--no-require-pretrained-backbone --no-pretrained` skip the pretrained backbone, so it also
   works offline. A real run needs a GPU, the default epochs and pretrained weights.

```bash
python -m doper.train --data-root "$WORK/data" --keypoints-3d "$WORK/keypoints_3d.json" \
  --num-keypoints 64 --work-dir "$WORK/run" --device cpu --epochs 1 --max-iters-per-epoch 2 \
  --batch-size 2 --num-workers 0 --no-require-pretrained-backbone --no-pretrained
```

## Notes

- `render.yaml` uses procedural ground and world backgrounds, so it needs no extra assets. To add
  your own, uncomment `scene.hdri_dir`, `scene.texture_dir` and `scene.distractor_dir` (see
  `configs/synth_example.yaml` for the full starter recipe) and point them at directories of
  HDRIs, texture sets and distractor meshes.
- Domain randomization. The starter recipes randomize object pose, camera distance and framing,
  and procedural ground and world backgrounds for every image. Stronger randomization is
  available but off by default; turn it on in a copy of the config when the deployment needs it:
  - Real backgrounds and clutter: set `scene.hdri_dir` and `scene.texture_dir` (with
    `scene.ground_style: texture`), and `scene.distractor_dir` with a nonzero
    `scene.num_distractors_max` for confounding objects.
  - Truncation and occlusion: add a `robustness:` section. Each image draws one mode from
    `mode_weights` (default `{easy: 0.4, clipping: 0.25, occlusion: 0.25, combined: 0.1}`);
    `clipping` pushes the object past an image edge by `clipping_fraction` (default `[0.1, 0.35]`)
    and `occlusion` puts `proxy_count` (default `[1, 3]`) capsule occluders, a rough grasp
    proxy, in front of it. Needs no assets; requires `output.render_masks: true`.
  - Specular stress: add `robust_appearance: {enabled: true, axis: z}` (the object's symmetry
    axis; `preset` mixed/glare/shadow/diffuse) for glare and cast-shadow views of metallic end
    faces. Blender only.
  Compare a run with `robustness` or `robust_appearance` against one without.
- The renders use EEVEE, which needs OpenGL. On a machine without a usable GPU, set
  `render.engine: CYCLES` (CPU) in a copy of `render.yaml`; `render.cycles_samples` and
  `render.threads` control sample count and thread count.
- For a new object, copy `configs/synth_example.yaml`, set `object.mesh_name`,
  `object.keypoint_prefix` (`KP_<mesh_name>_`) and `object.num_keypoints`, and repeat steps 1-4.
