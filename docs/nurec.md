# NuRec adaptation

DOPER includes experimental tooling for adapting a synthetic-trained model to a real object from
NuRec renders of a reconstruction of the object's captures. It is not part of the paper, and fuller
documentation will follow. The outputs and options of this tooling may change. The template
recipe's view choices and thresholds are a starting point: tune them for your object and capture.

## Requirements

- COLMAP 3.13, docker with an NVIDIA GPU, and NGC access to the NuRec images
  (`nvcr.io/nvidia/nre/...`).
- A capture bundle of the object: rectified images with their pinhole intrinsics and a known
  camera pose per image, the object's CAD mesh (meters) and its keypoint layout
  (`doper prepare --help` lists the bundle files under `--bundle`).
- A recipe for your capture: copy [examples/nurec/recipe.yaml](../examples/nurec/recipe.yaml) and
  set its object (mesh and keypoint-layout hashes) and capture views. `RECIPE` below is its path.
- Blender 4.2 or newer for the synthetic baseline, and the `nurec-sam` extra for SAM2 masks. The
  synthetic baseline renders the object in one dark material (`BODY_MATERIAL` in
  `rendering/nurec_assets.py`: base color 0.02, roughness 0.45); edit it to match your object.

## Stages

Every command takes `--dry-run`, which prints the ordered steps with their inputs and outputs and
touches no data: run it first. NuRec training itself runs outside DOPER, on the reconstruction
that `--stage reconstruct` writes; its artifact is `SCENE/last.usdz` below.

```bash
doper prepare RECIPE --root DATA --stage capture --bundle BUNDLE
doper prepare RECIPE --root DATA --stage reconstruct
#   ... train NuRec on DATA/reconstruction -> SCENE/last.usdz
doper prepare RECIPE --root DATA --stage synthetic --hdri-dir H --texture-dir T \
    --distractor-dir D
doper train RECIPE --root DATA --runs RUNS --stage base
doper evaluate RECIPE --root DATA --runs RUNS --bundle BUNDLE \
    --checkpoint RUNS/base/work/best.pth --out RUNS/eval/base
doper prepare RECIPE --root DATA --stage register --artifact SCENE/last.usdz \
    --keypoint-frames RUNS/eval/base/frames.json
doper prepare RECIPE --root DATA --stage register --accept-registration \
    --reviewer-note "overlays and held-out edge distances checked"
doper prepare RECIPE --root DATA --stage render --artifact SCENE/last.usdz
doper prepare RECIPE --root DATA --stage labels --artifact SCENE/last.usdz \
    --sam2-weights sam2.1_hiera_small.pt
doper train RECIPE --root DATA --runs RUNS --stage adapt
doper evaluate RECIPE --root DATA --runs RUNS --bundle BUNDLE \
    --checkpoint RUNS/adapt/work/best.pth --out RUNS/eval/adapt
doper report RECIPE --runs RUNS --baseline RUNS/eval/base --adapted RUNS/eval/adapt
```
