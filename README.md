# UAV Synthetic Augmentation (MVP)

Goal
----
Controlled data augmentation pipeline for UAV/drone object detection. Use the Hugging Face dataset `pathikg/drone-detection-dataset` (RGB images + COCO-style bboxes) and compare a YOLO baseline trained on real images vs. real + augmented images.

Design constraints
------------------
- Do not train diffusion from scratch. Diffusion generation uses pretrained models.
- Use Hugging Face `datasets` to load data.
- Use Ultralytics `YOLO` package for detector training.
- Start with small subsets by default (500 train, 100 val) for fast iteration.

Project structure
-----------------
- `src/` - Python modules (export, visualize, augment, train, val)
- `scripts/` - shell runners for quick experiments
- `requirements.txt` - Python deps

Quick steps
-----------
1. Export a small subset to YOLO format:

```bash
bash scripts/run_export.sh --output data/yolo_subset --train-size 500 --val-size 100
```

2. Sanity-check boxes:

```bash
bash scripts/run_visualize.sh --yolo-dir data/yolo_subset --num 5
```

3. Create night-augmented copies (labels preserved):

```bash
PY=.venv/bin/python bash scripts/04_generate_night_aug.sh
```

4. Train baseline YOLO (quick smoke):

```bash
PY=.venv/bin/python bash scripts/03_train_baseline.sh
```

5. Validate model:

```bash
PY=.venv/bin/python -m src.detection.val_yolo \
  --weights runs/baseline/yolov8n_smoke/weights/best.pt
```

Baseline smoke test
-------------------
Milestone 2 adds a lightweight YOLO baseline using Ultralytics, with `yolov8n.pt`,
`data/yolo/dataset.yaml`, `imgsz=640`, `epochs=5`, and `seed=42` by default.
Runs are saved under `runs/baseline/`.

This baseline is only a smoke test: it checks that the dataset, labels, training
loop, and validation loop work end to end. Final experiments should use a larger
exported subset, more epochs, and a deliberate train/validation protocol before
comparing real-only data against augmented data.

Fast debug training
-------------------
CPU training can be slow. For quick code/data checks, use the fast debug script:

```bash
PY=.venv/bin/python bash scripts/03_train_baseline_fast.sh
```

This defaults to `epochs=1`, `imgsz=320`, `fraction=0.1`, `batch=4`, and writes to
`runs/baseline/yolov8n_fast_debug/`. Use it only to check that the pipeline still
runs; its metrics are not meaningful for model comparison.

If your machine exposes Apple MPS or CUDA through PyTorch, override the device:

```bash
DEVICE=mps PY=.venv/bin/python bash scripts/03_train_baseline_fast.sh
DEVICE=0 PY=.venv/bin/python bash scripts/03_train_baseline_fast.sh
```

On Apple Silicon, use the dedicated Python 3.12 environment if the default
environment does not expose MPS:

```bash
PY=.venv-mps/bin/python bash scripts/03_train_baseline.sh --device mps
```

Classical night augmentation
----------------------------
Milestone 3 creates a second YOLO dataset at `data/yolo_aug_night/`:

- `images/train` contains both original real training images and night-augmented copies.
- `labels/train` contains copied YOLO labels for both versions.
- `images/val` and `labels/val` remain real-only and unchanged.
- `dataset.yaml` points YOLO to the augmented training set and real validation set.
- `previews/` contains contact sheets comparing original vs. night images with boxes.

Generate it with:

```bash
PY=.venv/bin/python bash scripts/04_generate_night_aug.sh
```

Then train on the augmented dataset with:

```bash
PY=.venv/bin/python bash scripts/05_train_augmented.sh
```

The script trains two runs with the same model, image size, epochs, seed, batch
setting, workers, and validation images:

- baseline: `data/yolo/dataset.yaml`
- augmented: `data/yolo_aug_night/dataset.yaml`

Balanced augmentation ablation
------------------------------
The full augmented dataset has twice as many training images as the baseline:
500 real images + 500 night copies. That is useful, but it does not isolate the
effect of the augmentation itself because the model also sees more examples.

For a fairer ablation with the same number of training samples as the baseline,
create a balanced dataset with 250 real images and their 250 night copies:

```bash
PY=.venv-mps/bin/python bash scripts/05_prepare_balanced_aug.sh
```

This writes `data/yolo_aug_night_balanced/dataset.yaml`. Its train split has 500
images total, and its validation split still points to the real-only validation
images from `data/yolo/`.

Train the balanced augmented run with the same setup as the 20-epoch baseline:

```bash
PY=.venv-mps/bin/python DEVICE=mps bash scripts/05_train_augmented_balanced.sh
```

Observed 20-epoch result on the current 500/100 split:

- baseline best mAP50/mAP50-95: `0.7555 / 0.2981`
- balanced real+night best mAP50/mAP50-95: `0.6961 / 0.2721`

Interpretation: with the same number of training samples, classical night
augmentation does not improve the detector on the real-only validation set and
slightly reduces mAP. The stronger result from the full real+night run is likely
explained mainly by the larger number of training examples, not by the night
augmentation alone.

Readable metrics
----------------
Ultralytics writes dense CSV logs to `results.csv`. Training now also writes a
human-readable `metrics_summary.md` in the run folder. To summarize an existing
run manually:

```bash
PY=.venv/bin/python bash scripts/06_summarize_results.sh runs/baseline/yolov8n_smoke/results.csv
```

Diffusion augmentation branch
-----------------------------
The diffusion branch is an experimental synthetic augmentation path for
night/low-light variants. It uses pretrained Diffusers pipelines only; it does
not train a diffusion model and does not alter YOLO labels.

The core preservation rule is: generated images must keep the same pixel
dimensions as the source image, and copied YOLO labels remain unchanged. For
object-sensitive generation, masks are built from YOLO boxes so the drone region
can be protected or reinserted after background editing.

Configuration lives in:

```bash
configs/diffusion.yaml
```

Generation modes:

- `global_img2img`: naive whole-image img2img baseline. It is useful for
  qualitative comparison, but it is unsafe for YOLO training unless it passes
  preservation filters because diffusion can alter small critical objects,
  including the drone, controller, or FPV screen.
- `background_inpaint_protected_box`: inpaint background while preserving an
  expanded drone protection region. Diffusers inpainting uses white pixels for
  repainting and black pixels for preservation, so the drone region is black in
  the inpaint mask.
- `background_inpaint_reinsert_object`: inpaint the editable background first,
  then paste the original protected drone pixels back before labels are copied.

Small diffusion smoke test:

```bash
PY=.venv-mps/bin/python DEVICE=mps bash scripts/06_generate_diffusion_grid.sh
```

This defaults to 10 source images, `night_lowlight`, all three modes,
strengths `0.2,0.35`, guidance values `5.0,7.5`, and seed `42`. Outputs are
written under `data/synthetic/diffusion_grid/` with a JSON metadata file per
sample and a global manifest:

```bash
data/synthetic/diffusion_grid/manifest.jsonl
```

Preview generated samples:

```bash
PY=.venv-mps/bin/python bash scripts/07_preview_diffusion_grid.sh
```

The preview contact sheet is saved under `data/previews/diffusion/` and shows:
original, mask overlay, generated image, and generated image with YOLO boxes.

Object-preserving debug smoke test:

```bash
PY=.venv/bin/python DEVICE=mps bash scripts/12_debug_object_preserving_generation.sh
```

This writes generation outputs to `data/synthetic/debug_object_preserving/` and
a stricter contact sheet to `data/previews/debug_object_preserving/`. The sheet
shows original, original with box, protection mask, inpaint mask, generated
before reinsertion, generated after reinsertion, and generated after reinsertion
with the YOLO box.

Diffusion samples are accepted for training only if automatic preservation
checks pass. The metadata records crop SSIM, object mean absolute difference,
background mean absolute difference outside the protected region, black-image
checks, and rejection reasons. Rejected samples remain useful for debugging but
should not be added to YOLO training data.

Ablation plan
--------------
- Baseline: train on `real` subset.
- Augmented: train on `real + night_augmented` (classical augmentation first).
- Diffusion: compare real-only vs. real + accepted diffusion samples, keeping
  validation real-only and unchanged.

TODO (future work)
------------------
- Evaluate diffusion samples across multiple seeds and train/validation splits.
- Add object-aware quality checks for duplicated/missing/deformed drones.

Notes for interview
-------------------
- Code is modular and each script accepts CLI args for reproducibility.
- Conversion logic uses COCO [x,y,w,h] -> YOLO normalized format `[class, x_c, y_c, w, h]`.
- Visual checks ensure coordinate transforms are correct.

License
-------
Academic/educational use.
