# UAV Synthetic Augmentation (MVP)

Goal
----
Controlled data augmentation pipeline for UAV/drone object detection. Use the Hugging Face dataset `pathikg/drone-detection-dataset` (RGB images + COCO-style bboxes) and compare a YOLO baseline trained on real images vs. real + augmented images.

Design constraints
------------------
- Do not train diffusion from scratch. Diffusion-based ideas are left as TODOs.
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
PY=.venv/bin/python bash scripts/03_train_baseline.sh \
  --data data/yolo_aug_night/dataset.yaml \
  --name yolov8n_night_smoke
```

Readable metrics
----------------
Ultralytics writes dense CSV logs to `results.csv`. Training now also writes a
human-readable `metrics_summary.md` in the run folder. To summarize an existing
run manually:

```bash
PY=.venv/bin/python bash scripts/05_summarize_results.sh runs/baseline/yolov8n_smoke/results.csv
```

Ablation plan
--------------
- Baseline: train on `real` subset.
- Augmented: train on `real + night_augmented` (classical augmentation first).
- Later: add diffusion-based night/weather augmentation (img2img/inpainting) and evaluate.

TODO (future work)
------------------
- Diffusion img2img/night — use pretrained diffusion models (no training from scratch).
- Object-protected masks during augmentation (keep drone pixels intact when applying global weather changes).
- Object extraction and reinsertion (copy-paste augmentation / compositing with realistic lighting).

Notes for interview
-------------------
- Code is modular and each script accepts CLI args for reproducibility.
- Conversion logic uses COCO [x,y,w,h] -> YOLO normalized format `[class, x_c, y_c, w, h]`.
- Visual checks ensure coordinate transforms are correct.

License
-------
Academic/educational use.
