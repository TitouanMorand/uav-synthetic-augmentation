# UAV Synthetic Augmentation

Clean Python-first project for object-preserving synthetic data augmentation applied to tiny drone detection in aerial imagery.

## Goal

The goal is to build a simple and reproducible pipeline to evaluate whether data augmentation improves tiny drone detection.

The project compares:

1. a YOLO baseline trained on real drone images only;
2. YOLO trained with classic augmentations;
3. YOLO trained with object-preserving synthetic augmentations.

## Dataset

Active dataset:

- Hugging Face dataset: `pathikg/drone-detection-dataset`
- Source annotation style: COCO-style bounding boxes `[x, y, width, height]`
- Target annotation style: YOLO
- Class: `drone`

## Pipeline

Run the project step by step:

```bash
python scripts/00_check_env.py
python scripts/01_prepare_dataset.py
python scripts/02_preview_dataset.py
python scripts/03_train_baseline.py
python scripts/04_make_augmentations.py
python scripts/05_train_augmented.py
python scripts/06_compare_experiments.py
python scripts/07_error_analysis.py
python scripts/08_make_report.py
