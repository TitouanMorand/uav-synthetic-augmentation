# UAV Synthetic Augmentation

Object-preserving synthetic data augmentation pipeline for tiny drone detection in aerial imagery.

This project studies whether controlled augmentation can improve YOLO drone detection under challenging visual conditions, especially low-light and night-like settings, while preserving small critical objects and their bounding-box labels.

## Motivation

Generic image augmentation can easily break object detection datasets when the object is small. For drone detection, the main risk is to generate visually plausible images while corrupting the tiny drone, hallucinating extra drones, or creating object-context inconsistencies.

The project therefore focuses on a data-centric question:

> Can we transform the image context while preserving the drone object and keeping YOLO labels valid?

## Dataset

Active dataset:

- Hugging Face dataset: `pathikg/drone-detection-dataset`
- Source annotation format: COCO-style bounding boxes `[x, y, width, height]`
- Target format: YOLO normalized labels
- Class: `drone`
- Working split used in this project: `300 train / 80 val / 80 test`

No generated datasets, trained weights, or local run outputs are versioned in Git.

## What the pipeline compares

The project compares:

1. **Real-only baseline**
   - YOLO trained on the real HF drone subset.

2. **Classical augmentation**
   - Standard image-space augmentations with YOLO boxes transformed accordingly.

3. **Object-preserving augmentation**
   - Context changes while preserving the drone label and object region.

4. **Diffusion-based night augmentation**
   - Pretrained diffusion only, no diffusion model training.
   - Source filtering to remove contaminated images.
   - Controlled night context generation.
   - Object matte extraction.
   - Local LAB delta transfer on the drone.
   - Hard reinsertion to preserve object geometry and labels.

## Repository structure

```text
configs/      Experiment and diffusion configuration files
src/          Reusable Python modules
scripts/      Step-by-step executable pipeline scripts
docs/         Project report and selected visual examples
data/         Local datasets, ignored by Git except .gitkeep files
artifacts/    Local generated reports, tables and previews, ignored by Git except .gitkeep files
runs/         YOLO training outputs, ignored by Git except .gitkeep
