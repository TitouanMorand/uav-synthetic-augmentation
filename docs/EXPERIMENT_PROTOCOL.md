# Experiment Protocol

## Goal

Evaluate whether augmentation improves tiny drone detection robustness without degrading real-test performance.

The project separates two evaluation dimensions:

1. **Real-data performance**
   - Evaluate on the unchanged validation and test splits.
   - This is the main metric.

2. **Synthetic stress-test robustness**
   - Evaluate on controlled transformations of validation and test images.
   - These stress sets are never used for training.

## Training experiments

The core experiments are:

| ID | Name | Train data |
|---|---|---|
| E0 | baseline_real_only_hf_drone_300 | 300 real images |
| E1 | real_plus_classic_hf_drone_300 | 300 real + 600 classic augmented |
| E2 | real_plus_object_preserving_v1_hf_drone_300 | 300 real + 600 object-preserving synthetic |
| E3 | real_plus_diffusion_reinsert_night_hf_drone_300 | 300 real + diffusion images passing quality gates |

## Evaluation datasets

Each trained model is evaluated on:

| Dataset | Description |
|---|---|
| real | Original validation/test split |
| stress_night | Dark low-light night-like transformation |
| stress_haze | Haze / reduced visibility transformation |
| stress_low_contrast | Low-contrast degradation |

## Main metrics

Standard detection metrics:

- precision
- recall
- mAP50
- mAP50-95

Robustness metrics:

- metric drop from real test to stress test
- recall under stress
- mAP50-95 under stress

Small-object metrics:

- AP50 by object-size bucket
- recall by object-size bucket
- false positives per image

## Object-size buckets

The project uses custom buckets at 640-pixel evaluation scale:

| Bucket | Definition |
|---|---|
| very_tiny | max box side < 16 px |
| tiny | 16 px <= max box side < 32 px |
| small | 32 px <= max box side < 64 px |
| medium_plus | max box side >= 64 px |

## Equal-compute principle

The cleanest comparison controls for training exposure:

- baseline: 300 images x 15 epochs = 4500 image exposures
- augmented datasets: 900 images x 5 epochs = 4500 image exposures

A second full-budget comparison may be added later:

- baseline: 15 epochs
- augmented: 15 epochs

The equal-compute comparison is preferred for scientific interpretation.
