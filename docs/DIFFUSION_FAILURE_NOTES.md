# Diffusion Failure Notes

## Context

An initial pool of night-oriented diffusion augmentations was generated using an inpaint-and-reinsert strategy.

The intended goal was:

- transform the context toward night / low-light conditions;
- preserve the drone object region;
- keep YOLO labels unchanged;
- use the generated images as additional training data.

## What was done

The current diffusion batch used:

- source images from the cleaned Hugging Face drone dataset;
- an inpainting mask derived from YOLO bounding boxes;
- object reinsertion after diffusion;
- quality metadata including:
  - background-region mean absolute difference;
  - object-region mean absolute difference;
  - mask coverage;
  - suspicious black image detection.

The batch was stopped after approximately 165 accepted generated images.

A diagnostic evaluation was then prepared using ablation datasets:

- real + 75 diffusion images;
- real + 150 diffusion images;
- real + 165 diffusion images.

## Observed failure modes

Visual inspection showed several significant issues.

### 1. Source contamination

Some source images already contain picture-in-picture views, HUD overlays, screens, or embedded drone-camera views.

Diffusion tends to amplify these artifacts and may generate split-screen or double-image compositions.

### 2. Scene drift

The generated background often changes to an unrelated scene, such as:

- city-at-night satellite views;
- unrealistic aerial maps;
- star fields;
- surreal landscapes;
- night scenes unrelated to the original geometry.

This violates the goal of preserving scene layout while changing only context.

### 3. Object-context discontinuity

Even when the drone is preserved, it often appears pasted onto a newly generated background.

This creates visible halos, rectangular seams, or lighting inconsistencies.

### 4. Over-aggressive diffusion

The inpainting settings were too strong for reliable dataset generation.

High strength and guidance made the model overrule the original scene rather than perform a controlled context transformation.

### 5. Object hallucination

Some generated backgrounds contain additional drone-like artifacts or unrelated flying objects.

This creates label noise because the YOLO label file still contains only the original drone annotations.

## Interpretation

This batch should be treated as a diagnostic failed diffusion batch, not as a validated synthetic training dataset.

It is useful because it reveals the real failure modes of naive diffusion-based augmentation:

- prompt drift;
- insufficient structural control;
- poor source selection;
- hard object reinsertion artifacts;
- insufficient quality gates.

## Next improvements

The next iteration should add:

1. source filtering before diffusion;
2. stricter prompt design;
3. softer masks and feathered object blending;
4. photometric adaptation of the reinserted object;
5. ControlNet Canny or Depth to preserve scene geometry;
6. hyperparameter grid search on a small validation subset before large-scale generation;
7. YOLO-based rejection of images with zero or multiple detected drones;
8. manual review for the first generated batch.

## Status

The generated diffusion images are kept only for diagnostic evaluation.

They should not be presented as final high-quality synthetic training data.
