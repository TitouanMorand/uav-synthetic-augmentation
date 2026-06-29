# Project History — UAV Synthetic Augmentation

This document summarizes the clean rebuild of the `uav-synthetic-augmentation` project on the `hf-clean-rebuild` branch.

The goal of the project is to build and evaluate an object-preserving synthetic data augmentation pipeline for tiny drone detection in aerial imagery, with a focus on robustness to night and degraded visual conditions.

---

## 1. Project restart and cleanup

The project was restarted from a cleaner structure because the previous implementation had become too complex to explain and justify clearly.

The clean rebuild follows these principles:

- use the Hugging Face drone dataset only;
- remove all traces of the earlier toy/fake generated dataset;
- keep a simple Python-first structure;
- avoid overly complex orchestration;
- make each script understandable and defensible;
- track source code and documentation in Git;
- keep datasets, YOLO runs, model weights and generated artifacts out of Git.

The active branch for the clean rebuild is `hf-clean-rebuild`.

---

## 2. Repository structure

The project uses a simple structure:

    uav-synthetic-augmentation/
    ├── configs/
    ├── src/
    ├── scripts/
    ├── docs/
    ├── data/
    ├── artifacts/
    └── runs/

### configs/

Contains configuration files for the project and for diffusion generation.

Important files:

- `configs/project.yaml`
- `configs/diffusion_v2.yaml`
- `configs/diffusion_source_blacklist.txt`

### src/

Contains reusable Python modules.

Main responsibilities:

- dataset utilities;
- augmentation utilities;
- diffusion helper functions;
- source filtering heuristics;
- object matte generation;
- LAB photometric transfer;
- evaluation helpers.

### scripts/

Contains executable step-by-step scripts.

The project is intentionally script-based so that each stage can be run, inspected and explained independently.

### data/

Local datasets are stored here but should not be versioned.

Important generated datasets include:

- `data/interim/yolo_drone_hf_300/`
- `data/augmented/`

### artifacts/

Generated reports, tables and previews are stored here locally.

Typical outputs:

- `artifacts/previews/`
- `artifacts/reports/`
- `artifacts/tables/`

### runs/

YOLO training outputs are stored here locally.

This includes training logs, validation plots, model weights and prediction previews.

---

## 3. Dataset choice

The active dataset is:

- Hugging Face dataset: `pathikg/drone-detection-dataset`

This is a drone detection dataset.

The source annotations are COCO-style bounding boxes:

    [x, y, width, height]

They are converted into YOLO normalized labels:

    class_id x_center y_center width height

The project uses a working split of:

    300 train / 80 val / 80 test

The converted dataset is written locally to:

    data/interim/yolo_drone_hf_300/

The dataset is not committed to Git.

---

## 4. Environment and sanity checks

The first step of the rebuild was to check the environment and make sure the project could run on a Mac laptop.

The environment check verifies:

- Python environment;
- PyTorch installation;
- MPS / CUDA / CPU device availability;
- Ultralytics YOLO;
- Hugging Face datasets;
- OpenCV;
- Diffusers;
- general import consistency.

This step was important because the project combines standard computer vision, YOLO training and diffusion inference.

---

## 5. Dataset preparation

The dataset preparation script loads the Hugging Face dataset and writes a YOLO-format subset.

The preparation step performs:

1. streaming load from Hugging Face;
2. extraction of images and bounding boxes;
3. conversion from COCO boxes to YOLO boxes;
4. creation of train / val / test folders;
5. generation of a `dataset.yaml` file;
6. creation of preparation reports.

The output is:

    data/interim/yolo_drone_hf_300/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml

This step was successfully run with:

    train = 300
    val   = 80
    test  = 80

---

## 6. Dataset validation and visualization

After conversion, the dataset was visually inspected.

The goals were:

- confirm that bounding boxes align with drones;
- confirm that labels are valid YOLO labels;
- inspect the distribution of object sizes;
- detect unusual source images;
- verify that the dataset is suitable for tiny-object detection experiments.

This step revealed that the dataset contains some problematic images, including:

- picture-in-picture drone views;
- controller overlays;
- HUD-like displays;
- text and watermarks;
- drone-camera screen inserts;
- images where the drone is very large or held close to the camera.

These images can still be part of the real training set, but they are dangerous as sources for diffusion augmentation.

---

## 7. Real-only YOLO baseline

A YOLO baseline was trained on the real Hugging Face drone subset.

The baseline is important because all augmentation methods must be compared against it.

The 15-epoch baseline reached approximately:

- precision: 0.902
- recall: 0.911
- mAP50: 0.916
- mAP50-95: 0.608

Interpretation:

- the converted dataset is valid;
- YOLO learns the drone class correctly;
- the baseline is already strong;
- any augmentation must be evaluated carefully because bad synthetic data can degrade performance.

This is a key lesson of the project: augmentation should not be assumed beneficial just because it increases dataset size.

---

## 8. Classical augmentation baseline

A classical augmentation baseline was added before diffusion experiments.

The goal was to create a controlled non-generative comparison.

Classical augmentations are useful because:

- they are deterministic or mostly controlled;
- bounding boxes can be transformed consistently;
- they provide a lower-risk augmentation baseline;
- they help separate the value of augmentation in general from the value of diffusion specifically.

The classical augmentation pipeline is used as one of the main experimental baselines.

---

## 9. First diffusion experiments

The initial diffusion experiments explored several strategies:

1. global image-to-image generation;
2. inpainting with protected object regions;
3. inpainting followed by object reinsertion.

The intended goal was:

    change the visual context, especially toward night,
    while preserving the drone and keeping YOLO labels valid.

The first results exposed major failure modes.

Observed issues included:

- scene drift toward unrelated city or satellite night scenes;
- unrealistic star fields or surreal landscapes;
- picture-in-picture amplification;
- hard discontinuity between generated context and original drone;
- rectangular halos around the drone;
- duplicated drone-like artifacts;
- hallucinated flying objects;
- strong object-context mismatch.

Conclusion:

    Naive diffusion augmentation is not reliable enough to be used directly for training.

This failure analysis became an important part of the project because it motivated stricter controls and quality gates.

---

## 10. Diagnostic evaluation of failed diffusion

Instead of discarding the failed diffusion batch immediately, it was kept as a diagnostic experiment.

The goal was to test the evaluation pipeline and measure how poor diffusion data could affect model performance.

This is useful because it demonstrates a rigorous data-centric mindset:

- do not assume synthetic data is useful;
- inspect generated samples;
- evaluate whether they help or hurt;
- document failure modes clearly.

A failure note was created to explain why this first diffusion batch should not be considered final training data.

---

## 11. Evaluation protocol

A full evaluation protocol was designed to compare:

1. real-only baseline;
2. real + classical augmentation;
3. real + object-preserving augmentation;
4. real + diffusion augmentation;
5. diffusion ablations with different numbers of generated images.

The evaluation is designed to measure not only standard performance but also robustness.

Main metrics:

- precision;
- recall;
- mAP50;
- mAP50-95;
- AP by object size;
- false positives per image.

Stress-test conditions:

- real test;
- night stress;
- haze stress;
- low-contrast stress.

This protocol is especially relevant for operational perception systems because they must remain robust under visual domain shifts.

---

## 12. Diffusion source filtering

A major improvement was to add filtering before diffusion generation.

The reason is that many source images are not good diffusion inputs.

Problematic source images include:

- picture-in-picture views;
- controller screens;
- FPV overlays;
- HUD elements;
- watermarks;
- strong internal rectangles;
- vertical seams;
- drones that are too large;
- sources with confusing embedded views.

The filtering script computes several heuristic scores:

- mask coverage;
- internal rectangle score;
- UI rectangle score;
- inset window score;
- vertical seam score;
- box size statistics.

A manual blacklist was also added:

    configs/diffusion_source_blacklist.txt

This makes the source filtering process both automatic and manually controllable.

The filtering was first too strict, keeping only around 24 sources. It was then relaxed to allow a larger candidate pool, with manual review used for final cleanup.

This reflects a practical data-centric workflow:

    automatic filtering removes obvious bad cases;
    manual inspection removes ambiguous remaining cases.

---

## 13. Diffusion V2: controlled night generation

The diffusion pipeline was redesigned to be more conservative.

Instead of asking the model to create an entirely new night scene, the pipeline now uses:

    source image
    → controlled night condition
    → conservative img2img
    → generated night context
    → object matte
    → local photometric correction
    → hard reinsertion

This reduces scene drift and keeps the generated image closer to the original geometry.

The selected generation mode is:

    preset = conservative
    mode   = hard_lab_delta

---

## 14. Object matte

To avoid preserving a large rectangular patch around the drone, an object matte is estimated from the YOLO bounding box.

The goal is to approximate the drone silhouette more closely than a raw box.

This step is important because bounding-box masks caused visible rectangular halos in earlier diffusion experiments.

The object matte is used to decide which pixels belong to the drone during reinsertion.

---

## 15. LAB delta transfer

A key improvement was the introduction of local LAB delta transfer.

The problem was that even when the drone was preserved, it often kept the color and lighting of the original daytime image.

This created a visible mismatch when pasted into a generated night context.

The solution is:

1. compute LAB statistics in a ring around the object in the original image;
2. compute LAB statistics in the same ring in the generated context;
3. estimate the local photometric delta;
4. apply this delta to the drone pixels only.

The transformation includes:

- luminance shift;
- mild contrast scaling;
- small A/B color shifts;
- slight blue bias for night consistency.

The corrected drone is then reinserted with a hard mask.

This preserves object geometry while reducing photometric mismatch.

---

## 16. Why hard LAB reinsertion was selected

Several modes were compared:

- hard_original;
- hard_lab_delta;
- microfeather_lab_delta.

### hard_original

The original drone is pasted directly into the generated context.

Advantage:

- perfect preservation of object pixels.

Issue:

- the drone may retain daylight colors and look pasted.

### hard_lab_delta

The drone is first photometrically adapted using local LAB delta transfer, then pasted with a hard mask.

Advantages:

- preserves object shape;
- keeps the YOLO label valid;
- reduces daylight/night mismatch;
- avoids blurring tiny drone edges.

This is the selected main mode.

### microfeather_lab_delta

The LAB-corrected drone is pasted with a very small feather.

Advantage:

- can soften boundaries.

Issue:

- for tiny drones, even slight feathering may blur useful object edges.

For this reason, it is kept for inspection but not selected as the main production mode.

---

## 17. Diffusion V2 inspection grids

Readable full and zoom inspection grids were added.

The full grid shows:

- original image;
- original with box;
- object matte;
- conservative condition;
- paste mask;
- LAB-corrected object;
- hard original;
- hard LAB delta;
- microfeather LAB delta;
- medium variants.

The zoom grid focuses on the drone region, which is necessary because the drone is often only a few pixels wide.

This made it possible to visually compare whether LAB delta transfer actually improved object-context consistency.

Selected visual examples can be stored under:

- `docs/images/source_filter_accepted.jpg`
- `docs/images/source_filter_rejected.jpg`
- `docs/images/diffusion_v2_full_grid.jpg`
- `docs/images/diffusion_v2_zoom_grid.jpg`

---

## 18. Current recommended diffusion production setting

The current recommended setting for pool generation is:

    preset = conservative
    mode   = hard_lab_delta

Rationale:

- conservative generation better preserves scene geometry;
- hard reinsertion keeps tiny drones sharp;
- LAB delta transfer reduces the color mismatch;
- labels remain valid because the drone is not regenerated.

The first recommended pool size is:

    N = 100 generated diffusion images

Ablation datasets should then be built as:

- real + 25 diffusion images;
- real + 50 diffusion images;
- real + 100 diffusion images.

This allows evaluation of whether increasing diffusion data helps or hurts.

---

## 19. Planned final evaluation

The planned final evaluation compares:

- E0 — real-only baseline;
- E1 — real + classical augmentation;
- E2 — real + object-preserving augmentation;
- E3 — real + diffusion V2 LAB delta N=25;
- E4 — real + diffusion V2 LAB delta N=50;
- E5 — real + diffusion V2 LAB delta N=100.

Optional:

- E6 — real + diffusion V2 medium LAB delta N=100.

Evaluation metrics:

- precision;
- recall;
- mAP50;
- mAP50-95;
- AP by object size;
- false positives per image.

Evaluation datasets:

- real test;
- stress night;
- stress haze;
- stress low contrast.

The key question is not only whether mAP improves, but whether diffusion improves robustness to night-like conditions without degrading the real test set.

---

## 20. Key lessons

This project demonstrates several important lessons.

### Synthetic data is not automatically useful

The first diffusion results looked superficially interesting but introduced severe label and distribution problems.

### Visual quality is not enough

A generated image can look plausible and still be harmful for object detection if the object is corrupted or if extra objects are hallucinated.

### Object preservation matters

For tiny drone detection, preserving object geometry is more important than generating visually impressive images.

### Source filtering matters

Bad source images cause bad diffusion outputs.

Filtering out HUDs, overlays, picture-in-picture views and large drones is necessary before generation.

### Local photometric correction is useful

LAB delta transfer helps reduce the mismatch between the preserved drone and the generated context.

### Evaluation must include robustness

A useful augmentation should be evaluated on:

- real test performance;
- stress conditions;
- small-object AP;
- false positives.

---

## 21. Current status

The project currently contains:

- a clean HF dataset preparation pipeline;
- a YOLO baseline;
- classical augmentation;
- object-preserving augmentation;
- diffusion failure analysis;
- source filtering for diffusion;
- diffusion V2 inspection grids;
- local LAB delta transfer;
- selected production mode for diffusion V2.

The next step is to generate the diffusion V2 pool using:

    preset = conservative
    mode   = hard_lab_delta

then build ablation datasets and run the full evaluation protocol.

---

## 22. Limitations

Current limitations:

- diffusion quality is still not guaranteed;
- source filtering is heuristic and partly manual;
- object matte is approximate;
- ControlNet is not yet used;
- generated data must be evaluated before being considered useful;
- the dataset subset is small;
- the evaluation is still ongoing.

These limitations are documented intentionally because they reflect the real constraints of using generative augmentation for object detection.

---

## 23. Next steps

Immediate next steps:

1. generate the conservative hard-LAB-delta diffusion pool;
2. build real + 25, real + 50, real + 100 datasets;
3. train YOLO on each dataset under comparable compute;
4. run standard and stress-test evaluation;
5. summarize results in tables and plots;
6. decide whether diffusion augmentation helps, hurts, or only helps specific stress conditions.

---

## 24. Summary

This project evolved from a simple drone detection augmentation pipeline into a rigorous object-preserving synthetic data evaluation framework.

The most important contribution is not just generating night images, but building a controlled process around synthetic data:

- filter sources;
- generate conservatively;
- preserve object geometry;
- adapt object photometry;
- inspect visually;
- evaluate quantitatively;
- document failure modes.

This is the core engineering and research logic of the project.
