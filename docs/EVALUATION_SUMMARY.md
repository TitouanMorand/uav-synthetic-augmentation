# Evaluation Summary — test


This report summarizes the augmentation evaluation protocol. It compares the real-only baseline against classic augmentation, object-preserving augmentation, and diffusion ablation runs when available.


## 1. Real test metrics


| experiment | run_name | precision | recall | map50 | map50_95 |
| --- | --- | --- | --- | --- | --- |
| baseline_real_only | baseline_real_only_hf_drone_300 | 0.9157 | 0.7952 | 0.891 | 0.5806 |
| real_plus_classic | real_plus_classic_hf_drone_300 | 0.9265 | 0.9277 | 0.9518 | 0.6066 |
| real_plus_object_preserving | real_plus_object_preserving_v1_hf_drone_300 | 0.8897 | 0.8675 | 0.8839 | 0.5716 |
| real_plus_diffusion_n075 | real_plus_diffusion_reinsert_night_n075_hf_drone_300 | 0.9095 | 0.8313 | 0.9263 | 0.5938 |
| real_plus_diffusion_n150 | real_plus_diffusion_reinsert_night_n150_hf_drone_300 | 0.9316 | 0.7711 | 0.8901 | 0.5473 |
| real_plus_diffusion_n165 | real_plus_diffusion_reinsert_night_n165_hf_drone_300 | 0.8234 | 0.8434 | 0.8636 | 0.5479 |


## 2. Stress-test mAP50-95


| experiment | real | stress_night | stress_haze | stress_low_contrast |
| --- | --- | --- | --- | --- |
| baseline_real_only | 0.5806 | 0.3882 | 0.5251 | 0.5503 |
| real_plus_classic | 0.6066 | 0.5272 | 0.5753 | 0.5941 |
| real_plus_object_preserving | 0.5716 | 0.416 | 0.5261 | 0.5469 |
| real_plus_diffusion_n075 | 0.5938 | 0.4405 | 0.5406 | 0.5782 |
| real_plus_diffusion_n150 | 0.5473 | 0.4419 | 0.4949 | 0.5117 |
| real_plus_diffusion_n165 | 0.5479 | 0.3445 | 0.5143 | 0.5433 |


## 3. Stress-test recall


| experiment | real | stress_night | stress_haze | stress_low_contrast |
| --- | --- | --- | --- | --- |
| baseline_real_only | 0.7952 | 0.6506 | 0.7229 | 0.759 |
| real_plus_classic | 0.9277 | 0.7831 | 0.8675 | 0.8735 |
| real_plus_object_preserving | 0.8675 | 0.6265 | 0.759 | 0.7909 |
| real_plus_diffusion_n075 | 0.8313 | 0.6265 | 0.8155 | 0.7952 |
| real_plus_diffusion_n150 | 0.7711 | 0.6145 | 0.7711 | 0.6947 |
| real_plus_diffusion_n165 | 0.8434 | 0.6043 | 0.7862 | 0.7952 |


## 4. Deltas vs baseline


| experiment | dataset | precision_delta | recall_delta | map50_delta | map50_95_delta |
| --- | --- | --- | --- | --- | --- |
| real_plus_classic | real | 0.0108 | 0.1325 | 0.0608 | 0.026 |
| real_plus_classic | stress_night | 0.1555 | 0.1325 | 0.1481 | 0.139 |
| real_plus_classic | stress_haze | 0.0694 | 0.1446 | 0.0949 | 0.0502 |
| real_plus_classic | stress_low_contrast | 0.0189 | 0.1145 | 0.0832 | 0.0438 |
| real_plus_object_preserving | stress_night | 0.1192 | -0.0241 | 0.0019 | 0.0277 |
| real_plus_object_preserving | real | -0.026 | 0.0723 | -0.0071 | -0.0091 |
| real_plus_object_preserving | stress_haze | 0.018 | 0.0361 | 0.006 | 0.001 |
| real_plus_object_preserving | stress_low_contrast | -0.0736 | 0.0318 | -0.02 | -0.0034 |
| real_plus_diffusion_n075 | stress_haze | -0.1441 | 0.0926 | -0.0039 | 0.0155 |
| real_plus_diffusion_n075 | stress_night | 0.107 | -0.0241 | 0.0227 | 0.0523 |
| real_plus_diffusion_n075 | real | -0.0062 | 0.0361 | 0.0352 | 0.0132 |
| real_plus_diffusion_n075 | stress_low_contrast | -0.045 | 0.0361 | 0.0345 | 0.0279 |
| real_plus_diffusion_n150 | real | 0.0159 | -0.0241 | -0.001 | -0.0333 |
| real_plus_diffusion_n150 | stress_haze | -0.0687 | 0.0482 | -0.006 | -0.0302 |
| real_plus_diffusion_n150 | stress_low_contrast | -0.0356 | -0.0643 | -0.0328 | -0.0387 |
| real_plus_diffusion_n150 | stress_night | 0.0665 | -0.0361 | 0.0409 | 0.0537 |
| real_plus_diffusion_n165 | real | -0.0923 | 0.0482 | -0.0274 | -0.0328 |
| real_plus_diffusion_n165 | stress_night | -0.0692 | -0.0463 | -0.1114 | -0.0437 |
| real_plus_diffusion_n165 | stress_haze | -0.0857 | 0.0633 | -0.0244 | -0.0109 |
| real_plus_diffusion_n165 | stress_low_contrast | -0.1038 | 0.0361 | -0.0041 | -0.007 |


## 5. AP50 by object size — real test


| experiment | very_tiny | tiny | small | medium_plus |
| --- | --- | --- | --- | --- |
| baseline_real_only | 0.0714 | 0.2932 | 0.3082 | 0.6804 |
| real_plus_classic | 0.1013 | 0.2931 | 0.3618 | 0.6567 |
| real_plus_object_preserving | 0.0883 | 0.2953 | 0.3222 | 0.6358 |
| real_plus_diffusion_n075 | 0.0804 | 0.2791 | 0.3605 | 0.6423 |
| real_plus_diffusion_n150 | 0.0791 | 0.3136 | 0.3067 | 0.4902 |
| real_plus_diffusion_n165 | 0.0769 | 0.2864 | 0.341 | 0.5305 |


## 6. Recall by object size — real test


| experiment | very_tiny | tiny | small | medium_plus |
| --- | --- | --- | --- | --- |
| baseline_real_only | 1.0 | 1.0 | 1.0 | 0.9688 |
| real_plus_classic | 1.0 | 1.0 | 1.0 | 0.9688 |
| real_plus_object_preserving | 1.0 | 1.0 | 1.0 | 0.9375 |
| real_plus_diffusion_n075 | 1.0 | 1.0 | 1.0 | 1.0 |
| real_plus_diffusion_n150 | 1.0 | 1.0 | 1.0 | 0.9688 |
| real_plus_diffusion_n165 | 1.0 | 1.0 | 1.0 | 0.9375 |


## 7. AP50 by object size — stress night


| experiment | very_tiny | tiny | small | medium_plus |
| --- | --- | --- | --- | --- |
| baseline_real_only | 0.077 | 0.1965 | 0.2872 | 0.4681 |
| real_plus_classic | 0.0953 | 0.2162 | 0.3747 | 0.5835 |
| real_plus_object_preserving | 0.0945 | 0.1376 | 0.2757 | 0.5104 |
| real_plus_diffusion_n075 | 0.0769 | 0.1977 | 0.3515 | 0.3905 |
| real_plus_diffusion_n150 | 0.0709 | 0.2031 | 0.3002 | 0.4615 |
| real_plus_diffusion_n165 | 0.0474 | 0.0794 | 0.2989 | 0.4529 |


## 8. False positives per image


| experiment | real | stress_night | stress_haze | stress_low_contrast |
| --- | --- | --- | --- | --- |
| baseline_real_only | 0.2 | 0.2625 | 0.125 | 0.0875 |
| real_plus_classic | 0.125 | 0.125 | 0.175 | 0.075 |
| real_plus_object_preserving | 0.15 | 0.1125 | 0.075 | 0.05 |
| real_plus_diffusion_n075 | 0.1625 | 0.0 | 0.125 | 0.05 |
| real_plus_diffusion_n150 | 0.1125 | 0.05 | 0.0375 | 0.05 |
| real_plus_diffusion_n165 | 0.1625 | 0.0375 | 0.0625 | 0.0625 |


## Generated files


- Standard metrics: `artifacts/tables/protocol_standard_metrics.csv`
- Deltas: `artifacts/tables/protocol_standard_metric_deltas.csv`
- Size metrics: `artifacts/tables/size_stratified_metrics.csv`
- FP/image metrics: `artifacts/tables/fp_per_image_metrics.csv`
- Summary tables: `artifacts/tables/summary_test_*.csv`
- Summary plots: `artifacts/previews/summary_test_*.png`
