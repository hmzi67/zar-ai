# Every trained model and its scores

202 trained models, from MLflow (finished, non-dry runs). Generated; the full table with run IDs is `experiments/all_models_scores.csv`. **Bold** = the model the chatbot serves.

Scores: *val macro-F1* picks the checkpoint. *Test macro-F1* is on the run's test set (see *test on*): for LOSO that is a whole source the model never saw, so it is much lower than pooled. *NWRD recall*: field stripe-rust tiles predicted yellow_rust. *Healthy FPR*: held-out healthy images predicted yellow_rust. *ECE*: calibration error after temperature scaling (lower is better). wheat_v1 scores are on 15 classes and are not comparable with wheat_v2 (5 classes).

## 1. Summary per configuration (mean ± SD over runs)

| group | model | setup | runs | val macro-F1 | test macro-F1 | NWRD recall | healthy FPR | ECE |
|---|---|---|---|---|---|---|---|---|
| v1 baseline | convnext_tiny | full | 4 | 0.858 ± 0.009 | 0.835 ± 0.010 | - | - | 0.038 ± 0.014 |
| v1 baseline | tf_efficientnetv2_s | full | 4 | 0.854 ± 0.004 | 0.847 ± 0.003 | - | - | 0.023 ± 0.004 |
| v1 baseline | vit_small_patch14_dinov2 | linear | 1 | 0.808 | 0.799 | - | - | 0.021 |
| v1 leave-source-out | tf_efficientnetv2_s | full | 4 | 0.836 ± 0.004 | 0.813 ± 0.003 | - | - | 0.024 ± 0.005 |
| v2 pooled | dinov2_vit_b14 | full / strong | 3 | 0.970 ± 0.008 | 0.947 ± 0.008 | 0.045 ± 0.028 | - | 0.018 ± 0.007 |
| v2 pooled | dinov2_vit_l14 | linear / basic | 3 | 0.941 ± 0.003 | 0.933 ± 0.001 | 0.122 ± 0.002 | - | 0.018 ± 0.001 |
| v2 pooled | efficientnetv2_s | full / strong | 3 | 0.947 ± 0.004 | 0.916 ± 0.008 | 0.040 ± 0.006 | - | 0.017 ± 0.004 |
| v2 LOSO | dinov2_vit_b14 | full / basic | 12 | 0.924 ± 0.078 | 0.457 ± 0.216 | 0.030 ± 0.037 | - | 0.427 ± 0.179 |
| v2 LOSO | dinov2_vit_b14 | full / strong | 12 | 0.927 ± 0.081 | 0.461 ± 0.232 | 0.065 ± 0.051 | - | 0.427 ± 0.176 |
| v2 LOSO | dinov2_vit_b14 | lora / basic | 12 | 0.921 ± 0.079 | 0.448 ± 0.210 | 0.021 ± 0.018 | - | 0.417 ± 0.172 |
| v2 LOSO | dinov2_vit_b14 | lora / strong | 12 | 0.925 ± 0.080 | 0.443 ± 0.195 | 0.047 ± 0.026 | - | 0.423 ± 0.151 |
| v2 LOSO | dinov2_vit_l14 | linear / basic | 12 | 0.903 ± 0.071 | 0.401 ± 0.253 | 0.086 ± 0.053 | - | 0.448 ± 0.216 |
| v2 LOSO | dinov2_vit_l14 | lora / strong | 12 | 0.929 ± 0.080 | 0.453 ± 0.211 | 0.042 ± 0.020 | - | 0.437 ± 0.163 |
| v2 LOSO | efficientnetv2_s | full / basic | 12 | 0.902 ± 0.073 | 0.428 ± 0.218 | 0.034 ± 0.015 | - | 0.378 ± 0.170 |
| v2 LOSO | efficientnetv2_s | full / strong | 12 | 0.905 ± 0.074 | 0.458 ± 0.254 | 0.038 ± 0.018 | - | 0.354 ± 0.196 |
| v2 LOSO | siglip2_vit_b16 | full / basic | 12 | 0.975 ± 0.011 | 0.418 ± 0.147 | 0.048 ± 0.076 | - | 0.478 ± 0.116 |
| v2 LOSO | siglip2_vit_b16 | full / strong | 12 | 0.960 ± 0.030 | 0.472 ± 0.213 | 0.040 ± 0.037 | - | 0.431 ± 0.167 |
| v2 few-shot NWRD | dinov2_vit_b14 | full/strong, k=10 | 5 | - | - | 0.974 ± 0.009 | 0.028 ± 0.002 | - |
| v2 few-shot NWRD | dinov2_vit_b14 | full/strong, k=25 | 5 | - | - | 0.979 ± 0.013 | 0.025 ± 0.004 | - |
| v2 few-shot NWRD | dinov2_vit_b14 | full/strong, k=50 | 5 | - | - | 0.986 ± 0.011 | 0.024 ± 0.005 | - |
| v2 few-shot NWRD | dinov2_vit_b14 | full/strong, k=100 | 5 | - | - | 0.989 ± 0.007 | 0.025 ± 0.003 | - |
| v2 few-shot NWRD | dinov2_vit_l14 | linear/basic, k=10 | 5 | - | - | 0.961 ± 0.026 | 0.049 ± 0.017 | - |
| v2 few-shot NWRD | dinov2_vit_l14 | linear/basic, k=25 | 5 | - | - | 0.975 ± 0.010 | 0.041 ± 0.018 | - |
| v2 few-shot NWRD | dinov2_vit_l14 | linear/basic, k=50 | 5 | - | - | 0.980 ± 0.004 | 0.032 ± 0.009 | - |
| v2 few-shot NWRD | dinov2_vit_l14 | linear/basic, k=100 | 5 | - | - | 0.983 ± 0.010 | 0.019 ± 0.018 | - |
| v2 few-shot NWRD | efficientnetv2_s | full/strong, k=10 | 5 | - | - | 0.951 ± 0.025 | 0.074 ± 0.025 | - |
| v2 few-shot NWRD | efficientnetv2_s | full/strong, k=25 | 5 | - | - | 0.964 ± 0.012 | 0.066 ± 0.025 | - |
| v2 few-shot NWRD | efficientnetv2_s | full/strong, k=50 | 5 | - | - | 0.983 ± 0.009 | 0.064 ± 0.022 | - |
| v2 few-shot NWRD | efficientnetv2_s | full/strong, k=100 | 5 | - | - | 0.972 ± 0.016 | 0.052 ± 0.015 | - |

LOSO test macro-F1 is averaged over all 4 held-out sources and 3 seeds.

## 2. Every model

### v1 baseline (9)

| # | model | setup | test on | seed | val macro-F1 | test macro-F1 | test accuracy | NWRD yellow-rust recall | healthy FPR | ECE | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny | full | wheat_v1 test | 0 | 0.865 | 0.838 | 0.863 | - | - | 0.057 | 463f5bae |
| 2 | convnext_tiny | full | wheat_v1 test | 1 | 0.863 | 0.836 | 0.857 | - | - | 0.034 | 73189d0b |
| 3 | convnext_tiny | full | wheat_v1 test | 2 | 0.845 | 0.821 | 0.851 | - | - | 0.024 | 089fa239 |
| 4 | convnext_tiny | full | wheat_v1 test | 42 | 0.858 | 0.843 | 0.866 | - | - | 0.037 | d18401f6 |
| 5 | tf_efficientnetv2_s | full | wheat_v1 test | 0 | 0.852 | 0.846 | 0.869 | - | - | 0.028 | 94142b12 |
| 6 | tf_efficientnetv2_s | full | wheat_v1 test | 1 | 0.852 | 0.843 | 0.864 | - | - | 0.023 | 1ec14ad0 |
| 7 | tf_efficientnetv2_s | full | wheat_v1 test | 2 | 0.860 | 0.849 | 0.872 | - | - | 0.018 | 1daa6583 |
| 8 | tf_efficientnetv2_s | full | wheat_v1 test | 42 | 0.853 | 0.849 | 0.868 | - | - | 0.022 | e4123b05 |
| 9 | vit_small_patch14_dinov2 | linear | wheat_v1 test | 42 | 0.808 | 0.799 | 0.821 | - | - | 0.021 | 384466b4 |

### v1 leave-source-out (4)

| # | model | setup | test on | seed | val macro-F1 | test macro-F1 | test accuracy | NWRD yellow-rust recall | healthy FPR | ECE | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | tf_efficientnetv2_s | full | wheat_v1 test (watermarked source held out) | 0 | 0.840 | 0.809 | 0.825 | - | - | 0.028 | af6d20f4 |
| 11 | tf_efficientnetv2_s | full | wheat_v1 test (watermarked source held out) | 1 | 0.837 | 0.813 | 0.832 | - | - | 0.021 | 3a1aceb0 |
| 12 | tf_efficientnetv2_s | full | wheat_v1 test (watermarked source held out) | 2 | 0.832 | 0.816 | 0.832 | - | - | 0.019 | 1215f894 |
| 13 | tf_efficientnetv2_s | full | wheat_v1 test (watermarked source held out) | 42 | 0.834 | 0.815 | 0.832 | - | - | 0.029 | 9f299562 |

### v2 pooled (9)

| # | model | setup | test on | seed | val macro-F1 | test macro-F1 | test accuracy | NWRD yellow-rust recall | healthy FPR | ECE | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | dinov2_vit_b14 | full / strong | pooled test split | 0 | 0.961 | 0.942 | 0.959 | 0.012 | - | 0.018 | 0217f25f |
| 15 | **dinov2_vit_b14** | **full / strong** | pooled test split | 1 | 0.976 | 0.943 | 0.962 | 0.059 | - | 0.025 | 59b15192 |
| 16 | dinov2_vit_b14 | full / strong | pooled test split | 42 | 0.973 | 0.956 | 0.972 | 0.064 | - | 0.012 | cad6fb9c |
| 17 | dinov2_vit_l14 | linear / basic | pooled test split | 0 | 0.945 | 0.932 | 0.946 | 0.120 | - | 0.018 | 6c1e2f32 |
| 18 | dinov2_vit_l14 | linear / basic | pooled test split | 1 | 0.939 | 0.933 | 0.949 | 0.122 | - | 0.017 | fc08e3ae |
| 19 | dinov2_vit_l14 | linear / basic | pooled test split | 42 | 0.938 | 0.935 | 0.949 | 0.125 | - | 0.019 | 5a76dbe8 |
| 20 | efficientnetv2_s | full / strong | pooled test split | 0 | 0.943 | 0.925 | 0.953 | 0.039 | - | 0.014 | 1b5e83a9 |
| 21 | efficientnetv2_s | full / strong | pooled test split | 1 | 0.946 | 0.914 | 0.946 | 0.046 | - | 0.015 | e201d10d |
| 22 | efficientnetv2_s | full / strong | pooled test split | 42 | 0.951 | 0.909 | 0.942 | 0.034 | - | 0.021 | 4ba61799 |

### v2 LOSO (120)

| # | model | setup | test on | seed | val macro-F1 | test macro-F1 | test accuracy | NWRD yellow-rust recall | healthy FPR | ECE | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | dinov2_vit_b14 | full / basic | held-out source: cgiar_wheat_rust | 0 | 0.796 | 0.442 | 0.383 | 0.088 | - | 0.555 | 4de6cf27 |
| 24 | dinov2_vit_b14 | full / basic | held-out source: cgiar_wheat_rust | 1 | 0.796 | 0.424 | 0.415 | 0.007 | - | 0.498 | 02d96282 |
| 25 | dinov2_vit_b14 | full / basic | held-out source: cgiar_wheat_rust | 42 | 0.793 | 0.426 | 0.385 | 0.117 | - | 0.550 | 9b41903a |
| 26 | dinov2_vit_b14 | full / basic | held-out source: kaggle | 0 | 0.947 | 0.278 | 0.460 | 0.002 | - | 0.482 | a7df85e6 |
| 27 | dinov2_vit_b14 | full / basic | held-out source: kaggle | 1 | 0.977 | 0.252 | 0.417 | 0.000 | - | 0.496 | 3cec63d0 |
| 28 | dinov2_vit_b14 | full / basic | held-out source: kaggle | 42 | 0.966 | 0.248 | 0.414 | 0.000 | - | 0.489 | 47745c13 |
| 29 | dinov2_vit_b14 | full / basic | held-out source: mendeley_iari_leaf_rust | 0 | 0.966 | 0.696 | 0.682 | 0.039 | - | 0.234 | 416feffb |
| 30 | dinov2_vit_b14 | full / basic | held-out source: mendeley_iari_leaf_rust | 1 | 0.962 | 0.903 | 0.868 | 0.005 | - | 0.028 | c9c39033 |
| 31 | dinov2_vit_b14 | full / basic | held-out source: mendeley_iari_leaf_rust | 42 | 0.963 | 0.772 | 0.734 | 0.027 | - | 0.167 | 5b37313f |
| 32 | dinov2_vit_b14 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.978 | 0.314 | 0.349 | 0.020 | - | 0.556 | 5f5ed906 |
| 33 | dinov2_vit_b14 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.965 | 0.354 | 0.353 | 0.032 | - | 0.518 | ca980cde |
| 34 | dinov2_vit_b14 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.978 | 0.370 | 0.369 | 0.017 | - | 0.548 | c30a6656 |
| 35 | dinov2_vit_b14 | full / strong | held-out source: cgiar_wheat_rust | 0 | 0.794 | 0.410 | 0.374 | 0.183 | - | 0.539 | 15cd8abf |
| 36 | dinov2_vit_b14 | full / strong | held-out source: cgiar_wheat_rust | 1 | 0.796 | 0.426 | 0.380 | 0.071 | - | 0.541 | a134bb60 |
| 37 | dinov2_vit_b14 | full / strong | held-out source: cgiar_wheat_rust | 42 | 0.790 | 0.442 | 0.370 | 0.144 | - | 0.502 | f3160e18 |
| 38 | dinov2_vit_b14 | full / strong | held-out source: kaggle | 0 | 0.970 | 0.251 | 0.384 | 0.022 | - | 0.478 | 4af8768b |
| 39 | dinov2_vit_b14 | full / strong | held-out source: kaggle | 1 | 0.962 | 0.254 | 0.359 | 0.010 | - | 0.475 | b8ab150f |
| 40 | dinov2_vit_b14 | full / strong | held-out source: kaggle | 42 | 0.965 | 0.250 | 0.400 | 0.024 | - | 0.485 | ccac6b91 |
| 41 | dinov2_vit_b14 | full / strong | held-out source: mendeley_iari_leaf_rust | 0 | 0.975 | 0.765 | 0.723 | 0.051 | - | 0.214 | bb829dc5 |
| 42 | dinov2_vit_b14 | full / strong | held-out source: mendeley_iari_leaf_rust | 1 | 0.970 | 0.806 | 0.765 | 0.056 | - | 0.166 | 955af824 |
| 43 | dinov2_vit_b14 | full / strong | held-out source: mendeley_iari_leaf_rust | 42 | 0.982 | 0.897 | 0.860 | 0.071 | - | 0.064 | a9c8278d |
| 44 | dinov2_vit_b14 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.971 | 0.302 | 0.339 | 0.059 | - | 0.561 | 494d41f1 |
| 45 | dinov2_vit_b14 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.977 | 0.431 | 0.410 | 0.068 | - | 0.478 | 4e051139 |
| 46 | dinov2_vit_b14 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.970 | 0.298 | 0.329 | 0.022 | - | 0.617 | 321cee27 |
| 47 | dinov2_vit_b14 | lora / basic | held-out source: cgiar_wheat_rust | 0 | 0.792 | 0.416 | 0.374 | 0.051 | - | 0.513 | 914f858d |
| 48 | dinov2_vit_b14 | lora / basic | held-out source: cgiar_wheat_rust | 1 | 0.794 | 0.421 | 0.385 | 0.029 | - | 0.512 | 9624f8e2 |
| 49 | dinov2_vit_b14 | lora / basic | held-out source: cgiar_wheat_rust | 42 | 0.790 | 0.448 | 0.403 | 0.044 | - | 0.451 | a031cbfe |
| 50 | dinov2_vit_b14 | lora / basic | held-out source: kaggle | 0 | 0.951 | 0.242 | 0.398 | 0.007 | - | 0.478 | 792a2b4d |
| 51 | dinov2_vit_b14 | lora / basic | held-out source: kaggle | 1 | 0.948 | 0.267 | 0.448 | 0.002 | - | 0.439 | d95188a1 |
| 52 | dinov2_vit_b14 | lora / basic | held-out source: kaggle | 42 | 0.958 | 0.257 | 0.407 | 0.002 | - | 0.442 | 12072e45 |
| 53 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_iari_leaf_rust | 0 | 0.962 | 0.821 | 0.779 | 0.032 | - | 0.105 | b345557a |
| 54 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_iari_leaf_rust | 1 | 0.969 | 0.860 | 0.828 | 0.024 | - | 0.064 | 416dd7de |
| 55 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_iari_leaf_rust | 42 | 0.962 | 0.617 | 0.618 | 0.015 | - | 0.304 | 2bcfb4f9 |
| 56 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.975 | 0.343 | 0.325 | 0.034 | - | 0.550 | 69c42bc5 |
| 57 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.977 | 0.366 | 0.319 | 0.000 | - | 0.571 | 8ca089bd |
| 58 | dinov2_vit_b14 | lora / basic | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.977 | 0.320 | 0.325 | 0.005 | - | 0.570 | d50b9de6 |
| 59 | dinov2_vit_b14 | lora / strong | held-out source: cgiar_wheat_rust | 0 | 0.795 | 0.434 | 0.395 | 0.061 | - | 0.509 | 543525ae |
| 60 | dinov2_vit_b14 | lora / strong | held-out source: cgiar_wheat_rust | 1 | 0.791 | 0.442 | 0.388 | 0.064 | - | 0.486 | 7de48eba |
| 61 | dinov2_vit_b14 | lora / strong | held-out source: cgiar_wheat_rust | 42 | 0.792 | 0.422 | 0.372 | 0.068 | - | 0.487 | 42e95c58 |
| 62 | dinov2_vit_b14 | lora / strong | held-out source: kaggle | 0 | 0.964 | 0.252 | 0.411 | 0.017 | - | 0.461 | 335c422b |
| 63 | dinov2_vit_b14 | lora / strong | held-out source: kaggle | 1 | 0.958 | 0.266 | 0.397 | 0.015 | - | 0.407 | bad3b7eb |
| 64 | dinov2_vit_b14 | lora / strong | held-out source: kaggle | 42 | 0.961 | 0.260 | 0.393 | 0.012 | - | 0.456 | 625161cb |
| 65 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 0 | 0.970 | 0.550 | 0.591 | 0.068 | - | 0.364 | e01daae5 |
| 66 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 1 | 0.975 | 0.849 | 0.804 | 0.066 | - | 0.095 | c9eea3b6 |
| 67 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 42 | 0.974 | 0.781 | 0.743 | 0.051 | - | 0.163 | 61fe99cc |
| 68 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.969 | 0.310 | 0.308 | 0.042 | - | 0.607 | 2f35a6c3 |
| 69 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.972 | 0.369 | 0.356 | 0.017 | - | 0.513 | c011dafd |
| 70 | dinov2_vit_b14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.977 | 0.379 | 0.366 | 0.086 | - | 0.531 | 2b6a0215 |
| 71 | dinov2_vit_l14 | linear / basic | held-out source: cgiar_wheat_rust | 0 | 0.786 | 0.403 | 0.321 | 0.164 | - | 0.527 | ca299701 |
| 72 | dinov2_vit_l14 | linear / basic | held-out source: cgiar_wheat_rust | 1 | 0.786 | 0.414 | 0.321 | 0.142 | - | 0.522 | 94dedb48 |
| 73 | dinov2_vit_l14 | linear / basic | held-out source: cgiar_wheat_rust | 42 | 0.786 | 0.417 | 0.325 | 0.144 | - | 0.517 | 5b00a085 |
| 74 | dinov2_vit_l14 | linear / basic | held-out source: kaggle | 0 | 0.930 | 0.157 | 0.294 | 0.017 | - | 0.555 | cc90aa99 |
| 75 | dinov2_vit_l14 | linear / basic | held-out source: kaggle | 1 | 0.955 | 0.161 | 0.300 | 0.015 | - | 0.555 | a434474b |
| 76 | dinov2_vit_l14 | linear / basic | held-out source: kaggle | 42 | 0.954 | 0.204 | 0.349 | 0.007 | - | 0.493 | f5131c92 |
| 77 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_iari_leaf_rust | 0 | 0.938 | 0.785 | 0.756 | 0.122 | - | 0.096 | e0047ee2 |
| 78 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_iari_leaf_rust | 1 | 0.939 | 0.763 | 0.734 | 0.098 | - | 0.128 | 98ece469 |
| 79 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_iari_leaf_rust | 42 | 0.933 | 0.821 | 0.792 | 0.095 | - | 0.079 | 1cc51e9e |
| 80 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.944 | 0.239 | 0.237 | 0.095 | - | 0.579 | cb0c98b3 |
| 81 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.939 | 0.211 | 0.166 | 0.078 | - | 0.673 | 0f9c9a2c |
| 82 | dinov2_vit_l14 | linear / basic | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.942 | 0.233 | 0.190 | 0.059 | - | 0.657 | 38f583d3 |
| 83 | dinov2_vit_l14 | lora / strong | held-out source: cgiar_wheat_rust | 0 | 0.796 | 0.426 | 0.393 | 0.054 | - | 0.530 | bb067ef4 |
| 84 | dinov2_vit_l14 | lora / strong | held-out source: cgiar_wheat_rust | 1 | 0.797 | 0.449 | 0.403 | 0.034 | - | 0.514 | 814a76de |
| 85 | dinov2_vit_l14 | lora / strong | held-out source: cgiar_wheat_rust | 42 | 0.794 | 0.447 | 0.399 | 0.054 | - | 0.498 | caf8d9d6 |
| 86 | dinov2_vit_l14 | lora / strong | held-out source: kaggle | 0 | 0.960 | 0.234 | 0.378 | 0.005 | - | 0.512 | 78d78d4d |
| 87 | dinov2_vit_l14 | lora / strong | held-out source: kaggle | 1 | 0.972 | 0.245 | 0.388 | 0.068 | - | 0.465 | 4f3060df |
| 88 | dinov2_vit_l14 | lora / strong | held-out source: kaggle | 42 | 0.961 | 0.243 | 0.384 | 0.024 | - | 0.490 | 327a2cb5 |
| 89 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 0 | 0.972 | 0.675 | 0.665 | 0.020 | - | 0.263 | 1c5c9f7d |
| 90 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 1 | 0.975 | 0.852 | 0.814 | 0.049 | - | 0.102 | f038d6be |
| 91 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_iari_leaf_rust | 42 | 0.983 | 0.797 | 0.758 | 0.032 | - | 0.168 | 6e434a56 |
| 92 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.981 | 0.380 | 0.359 | 0.042 | - | 0.569 | 39988db8 |
| 93 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.979 | 0.353 | 0.356 | 0.076 | - | 0.586 | a24e0970 |
| 94 | dinov2_vit_l14 | lora / strong | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.974 | 0.340 | 0.346 | 0.046 | - | 0.544 | 03fd33cb |
| 95 | efficientnetv2_s | full / basic | held-out source: cgiar_wheat_rust | 0 | 0.787 | 0.398 | 0.364 | 0.066 | - | 0.446 | 87910fb8 |
| 96 | efficientnetv2_s | full / basic | held-out source: cgiar_wheat_rust | 1 | 0.786 | 0.394 | 0.380 | 0.032 | - | 0.419 | e3d08e1a |
| 97 | efficientnetv2_s | full / basic | held-out source: cgiar_wheat_rust | 42 | 0.783 | 0.414 | 0.398 | 0.022 | - | 0.391 | 51116f52 |
| 98 | efficientnetv2_s | full / basic | held-out source: kaggle | 0 | 0.895 | 0.157 | 0.288 | 0.061 | - | 0.555 | 595f0fdc |
| 99 | efficientnetv2_s | full / basic | held-out source: kaggle | 1 | 0.920 | 0.232 | 0.383 | 0.017 | - | 0.448 | 95211090 |
| 100 | efficientnetv2_s | full / basic | held-out source: kaggle | 42 | 0.916 | 0.200 | 0.340 | 0.037 | - | 0.487 | 6f58af42 |
| 101 | efficientnetv2_s | full / basic | held-out source: mendeley_iari_leaf_rust | 0 | 0.944 | 0.792 | 0.742 | 0.039 | - | 0.047 | 9f1bf300 |
| 102 | efficientnetv2_s | full / basic | held-out source: mendeley_iari_leaf_rust | 1 | 0.951 | 0.837 | 0.799 | 0.022 | - | 0.058 | d764c926 |
| 103 | efficientnetv2_s | full / basic | held-out source: mendeley_iari_leaf_rust | 42 | 0.949 | 0.626 | 0.597 | 0.029 | - | 0.244 | 248d6aac |
| 104 | efficientnetv2_s | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.957 | 0.379 | 0.346 | 0.029 | - | 0.478 | 691c5486 |
| 105 | efficientnetv2_s | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.969 | 0.330 | 0.298 | 0.020 | - | 0.489 | 0e4f2633 |
| 106 | efficientnetv2_s | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.961 | 0.372 | 0.312 | 0.032 | - | 0.478 | 7a513ee3 |
| 107 | efficientnetv2_s | full / strong | held-out source: cgiar_wheat_rust | 0 | 0.785 | 0.390 | 0.367 | 0.066 | - | 0.466 | f7c0598b |
| 108 | efficientnetv2_s | full / strong | held-out source: cgiar_wheat_rust | 1 | 0.791 | 0.397 | 0.373 | 0.061 | - | 0.453 | daced77b |
| 109 | efficientnetv2_s | full / strong | held-out source: cgiar_wheat_rust | 42 | 0.785 | 0.384 | 0.351 | 0.042 | - | 0.437 | 9932cea6 |
| 110 | efficientnetv2_s | full / strong | held-out source: kaggle | 0 | 0.908 | 0.209 | 0.340 | 0.039 | - | 0.441 | 95199733 |
| 111 | efficientnetv2_s | full / strong | held-out source: kaggle | 1 | 0.949 | 0.186 | 0.321 | 0.020 | - | 0.500 | 37b4feb3 |
| 112 | efficientnetv2_s | full / strong | held-out source: kaggle | 42 | 0.894 | 0.223 | 0.357 | 0.044 | - | 0.448 | 8ede6e7b |
| 113 | efficientnetv2_s | full / strong | held-out source: mendeley_iari_leaf_rust | 0 | 0.957 | 0.848 | 0.807 | 0.054 | - | 0.042 | ecce9d12 |
| 114 | efficientnetv2_s | full / strong | held-out source: mendeley_iari_leaf_rust | 1 | 0.952 | 0.863 | 0.849 | 0.039 | - | 0.021 | 6e52d84c |
| 115 | efficientnetv2_s | full / strong | held-out source: mendeley_iari_leaf_rust | 42 | 0.959 | 0.868 | 0.855 | 0.032 | - | 0.027 | c91f72c7 |
| 116 | efficientnetv2_s | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.955 | 0.374 | 0.373 | 0.042 | - | 0.463 | 2bb70bf9 |
| 117 | efficientnetv2_s | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.961 | 0.379 | 0.356 | 0.010 | - | 0.484 | 341dc95a |
| 118 | efficientnetv2_s | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.964 | 0.374 | 0.349 | 0.007 | - | 0.464 | ba78b246 |
| 119 | siglip2_vit_b16 | full / basic | held-out source: cgiar_wheat_rust | 0 | 0.994 | 0.476 | 0.408 | 0.027 | - | 0.519 | 23fa57e1 |
| 120 | siglip2_vit_b16 | full / basic | held-out source: cgiar_wheat_rust | 1 | 0.984 | 0.471 | 0.414 | 0.002 | - | 0.446 | 2363196e |
| 121 | siglip2_vit_b16 | full / basic | held-out source: cgiar_wheat_rust | 42 | 0.992 | 0.481 | 0.379 | 0.271 | - | 0.548 | 8aca83ae |
| 122 | siglip2_vit_b16 | full / basic | held-out source: kaggle | 0 | 0.958 | 0.233 | 0.369 | 0.002 | - | 0.481 | da60d985 |
| 123 | siglip2_vit_b16 | full / basic | held-out source: kaggle | 1 | 0.962 | 0.260 | 0.443 | 0.000 | - | 0.476 | 1f930a50 |
| 124 | siglip2_vit_b16 | full / basic | held-out source: kaggle | 42 | 0.977 | 0.261 | 0.441 | 0.000 | - | 0.455 | f998c7c3 |
| 125 | siglip2_vit_b16 | full / basic | held-out source: mendeley_iari_leaf_rust | 0 | 0.969 | 0.519 | 0.582 | 0.081 | - | 0.394 | c9affeba |
| 126 | siglip2_vit_b16 | full / basic | held-out source: mendeley_iari_leaf_rust | 1 | 0.969 | 0.616 | 0.619 | 0.029 | - | 0.327 | 5baa8827 |
| 127 | siglip2_vit_b16 | full / basic | held-out source: mendeley_iari_leaf_rust | 42 | 0.977 | 0.690 | 0.686 | 0.044 | - | 0.246 | 747d99e3 |
| 128 | siglip2_vit_b16 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.980 | 0.328 | 0.332 | 0.017 | - | 0.612 | 8555863c |
| 129 | siglip2_vit_b16 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.969 | 0.361 | 0.305 | 0.076 | - | 0.594 | 1532bbf3 |
| 130 | siglip2_vit_b16 | full / basic | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.975 | 0.323 | 0.329 | 0.022 | - | 0.631 | 4f9fb788 |
| 131 | siglip2_vit_b16 | full / strong | held-out source: cgiar_wheat_rust | 0 | 0.915 | 0.521 | 0.421 | 0.112 | - | 0.430 | 60e149fc |
| 132 | siglip2_vit_b16 | full / strong | held-out source: cgiar_wheat_rust | 1 | 0.986 | 0.507 | 0.433 | 0.012 | - | 0.423 | 12cd517a |
| 133 | siglip2_vit_b16 | full / strong | held-out source: cgiar_wheat_rust | 42 | 0.881 | 0.516 | 0.418 | 0.017 | - | 0.407 | 8ffdf23d |
| 134 | siglip2_vit_b16 | full / strong | held-out source: kaggle | 0 | 0.969 | 0.252 | 0.409 | 0.017 | - | 0.462 | 455d3277 |
| 135 | siglip2_vit_b16 | full / strong | held-out source: kaggle | 1 | 0.966 | 0.242 | 0.404 | 0.042 | - | 0.499 | ddb37fcb |
| 136 | siglip2_vit_b16 | full / strong | held-out source: kaggle | 42 | 0.971 | 0.250 | 0.407 | 0.020 | - | 0.494 | 5d4d51b9 |
| 137 | siglip2_vit_b16 | full / strong | held-out source: mendeley_iari_leaf_rust | 0 | 0.966 | 0.600 | 0.599 | 0.112 | - | 0.349 | 48b503bb |
| 138 | siglip2_vit_b16 | full / strong | held-out source: mendeley_iari_leaf_rust | 1 | 0.968 | 0.825 | 0.778 | 0.034 | - | 0.165 | cc6e6f45 |
| 139 | siglip2_vit_b16 | full / strong | held-out source: mendeley_iari_leaf_rust | 42 | 0.975 | 0.869 | 0.827 | 0.059 | - | 0.096 | 44e4250c |
| 140 | siglip2_vit_b16 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 0 | 0.975 | 0.389 | 0.356 | 0.017 | - | 0.562 | 0c5a7d41 |
| 141 | siglip2_vit_b16 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 1 | 0.975 | 0.296 | 0.254 | 0.007 | - | 0.680 | 6a82d8cc |
| 142 | siglip2_vit_b16 | full / strong | held-out source: mendeley_wheat_leaf_ethiopia | 42 | 0.977 | 0.393 | 0.342 | 0.027 | - | 0.602 | 923daade |

### v2 few-shot NWRD (60)

| # | model | setup | test on | seed | val macro-F1 | test macro-F1 | test accuracy | NWRD yellow-rust recall | healthy FPR | ECE | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 143 | dinov2_vit_b14 | full/strong, k=10 draw 0 | NWRD test images | 0 | - | - | - | 0.965 | 0.025 | - | 4783850d |
| 144 | dinov2_vit_b14 | full/strong, k=10 draw 1 | NWRD test images | 1 | - | - | - | 0.985 | 0.029 | - | 5e953f4c |
| 145 | dinov2_vit_b14 | full/strong, k=10 draw 2 | NWRD test images | 2 | - | - | - | 0.980 | 0.029 | - | cacd1db1 |
| 146 | dinov2_vit_b14 | full/strong, k=10 draw 3 | NWRD test images | 3 | - | - | - | 0.975 | 0.029 | - | e6c09468 |
| 147 | dinov2_vit_b14 | full/strong, k=10 draw 4 | NWRD test images | 4 | - | - | - | 0.965 | 0.029 | - | 2c4346f2 |
| 148 | dinov2_vit_b14 | full/strong, k=25 draw 0 | NWRD test images | 0 | - | - | - | 0.960 | 0.025 | - | 71a6e7aa |
| 149 | dinov2_vit_b14 | full/strong, k=25 draw 1 | NWRD test images | 1 | - | - | - | 0.975 | 0.029 | - | 8727bd25 |
| 150 | dinov2_vit_b14 | full/strong, k=25 draw 2 | NWRD test images | 2 | - | - | - | 0.995 | 0.021 | - | 9f99aa1c |
| 151 | dinov2_vit_b14 | full/strong, k=25 draw 3 | NWRD test images | 3 | - | - | - | 0.985 | 0.021 | - | de7c393c |
| 152 | dinov2_vit_b14 | full/strong, k=25 draw 4 | NWRD test images | 4 | - | - | - | 0.980 | 0.029 | - | 4a266947 |
| 153 | dinov2_vit_b14 | full/strong, k=50 draw 0 | NWRD test images | 0 | - | - | - | 0.980 | 0.025 | - | f0475132 |
| 154 | dinov2_vit_b14 | full/strong, k=50 draw 1 | NWRD test images | 1 | - | - | - | 0.990 | 0.025 | - | 01fb0291 |
| 155 | dinov2_vit_b14 | full/strong, k=50 draw 2 | NWRD test images | 2 | - | - | - | 0.995 | 0.029 | - | 83452fe4 |
| 156 | dinov2_vit_b14 | full/strong, k=50 draw 3 | NWRD test images | 3 | - | - | - | 0.970 | 0.017 | - | 557c35cc |
| 157 | dinov2_vit_b14 | full/strong, k=50 draw 4 | NWRD test images | 4 | - | - | - | 0.995 | 0.025 | - | 2198d5ad |
| 158 | dinov2_vit_b14 | full/strong, k=100 draw 0 | NWRD test images | 0 | - | - | - | 0.995 | 0.025 | - | 17cdaa86 |
| 159 | dinov2_vit_b14 | full/strong, k=100 draw 1 | NWRD test images | 1 | - | - | - | 0.990 | 0.025 | - | 34fce5c0 |
| 160 | dinov2_vit_b14 | full/strong, k=100 draw 2 | NWRD test images | 2 | - | - | - | 0.980 | 0.025 | - | ace6da22 |
| 161 | dinov2_vit_b14 | full/strong, k=100 draw 3 | NWRD test images | 3 | - | - | - | 0.985 | 0.021 | - | 2c18e9e7 |
| 162 | dinov2_vit_b14 | full/strong, k=100 draw 4 | NWRD test images | 4 | - | - | - | 0.995 | 0.029 | - | 5806779d |
| 163 | dinov2_vit_l14 | linear/basic, k=10 draw 0 | NWRD test images | 0 | - | - | - | 0.975 | 0.062 | - | 2b72fe45 |
| 164 | dinov2_vit_l14 | linear/basic, k=10 draw 1 | NWRD test images | 1 | - | - | - | 0.970 | 0.058 | - | 2e530e00 |
| 165 | dinov2_vit_l14 | linear/basic, k=10 draw 2 | NWRD test images | 2 | - | - | - | 0.955 | 0.037 | - | 2832ef24 |
| 166 | dinov2_vit_l14 | linear/basic, k=10 draw 3 | NWRD test images | 3 | - | - | - | 0.919 | 0.025 | - | 9637acd1 |
| 167 | dinov2_vit_l14 | linear/basic, k=10 draw 4 | NWRD test images | 4 | - | - | - | 0.985 | 0.062 | - | 8cfba21f |
| 168 | dinov2_vit_l14 | linear/basic, k=25 draw 0 | NWRD test images | 0 | - | - | - | 0.970 | 0.041 | - | a8703dbf |
| 169 | dinov2_vit_l14 | linear/basic, k=25 draw 1 | NWRD test images | 1 | - | - | - | 0.960 | 0.037 | - | 32b5fc21 |
| 170 | dinov2_vit_l14 | linear/basic, k=25 draw 2 | NWRD test images | 2 | - | - | - | 0.985 | 0.021 | - | 5fad04b5 |
| 171 | dinov2_vit_l14 | linear/basic, k=25 draw 3 | NWRD test images | 3 | - | - | - | 0.980 | 0.037 | - | 5a5ef9d0 |
| 172 | dinov2_vit_l14 | linear/basic, k=25 draw 4 | NWRD test images | 4 | - | - | - | 0.980 | 0.071 | - | d463db53 |
| 173 | dinov2_vit_l14 | linear/basic, k=50 draw 0 | NWRD test images | 0 | - | - | - | 0.985 | 0.025 | - | 31f8048d |
| 174 | dinov2_vit_l14 | linear/basic, k=50 draw 1 | NWRD test images | 1 | - | - | - | 0.975 | 0.029 | - | be3bfc0c |
| 175 | dinov2_vit_l14 | linear/basic, k=50 draw 2 | NWRD test images | 2 | - | - | - | 0.980 | 0.046 | - | 153d6b18 |
| 176 | dinov2_vit_l14 | linear/basic, k=50 draw 3 | NWRD test images | 3 | - | - | - | 0.980 | 0.025 | - | bbe2f886 |
| 177 | dinov2_vit_l14 | linear/basic, k=50 draw 4 | NWRD test images | 4 | - | - | - | 0.980 | 0.033 | - | f17fcf23 |
| 178 | dinov2_vit_l14 | linear/basic, k=100 draw 0 | NWRD test images | 0 | - | - | - | 0.985 | 0.000 | - | 3ac2c7b4 |
| 179 | dinov2_vit_l14 | linear/basic, k=100 draw 1 | NWRD test images | 1 | - | - | - | 0.965 | 0.008 | - | 8dff00fb |
| 180 | dinov2_vit_l14 | linear/basic, k=100 draw 2 | NWRD test images | 2 | - | - | - | 0.985 | 0.033 | - | e6e089e9 |
| 181 | dinov2_vit_l14 | linear/basic, k=100 draw 3 | NWRD test images | 3 | - | - | - | 0.990 | 0.012 | - | 75fcfee2 |
| 182 | dinov2_vit_l14 | linear/basic, k=100 draw 4 | NWRD test images | 4 | - | - | - | 0.990 | 0.041 | - | 513c8867 |
| 183 | efficientnetv2_s | full/strong, k=10 draw 0 | NWRD test images | 0 | - | - | - | 0.929 | 0.041 | - | 51dceaf3 |
| 184 | efficientnetv2_s | full/strong, k=10 draw 1 | NWRD test images | 1 | - | - | - | 0.975 | 0.071 | - | 6739b542 |
| 185 | efficientnetv2_s | full/strong, k=10 draw 2 | NWRD test images | 2 | - | - | - | 0.939 | 0.079 | - | 3b1a78a7 |
| 186 | efficientnetv2_s | full/strong, k=10 draw 3 | NWRD test images | 3 | - | - | - | 0.929 | 0.066 | - | 72f60308 |
| 187 | efficientnetv2_s | full/strong, k=10 draw 4 | NWRD test images | 4 | - | - | - | 0.980 | 0.112 | - | b7c809fc |
| 188 | efficientnetv2_s | full/strong, k=25 draw 0 | NWRD test images | 0 | - | - | - | 0.955 | 0.054 | - | 38d60ef4 |
| 189 | efficientnetv2_s | full/strong, k=25 draw 1 | NWRD test images | 1 | - | - | - | 0.949 | 0.071 | - | 2e9f554e |
| 190 | efficientnetv2_s | full/strong, k=25 draw 2 | NWRD test images | 2 | - | - | - | 0.975 | 0.046 | - | 70e88001 |
| 191 | efficientnetv2_s | full/strong, k=25 draw 3 | NWRD test images | 3 | - | - | - | 0.965 | 0.054 | - | c0d5dbd9 |
| 192 | efficientnetv2_s | full/strong, k=25 draw 4 | NWRD test images | 4 | - | - | - | 0.975 | 0.108 | - | 5ddb9af5 |
| 193 | efficientnetv2_s | full/strong, k=50 draw 0 | NWRD test images | 0 | - | - | - | 0.985 | 0.041 | - | 65792f1b |
| 194 | efficientnetv2_s | full/strong, k=50 draw 1 | NWRD test images | 1 | - | - | - | 0.980 | 0.083 | - | ed5e10cf |
| 195 | efficientnetv2_s | full/strong, k=50 draw 2 | NWRD test images | 2 | - | - | - | 0.995 | 0.091 | - | 99a2a6a8 |
| 196 | efficientnetv2_s | full/strong, k=50 draw 3 | NWRD test images | 3 | - | - | - | 0.970 | 0.046 | - | f1ee3f1c |
| 197 | efficientnetv2_s | full/strong, k=50 draw 4 | NWRD test images | 4 | - | - | - | 0.985 | 0.058 | - | 1a97d8f1 |
| 198 | efficientnetv2_s | full/strong, k=100 draw 0 | NWRD test images | 0 | - | - | - | 0.985 | 0.050 | - | 2ed08635 |
| 199 | efficientnetv2_s | full/strong, k=100 draw 1 | NWRD test images | 1 | - | - | - | 0.960 | 0.041 | - | 2ba2f9aa |
| 200 | efficientnetv2_s | full/strong, k=100 draw 2 | NWRD test images | 2 | - | - | - | 0.949 | 0.041 | - | 95e7003a |
| 201 | efficientnetv2_s | full/strong, k=100 draw 3 | NWRD test images | 3 | - | - | - | 0.985 | 0.079 | - | 2c987997 |
| 202 | efficientnetv2_s | full/strong, k=100 draw 4 | NWRD test images | 4 | - | - | - | 0.980 | 0.050 | - | 0584d43b |
