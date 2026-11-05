# Experiments

Every run writes `config.yaml`, `meta.json` (seed, commit) and logs to `experiments/runs/<time>_<name>/` (gitignored). Copy results here only with their run directory named.

> The E1-E5 names below are inferred from the repo layout. **Replace them with the definitions in the project plan.**

| ID | Question | Configs | Metric | Baseline | Best | Run directory |
|---|---|---|---|---|---|---|
| E1 | Which backbone works best on public images? | `configs/vision/wheat_v1_efficientnetv2.yaml`, `wheat_v1_convnext_t.yaml`, `wheat_v1_dinov2_linear.yaml` | Macro-F1 | [`wheat_v1_baselines.md`](wheat_v1_baselines.md) | [`wheat_v1_baselines.md`](wheat_v1_baselines.md) | MLflow run IDs in that file |
| E2 | Does fine-tuning on local images help on unseen sites? | `configs/vision/finetune_local.yaml` | Macro-F1 on `field_test`, ECE, coverage-accuracy | [F1] | [F1] | [RUN] |
| E3 | Which retriever works best across the three languages? | `configs/nlp/retriever.yaml` | Recall@5, MRR | [R@5] | [R@5] | [RUN] |
| E4 | Native generation or fine-tuning, versus the translate baseline? | `configs/nlp/generator_bakeoff.yaml`, `qlora.yaml`, `translate_baseline.yaml` | Expert rubric | [SCORE] | [SCORE] | [RUN] |
| E5 | Does image + text fusion help in the farmer pilot? | `configs/app.yaml` | Expert rubric, SUS | [SCORE] | [SUS] | [RUN] |
