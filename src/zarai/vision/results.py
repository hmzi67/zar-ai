"""Fill experiments/wheat_v1_baselines.md from logged MLflow runs. Prints the selected run last.

Numbers only ever come from MLflow: anything not logged is written as [TBD]. The model is selected
on validation macro-F1 alone, and only runs on the current DVC dataset version are counted.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from zarai.vision.eval import FOCUS_PAIRS
from zarai.vision.tracking import dvc_hashes, set_tracking_uri

BASELINES = {
    "wheat_v1_efficientnetv2": "EfficientNetV2-S",
    "wheat_v1_convnext_t": "ConvNeXt-T",
    "wheat_v1_dinov2_linear": "DINOv2 ViT-S/14 linear probe",
}
BASELINE_SEED = "42"
RERUN_SEEDS = ("0", "1", "2")
TBD = "[TBD]"
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def number(value: float | None, digits: int = 4) -> str:
    return TBD if value is None else f"{value:.{digits}f}"


def with_ci(metrics: dict, name: str) -> str:
    """Format a metric as `value (low-high)`, or [TBD] if any part wasn't logged."""
    parts = [metrics.get(name), metrics.get(f"{name}_low"), metrics.get(f"{name}_high")]
    if any(part is None for part in parts):
        return TBD
    return f"{parts[0]:.4f} ({parts[1]:.4f}-{parts[2]:.4f})"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def select_best(runs: dict[tuple[str, str], dict]) -> tuple[str, dict] | None:
    """Return (config, run) of the seed 42 baseline with the highest validation macro-F1."""
    scored = [
        (config, runs[config, BASELINE_SEED])
        for config in BASELINES
        if (config, BASELINE_SEED) in runs
        and runs[config, BASELINE_SEED]["metrics"].get("best_val_macro_f1") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1]["metrics"]["best_val_macro_f1"])


def _epochs(metrics: dict) -> str:
    if metrics.get("best_epoch") is None or metrics.get("epochs_run") is None:
        return TBD
    return f"{metrics['best_epoch']:.0f} / {metrics['epochs_run']:.0f}"


def _result_row(label: str, run: dict | None) -> list:
    metrics = run["metrics"] if run else {}
    return [
        label,
        number(metrics.get("best_val_macro_f1")),
        _epochs(metrics),
        with_ci(metrics, "test_macro_f1"),
        with_ci(metrics, "test_accuracy"),
        number(metrics.get("test_ece_before")),
        number(metrics.get("test_ece_after")),
        number(metrics.get("temperature"), 3),
        f"`{run['run_id']}`" if run else TBD,
    ]


RESULT_HEADER = [
    "model",
    "val macro-F1 (best)",
    "best epoch / epochs run",
    "test macro-F1 (95% CI)",
    "test accuracy (95% CI)",
    "test ECE before",
    "test ECE after",
    "temperature",
    "MLflow run ID",
]


def seed_summary(runs: dict[tuple[str, str], dict]) -> list[list]:
    """Mean and sample SD per model over all logged seeds, for models with at least two."""
    columns = (
        "best_val_macro_f1",
        "test_macro_f1",
        "test_accuracy",
        "test_ece_after",
        "test_accuracy_at_threshold",
    )
    rows = []
    for config, name in BASELINES.items():
        seeds = sorted((s for c, s in runs if c == config), key=int)
        if len(seeds) < 2:
            continue
        row = [name, ", ".join(seeds)]
        for column in columns:
            values = [runs[config, s]["metrics"].get(column) for s in seeds]
            if any(value is None for value in values):
                row.append(TBD)
            else:
                row.append(f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
        rows.append(row)
    return rows


def render_results(
    runs: dict[tuple[str, str], dict],
    per_class: dict | None,
    top_errors: list[dict] | None,
    dvc: dict[str, str],
) -> list[str]:
    """Build the generated part of the results document."""
    best = select_best(runs)
    baseline = {config: runs.get((config, BASELINE_SEED)) for config in BASELINES}
    lines = [
        f"Dataset DVC hashes: processed `{dvc['dvc.processed_md5']}`, "
        f"splits `{dvc['dvc.splits_md5']}`, raw `{dvc['dvc.raw_md5']}`.",
        "",
        f"## Baselines (seed {BASELINE_SEED})",
        "",
        *table(
            RESULT_HEADER,
            [[BASELINES[c], *_result_row(c, baseline[c])[1:]] for c in BASELINES],
        ),
        "",
        f"Highest seed {BASELINE_SEED} validation macro-F1: "
        f"**{BASELINES[best[0]] if best else TBD}**.",
        "",
        f"## Reruns of {BASELINES[best[0]] if best else 'the selected model'}",
        "",
    ]
    rerun_rows = [
        _result_row(f"seed {seed}", runs.get((best[0], seed)) if best else None)
        for seed in RERUN_SEEDS
    ]
    lines += [*table(["seed", *RESULT_HEADER[1:]], rerun_rows), ""]

    lines += ["## All logged seeds", ""]
    all_rows = [
        [f"{BASELINES[config]}, seed {seed}", *_result_row("", runs[config, seed])[1:]]
        for config in BASELINES
        for seed in sorted((s for c, s in runs if c == config), key=int)
    ]
    lines += table(["run", *RESULT_HEADER[1:]], all_rows) if all_rows else [TBD]
    lines += ["", "## Mean and SD across seeds", ""]
    summary = seed_summary(runs)
    if summary:
        lines += table(
            [
                "model",
                "seeds",
                "val macro-F1",
                "test macro-F1",
                "test accuracy",
                "test ECE after",
                "accuracy when answering",
            ],
            summary,
        )
    else:
        lines.append(TBD)
    lines.append("")

    lines += ["## Confusion between similar classes (test)", ""]
    pair_rows = []
    for a, b in FOCUS_PAIRS:
        row = [f"`{a}` / `{b}`"]
        for config in BASELINES:
            metrics = baseline[config]["metrics"] if baseline[config] else {}
            a_as_b = metrics.get(f"test_confusion_{a}_as_{b}")
            b_as_a = metrics.get(f"test_confusion_{b}_as_{a}")
            row.append(TBD if a_as_b is None or b_as_a is None else f"{a_as_b:.0f} / {b_as_a:.0f}")
        pair_rows.append(row)
    lines += [
        "Each cell is the count of `a` predicted as `b`, then `b` predicted as `a`.",
        "",
        *table(["pair", *BASELINES.values()], pair_rows),
        "",
        "## Background and watermark checks (test)",
        "",
        "Grad-CAM grids are in each run's `plots/gradcam_grid.png`. If the mass on dark pixels is "
        "well above the dark area, the model is looking at the background. Watermarks are only "
        "checked by viewing the grid.",
        "",
    ]
    check_rows = []
    for name, key in (
        ("Mildew recall, black background", "test_mildew_recall_dark"),
        ("Mildew recall, other", "test_mildew_recall_other"),
        ("Mildew: Grad-CAM mass on dark pixels", "cam_mildew_mass_on_dark"),
        ("Mildew: dark area", "cam_mildew_dark_area"),
        ("Fusarium: Grad-CAM mass on dark pixels", "cam_fusarium_head_blight_mass_on_dark"),
        ("Fusarium: dark area", "cam_fusarium_head_blight_dark_area"),
    ):
        check_rows.append(
            [
                name,
                *[
                    number((baseline[c]["metrics"] if baseline[c] else {}).get(key), 3)
                    for c in BASELINES
                ],
            ]
        )
    lines += [*table(["check", *BASELINES.values()], check_rows), ""]

    lines += ["## Abstention (test, threshold chosen on val)", ""]
    abstain_rows = []
    for config in BASELINES:
        metrics = baseline[config]["metrics"] if baseline[config] else {}
        abstain_rows.append(
            [
                BASELINES[config],
                number(metrics.get("abstain_threshold"), 3),
                number(metrics.get("test_coverage_at_threshold")),
                number(metrics.get("test_accuracy_at_threshold")),
            ]
        )
    lines += [*table(["model", "threshold", "coverage", "accuracy on answered"], abstain_rows), ""]

    selected = best[1] if best else None
    onnx = selected["metrics"].get("onnx_max_abs_diff") if selected else None
    lines += [
        "## ONNX export of the selected model",
        "",
        f"Largest absolute logit difference from PyTorch: {TBD if onnx is None else f'{onnx:.2e}'} "
        "(required: at most 1e-4).",
        "",
        "## Per-class results for the selected model (test)",
        "",
    ]
    if per_class:
        lines += table(
            ["class", "precision", "recall", "F1", "support"],
            [
                [
                    name,
                    number(values["precision"]),
                    number(values["recall"]),
                    number(values["f1-score"]),
                    f"{values['support']:.0f}",
                ]
                for name, values in per_class.items()
            ],
        )
    else:
        lines.append(TBD)
    lines += ["", "## Most common errors for the selected model (test)", ""]
    if top_errors:
        lines += table(
            ["true", "predicted", "count"],
            [[e["true"], e["predicted"], e["count"]] for e in top_errors],
        )
    else:
        lines.append(TBD)
    return lines


def replace_section(document: str, generated: list[str]) -> str:
    """Replace only the text between the markers, so the written analysis survives a rerun."""
    if START not in document or END not in document:
        raise ValueError(f"results document needs the markers {START} and {END}")
    before = document[: document.index(START) + len(START)]
    after = document[document.index(END) :]
    return before + "\n" + "\n".join(generated) + "\n" + after


def load_runs(tracking_uri: str, experiment: str, dvc: dict[str, str]) -> dict:
    """Return the latest finished run per (config, seed) on the current dataset version."""
    import mlflow

    set_tracking_uri(tracking_uri)
    found = mlflow.search_runs(
        experiment_names=[experiment],
        output_format="list",
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    runs = {}
    for run in found:
        params = run.data.params
        if params.get("dvc.processed_md5") != dvc["dvc.processed_md5"]:
            continue
        runs[params.get("experiment"), params.get("seed")] = {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
        }
    return runs


def _artifact_json(run_id: str, path: str):
    import mlflow

    try:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    except Exception:  # MLflow raises several types for a missing artifact.
        return None
    return json.loads(Path(local).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="wheat_v1_baselines")
    parser.add_argument("--doc", type=Path, default=Path("experiments/wheat_v1_baselines.md"))
    parser.add_argument("--splits-csv", default="data/splits/wheat_v1.csv")
    args = parser.parse_args()

    dvc = dvc_hashes(args.splits_csv)
    runs = load_runs(args.tracking_uri, args.experiment, dvc)
    best = select_best(runs)
    per_class = _artifact_json(best[1]["run_id"], "metrics/per_class.json") if best else None
    errors = _artifact_json(best[1]["run_id"], "metrics/top_confusions.json") if best else None
    generated = render_results(runs, per_class, errors, dvc)
    args.doc.write_text(
        replace_section(args.doc.read_text(encoding="utf-8"), generated), encoding="utf-8"
    )
    if best:
        print(f"{best[0]} {best[1]['run_id']}")


if __name__ == "__main__":
    main()
