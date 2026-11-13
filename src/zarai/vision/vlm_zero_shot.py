"""Gemma 4 26B-A4B-it (4-bit) as a zero-shot wheat disease classifier, next to DINOv2-B.

The test sets are the wheat_v2 ones: the pooled test split, each LOSO fold's held-out source and
the NWRD tiles. Gemma doesn't train, so it is run once per image over their union, and each test
set is scored from those predictions. The prompt, image token budgets and in-context references
are fixed in configs/vision/vlm_zero_shot.yaml.

An answer counts as a prediction only if it parses as JSON with a known label. "unsure" and
invalid answers are scored as wrong (an extra "no label" column), and their rates are reported.

    python -m zarai.vision.vlm_zero_shot run --variant zero_shot_280 [--limit N]
    python -m zarai.vision.vlm_zero_shot latency --variant zero_shot_280
    python -m zarai.vision.vlm_zero_shot dinov2-timing
    python -m zarai.vision.vlm_zero_shot report
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

CONFIG = "configs/vision/vlm_zero_shot.yaml"
NO_LABEL = "no_label"  # unsure or invalid, scored as a sixth, always-wrong prediction


def load_cfg(path: str = CONFIG):
    return OmegaConf.load(path)


# ---------------------------------------------------------------------------------------------
# Test sets and references


def test_sets(rows) -> dict[str, list[dict]]:
    """{name: rows}: the pooled test split, each LOSO held-out source, and NWRD (CSV order, as
    the vision runs saw them)."""
    sets = {"pooled_test": [r for r in rows if r["split_pooled"] == "test"]}
    for column in rows[0]:
        if column.startswith("split_loso_"):
            sets[f"loso:{column[len('split_loso_') :]}"] = [r for r in rows if r[column] == "test"]
    sets["nwrd"] = [r for r in rows if r["split_pooled"] == "nwrd"]
    return sets


def images_to_run(rows) -> list[str]:
    """Every image in any test set, once, in CSV order."""
    wanted = {r["path"] for rs in test_sets(rows).values() for r in rs}
    return [r["path"] for r in rows if r["path"] in wanted]


def references(rows, classes, shots: int, seed: int) -> list[tuple[str, dict]]:
    """`shots` (label, row) per class from the pooled train split, in class order."""
    rng = np.random.default_rng(seed)
    out = []
    for label in classes:
        pool = [r for r in rows if r["split_pooled"] == "train" and r["label"] == label]
        for i in sorted(rng.choice(len(pool), size=shots, replace=False).tolist()):
            out.append((label, pool[i]))
    return out


# ---------------------------------------------------------------------------------------------
# Prompt and parsing


def task_text(cfg) -> str:
    p = cfg.prompt
    lines = "\n".join(f"- {key}: {desc}" for key, desc in p.classes.items())
    return p.task.format(classes=lines, labels=", ".join(p.classes))


def messages(cfg, image, refs: list[tuple[str, object]]) -> list[dict]:
    """One chat: system prompt, then (optional) labelled references, the photo and the task."""
    p = cfg.prompt
    content = []
    if refs:
        content.append({"type": "text", "text": p.references_intro})
        for label, ref in refs:
            content.append({"type": "text", "text": p.reference.format(label=label)})
            content.append({"type": "image_pil", "image_pil": ref})
        content.append({"type": "text", "text": p.query_intro})
    content.append({"type": "image_pil", "image_pil": image})
    content.append({"type": "text", "text": task_text(cfg)})
    return [
        {"role": "system", "content": p.system},
        {"role": "user", "content": content},
    ]


_JSON = re.compile(r"\{.*\}", re.S)


def parse(text: str, labels) -> dict:
    """{valid, label, confidence, evidence}. Valid = a JSON object with a known label and a
    confidence in [0, 1]; a ```json fence around it is allowed."""
    out = {"valid": False, "label": None, "confidence": None, "evidence": None}
    found = _JSON.search(text or "")
    if not found:
        return out
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    label = str(data.get("label", "")).strip().lower()
    confidence = data.get("confidence")
    if label not in labels or isinstance(confidence, bool):
        return out
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return out
    if not 0.0 <= confidence <= 1.0:
        return out
    out.update(valid=True, label=label, confidence=confidence, evidence=data.get("evidence"))
    return out


# ---------------------------------------------------------------------------------------------
# Running Gemma


def _load_image(path: str):
    from zarai.data.clean_wheat import _open_rgb

    return _open_rgb(path)[0]


def build_llm(cfg, variant):
    """A vLLM engine sized for one variant: its context length, image count and image budget
    (so memory profiling assumes this variant's images, not the largest)."""
    from vllm import LLM

    engine = OmegaConf.to_container(OmegaConf.merge(cfg.model.engine, variant.get("engine", {})))
    return LLM(
        model=cfg.model.id,
        revision=cfg.model.revision,
        limit_mm_per_prompt={"image": 1 + int(variant.shots) * len(cfg.classes)},
        mm_processor_kwargs={"max_soft_tokens": int(variant.budget)},
        seed=cfg.generation.seed,
        **engine,
    )


def _generate(llm, cfg, variant, images, refs):
    from vllm import SamplingParams

    g = cfg.generation
    params = SamplingParams(temperature=g.temperature, max_tokens=g.max_tokens, seed=g.seed)
    outputs = llm.chat(
        [messages(cfg, image, refs) for image in images],
        params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": bool(cfg.model.enable_thinking)},
        mm_processor_kwargs={"max_soft_tokens": int(variant.budget)},
    )
    return [
        (o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outputs
    ]


def _variant_refs(cfg, rows, variant):
    if not variant.shots:
        return [], []
    picked = references(rows, list(cfg.classes), int(variant.shots), int(cfg.references.seed))
    return picked, [(label, _load_image(r["path"])) for label, r in picked]


def run(cfg, name: str, limit: int | None = None, log=print) -> Path:
    """Predict every test image with one variant; appends to <output.dir>/<name>.jsonl."""
    from concurrent.futures import ThreadPoolExecutor

    from zarai.common.io import read_csv

    variant = cfg.variants[name]
    rows = read_csv(cfg.splits_csv)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["path"] for line in out.read_text().splitlines() if line.strip()}
    picked, refs = _variant_refs(cfg, rows, variant)
    (out_dir / f"{name}.references.json").write_text(
        json.dumps(
            [{"label": lab, "path": r["path"], "source": r["source"]} for lab, r in picked],
            indent=1,
        )
    )
    todo = [p for p in images_to_run(rows) if p not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"{name}: {len(done)} done, {len(todo)} to run")
    llm = build_llm(cfg, variant)
    labels = list(cfg.prompt.classes)
    with ThreadPoolExecutor(8) as pool, out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), int(cfg.batch)):
            paths = todo[start : start + int(cfg.batch)]
            images = list(pool.map(_load_image, paths))
            t0 = time.perf_counter()
            results = _generate(llm, cfg, variant, images, refs)
            seconds = time.perf_counter() - t0
            for path, (text, n_in, n_out) in zip(paths, results, strict=True):
                record = {
                    "path": path,
                    "text": text,
                    "prompt_tokens": n_in,
                    "output_tokens": n_out,
                    "batch_seconds": seconds,
                    "batch_size": len(paths),
                    **parse(text, labels),
                }
                f.write(json.dumps(record) + "\n")
            f.flush()
            log(f"{name}: {start + len(paths)}/{len(todo)}  {seconds / len(paths):.3f} s/image")
    return out


def latency(cfg, name: str, log=print) -> dict:
    """Seconds per image at batch size 1 (one request at a time), image decode included."""
    from zarai.common.io import read_csv

    variant = cfg.variants[name]
    rows = read_csv(cfg.splits_csv)
    paths = [r["path"] for r in test_sets(rows)["pooled_test"]][: int(cfg.timing.latency_images)]
    _, refs = _variant_refs(cfg, rows, variant)
    llm = build_llm(cfg, variant)
    _generate(llm, cfg, variant, [_load_image(paths[0])], refs)  # warm-up
    times = []
    for path in paths:
        t0 = time.perf_counter()
        _generate(llm, cfg, variant, [_load_image(path)], refs)
        times.append(time.perf_counter() - t0)
    result = {
        "variant": name,
        "n": len(times),
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
    }
    _merge_timing(cfg, f"latency:{name}", result)
    log(result)
    return result


def _merge_timing(cfg, key: str, value) -> None:
    path = Path(cfg.output.dir) / "timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------------------------------------
# DINOv2-B


def dinov2_runs(cfg) -> dict[str, dict[str, str]]:
    """{fold: {seed: run_id}} for the DINOv2-B grid runs (latest run per seed)."""
    import mlflow

    from zarai.vision.tracking import set_tracking_uri

    d = cfg.dinov2
    set_tracking_uri(d.tracking_uri)
    where = " and ".join(f"tags.{k} = '{v}'" for k, v in d.tags.items())
    found = mlflow.search_runs(
        experiment_names=[d.experiment],
        output_format="list",
        filter_string=f"attributes.status = 'FINISHED' and tags.dry_run = 'false' and {where}",
        order_by=["attributes.start_time ASC"],
    )
    runs: dict[str, dict[str, str]] = {}
    for run in found:
        runs.setdefault(run.data.tags["fold"], {})[run.data.tags["seed"]] = run.info.run_id
    return runs


def dinov2_predictions(cfg, sets) -> dict[str, dict[str, np.ndarray]]:
    """{test set: {seed: predicted class index per row}} from the logits each run logged."""
    from zarai.vision.tracking import artifact_path

    classes = list(cfg.classes)
    runs = dinov2_runs(cfg)
    out = {}
    for name, rows in sets.items():
        fold, split = (
            ("pooled", "nwrd")
            if name == "nwrd"
            else (
                ("pooled", "heldout")
                if name == "pooled_test"
                else (name.split(":", 1)[1], "heldout")
            )
        )
        out[name] = {}
        for seed, run_id in sorted(runs[fold].items()):
            data = np.load(artifact_path(run_id, f"eval/{split}/{split}_logits.npz"))
            labels = np.array([classes.index(r["label"]) for r in rows])
            if not np.array_equal(data["labels"], labels):
                raise ValueError(f"{name}: run {run_id} logits are not in CSV order")
            out[name][seed] = data["logits"].argmax(axis=1)
    return out


def dinov2_timing(cfg, log=print) -> dict:
    """Seconds per image for the served DINOv2-B run: batched (64) and batch 1, decode included."""
    import torch

    from zarai.common.io import read_csv
    from zarai.vision.serving import load_run_model
    from zarai.vision.tracking import set_tracking_uri
    from zarai.vision.transforms import build_transforms

    set_tracking_uri(cfg.dinov2.tracking_uri)
    model, checkpoint, run_cfg = load_run_model(cfg.dinov2.timing_run, "cuda")
    model.eval()
    transform = build_transforms(
        run_cfg.data.image_size,
        train=False,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        eval_resize=run_cfg.data.eval_resize,
    )
    rows = read_csv(cfg.splits_csv)
    paths = [r["path"] for r in test_sets(rows)["pooled_test"]]

    def predict(batch_paths):
        x = torch.stack([transform(_load_image(p)) for p in batch_paths]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x).float().cpu()
        torch.cuda.synchronize()

    predict(paths[:8])
    t0 = time.perf_counter()
    for start in range(0, len(paths), 64):
        predict(paths[start : start + 64])
    batched = (time.perf_counter() - t0) / len(paths)
    times = []
    for p in paths[: int(cfg.timing.latency_images)]:
        t0 = time.perf_counter()
        predict([p])
        times.append(time.perf_counter() - t0)
    result = {
        "batched_s": batched,
        "n_batched": len(paths),
        "latency_median_s": float(np.median(times)),
        "n_latency": len(times),
    }
    _merge_timing(cfg, "dinov2", result)
    log(result)
    return result


# ---------------------------------------------------------------------------------------------
# Scoring


def scores(labels: np.ndarray, preds: np.ndarray, n_classes: int, seed: int = 42) -> dict:
    """Macro-F1 (95% CI) over the classes present, accuracy and per-class recall. A prediction
    index of n_classes means "no label" and is always wrong."""
    from zarai.eval.vision_metrics import bootstrap_class_scores

    present = sorted(int(c) for c in np.unique(labels))
    s = bootstrap_class_scores(labels, preds, present, n_classes + 1, 1000, seed)
    return {k: v for k, v in s.items()}


def load_predictions(cfg, name: str) -> dict[str, dict]:
    path = Path(cfg.output.dir) / f"{name}.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["path"]] = r
    return records


def gemma_index(record, classes) -> int:
    """Class index, or len(classes) for unsure / invalid."""
    if record["valid"] and record["label"] in classes:
        return classes.index(record["label"])
    return len(classes)


def answer_column(record, classes) -> int:
    """Confusion-matrix column: the class index, then len(classes) for unsure and
    len(classes) + 1 for an invalid answer."""
    if not record["valid"]:
        return len(classes) + 1
    if record["label"] == "unsure":
        return len(classes)
    return classes.index(record["label"])


def confusion(labels, columns, n_classes) -> list[list[int]]:
    """Counts, rows = true class, columns = the classes, then unsure, then invalid."""
    m = np.zeros((n_classes, n_classes + 2), dtype=int)
    np.add.at(m, (np.asarray(labels, dtype=int), np.asarray(columns, dtype=int)), 1)
    return m.tolist()


def called_as(matrix, true_idx: int, pred_idx: int) -> tuple[int, int]:
    """(images of true_idx answered pred_idx, images of true_idx)."""
    row = matrix[true_idx]
    return int(row[pred_idx]), int(sum(row))


def precision_of(matrix, idx: int) -> tuple[int, int]:
    """(correct answers of class idx, all answers of class idx)."""
    return int(matrix[idx][idx]), int(sum(row[idx] for row in matrix))


def agreement(labels, dino, gemma, n_classes) -> dict:
    """Agreement of DINOv2-B and Gemma, and accuracy when they agree or disagree. Gemma answers
    of unsure / invalid count as disagreement and are also split out."""
    agree = dino == gemma
    gemma_labelled = gemma < n_classes
    disagree = ~agree & gemma_labelled
    out = {
        "agree": float(agree.mean()),
        "acc_agree": float((dino[agree] == labels[agree]).mean()) if agree.any() else None,
        "disagree": float(disagree.mean()),
        "dino_acc_disagree": (
            float((dino[disagree] == labels[disagree]).mean()) if disagree.any() else None
        ),
        "gemma_acc_disagree": (
            float((gemma[disagree] == labels[disagree]).mean()) if disagree.any() else None
        ),
        "gemma_no_label": float((~gemma_labelled).mean()),
        "dino_acc_gemma_no_label": (
            float((dino[~gemma_labelled] == labels[~gemma_labelled]).mean())
            if (~gemma_labelled).any()
            else None
        ),
    }
    return out


# ---------------------------------------------------------------------------------------------
# Report


def _f(x, digits=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.1%}" if pct else f"{x:.{digits}f}"


def evaluate(cfg) -> dict:
    """Everything the report needs, from the saved predictions and the MLflow logits."""
    from zarai.common.io import read_csv

    classes = list(cfg.classes)
    n = len(classes)
    rows = read_csv(cfg.splits_csv)
    label_of = {r["path"]: r["label"] for r in rows}
    sets = test_sets(rows)
    dino = dinov2_predictions(cfg, sets)
    result = {"sets": {}, "variants": {}, "dino": {}}
    for set_name, set_rows in sets.items():
        labels = np.array([classes.index(r["label"]) for r in set_rows])
        result["sets"][set_name] = {
            "n": len(set_rows),
            "classes": sorted({r["label"] for r in set_rows}),
        }
        result["dino"][set_name] = {
            seed: scores(labels, preds, n) for seed, preds in dino[set_name].items()
        }
    for name, variant in cfg.variants.items():
        records = load_predictions(cfg, name)
        if not records:
            continue
        ref_path = Path(cfg.output.dir) / f"{name}.references.json"
        ref_paths = (
            {r["path"] for r in json.loads(ref_path.read_text())} if ref_path.exists() else set()
        )
        per_set = {}
        for set_name, set_rows in sets.items():
            keep = [
                i
                for i, r in enumerate(set_rows)
                if r["path"] in records and r["path"] not in ref_paths
            ]
            if len(keep) < len(set_rows) - len(ref_paths & {r["path"] for r in set_rows}):
                continue  # not finished for this set
            labels = np.array([classes.index(set_rows[i]["label"]) for i in keep])
            recs = [records[set_rows[i]["path"]] for i in keep]
            preds = np.array([gemma_index(r, classes) for r in recs])
            per_set[set_name] = {
                "n": len(keep),
                "excluded_references": len(set_rows) - len(keep),
                "scores": scores(labels, preds, n),
                "unsure": float(np.mean([r["valid"] and r["label"] == "unsure" for r in recs])),
                "invalid": float(np.mean([not r["valid"] for r in recs])),
                "confusion": confusion(labels, [answer_column(r, classes) for r in recs], n),
                "agreement": {
                    seed: agreement(labels, d[keep], preds, n) for seed, d in dino[set_name].items()
                },
            }
        all_recs = list(records.values())
        # All sets combined: every image once (the pooled test split is inside the LOSO sets).
        union = [r for r in all_recs if r["path"] not in ref_paths]
        combined = {
            "n": len(union),
            "confusion": confusion(
                [classes.index(label_of[r["path"]]) for r in union],
                [answer_column(r, classes) for r in union],
                n,
            ),
        }
        per_image = [r["batch_seconds"] / r["batch_size"] for r in all_recs]
        result["variants"][name] = {
            "budget": int(variant.budget),
            "shots": int(variant.shots),
            "n_images": len(all_recs),
            "batched_s": float(np.sum(per_image) / len(per_image)),
            "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in all_recs])),
            "mean_output_tokens": float(np.mean([r["output_tokens"] for r in all_recs])),
            "sets": per_set,
            "combined": combined,
        }
    timing_path = Path(cfg.output.dir) / "timings.json"
    result["timings"] = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    return result


def _mean_seeds(by_seed: dict, key: str):
    vals = [s[key]["value"] if isinstance(s[key], dict) else s[key] for s in by_seed.values()]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


def render(cfg, result) -> list[str]:
    from zarai.vision.results import table

    classes = list(cfg.classes)
    t = result["timings"]
    lines = [
        "# Gemma 4 as a zero-shot wheat disease classifier",
        "",
        "Generated by `python -m zarai.vision.vlm_zero_shot report` from configs/vision/vlm_zero_shot.yaml. "
        f"Model `{cfg.model.id}` at `{cfg.model.revision[:12]}` (4-bit AWQ of Google's "
        f"quantisation-aware-trained `{cfg.model.base}`; vision tower in bf16), served with vLLM, "
        "greedy decoding, thinking off. All numbers below were measured in this run.",
        "",
        "**Scoring.** Gemma's answer counts only if it is JSON with one of the six labels and a "
        "confidence in [0, 1]. `unsure` and invalid answers are scored as wrong (a sixth "
        '"no label" prediction), so macro-F1 and recall include them; their rates are given '
        "separately. Macro-F1 averages the classes present in the set, as for the vision models "
        "(experiments/wheat_v2_results.md). DINOv2-B is `dinov2_vit_b14` full fine-tune, strong "
        "augmentation, mean over its 3 seeds: the pooled run on the pooled test split and NWRD, and "
        "each LOSO fold's run on its held-out source. Its numbers are recomputed from the logits "
        "the runs logged.",
        "",
        "**In-context variants** put one labelled reference photo per class (5 images) from the "
        "pooled train split before the query. These references are excluded from every set they "
        "fall in (each is in its own source's LOSO test set). Because the references come from the "
        "pooled train split, the LOSO rows for in-context variants are not out-of-source: a "
        "reference may come from the held-out source.",
        "",
    ]
    refs = {}
    for name in result["variants"]:
        p = Path(cfg.output.dir) / f"{name}.references.json"
        if p.exists() and json.loads(p.read_text()):
            refs[name] = json.loads(p.read_text())
    if refs:
        first = next(iter(refs.values()))
        lines += [f"References (seed {cfg.references.seed}):", ""]
        lines += table(
            ["label", "source", "path"],
            [[r["label"], r["source"], f"`{r['path']}`"] for r in first],
        )
        lines.append("")

    # Main table, one block per test set.
    dino_t = t.get("dinov2", {})
    for set_name, info in result["sets"].items():
        present = info["classes"]
        title = {
            "pooled_test": "Pooled test split",
            "nwrd": "NWRD field tiles (all yellow_rust)",
        }.get(set_name, f"LOSO held-out source: {set_name.split(':', 1)[-1]}")
        lines += [f"## {title} ({info['n']} images)", ""]
        header = [
            "model",
            "n",
            "macro-F1 (95% CI)",
            *[f"recall {c}" for c in present],
            "unsure",
            "invalid JSON",
            "s/image batched",
            "s/image batch 1",
        ]
        body = []
        d = result["dino"][set_name]
        body.append(
            [
                "DINOv2-B (3 seeds)",
                info["n"],
                _f(_mean_seeds(d, "macro_f1")),
                *[_f(_mean_seeds(d, f"recall_{classes.index(c)}")) for c in present],
                "-",
                "-",
                _f(dino_t.get("batched_s"), 4),
                _f(dino_t.get("latency_median_s"), 4),
            ]
        )
        for name, v in result["variants"].items():
            s = v["sets"].get(set_name)
            if s is None:
                continue
            sc = s["scores"]
            lat = t.get(f"latency:{name}", {}).get("median_s")
            body.append(
                [
                    f"Gemma {name}",
                    s["n"],
                    f"{sc['macro_f1']['value']:.3f} ({sc['macro_f1']['low']:.3f}-{sc['macro_f1']['high']:.3f})",
                    *[_f(sc[f"recall_{classes.index(c)}"]["value"]) for c in present],
                    _f(s["unsure"], pct=True),
                    _f(s["invalid"], pct=True),
                    _f(v["batched_s"], 3),
                    _f(lat, 2),
                ]
            )
        lines += table(header, body)
        lines.append("")

    lines += _yellow_rust_section(cfg, result)

    # Agreement.
    lines += [
        "## Agreement between DINOv2-B and Gemma",
        "",
        "Per image, mean over DINOv2-B's 3 seeds. *agree* = same class. *disagree* = Gemma gave a "
        "different class. *Gemma no label* = Gemma said unsure or its answer was invalid. Accuracy "
        "is against the true label.",
        "",
    ]
    body = []
    for name, v in result["variants"].items():
        for set_name, s in v["sets"].items():
            a = s["agreement"]
            body.append(
                [
                    name,
                    set_name,
                    s["n"],
                    _f(_mean_seeds(a, "agree"), pct=True),
                    _f(_mean_seeds(a, "acc_agree")),
                    _f(_mean_seeds(a, "disagree"), pct=True),
                    _f(_mean_seeds(a, "dino_acc_disagree")),
                    _f(_mean_seeds(a, "gemma_acc_disagree")),
                    _f(_mean_seeds(a, "gemma_no_label"), pct=True),
                    _f(_mean_seeds(a, "dino_acc_gemma_no_label")),
                ]
            )
    lines += table(
        [
            "Gemma variant",
            "test set",
            "n",
            "agree",
            "accuracy when agree",
            "disagree",
            "DINOv2-B accuracy when disagree",
            "Gemma accuracy when disagree",
            "Gemma no label",
            "DINOv2-B accuracy when Gemma no label",
        ],
        body,
    )
    lines += [""]
    lines += _confusion_section(cfg, result)
    lines += ["## Run details", ""]
    lines += table(
        [
            "variant",
            "image budget",
            "references",
            "images run",
            "mean prompt tokens",
            "mean output tokens",
        ],
        [
            [
                n,
                v["budget"],
                v["shots"] * len(classes),
                v["n_images"],
                f"{v['mean_prompt_tokens']:.0f}",
                f"{v['mean_output_tokens']:.0f}",
            ]
            for n, v in result["variants"].items()
        ],
    )
    lines += [
        "",
        "Seconds per image: *batched* is wall time of the vLLM calls (64 images each) divided by "
        "images, over the whole run; *batch 1* is the median of one request at a time on the first "
        f"{cfg.timing.latency_images} pooled-test images. Image decoding is included for DINOv2-B "
        "and excluded from the batched Gemma figure (it is done in parallel threads). One "
        "RTX 4000 Ada (20 GB).",
        "",
        "Not run: in-context at the 1120 budget. At about 7.3k tokens per request its KV cache "
        "doesn't fit beside the 16.6 GiB of 4-bit weights on this card. Offloading 3 GB of weights "
        "to CPU memory would make room, but in a 128-image check it slowed zero_shot_280 from 0.53 "
        "to 2.39 s/image.",
        "",
    ]
    return lines


def _set_title(set_name: str) -> str:
    return {"pooled_test": "pooled test", "nwrd": "NWRD"}.get(set_name, set_name.split(":", 1)[-1])


def _share(k: int, n: int) -> str:
    return f"{k / n:.1%} ({k}/{n})" if n else "-"


def _range(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.0%}" if round(lo, 2) == round(hi, 2) else f"{lo:.0%}-{hi:.0%}"


def _yellow_rust_section(cfg, result) -> list[str]:
    """yellow_rust precision, how often other rusts are called yellow_rust, and what that means
    for the sets whose only rust is yellow_rust."""
    from zarai.vision.results import table

    classes = list(cfg.classes)
    y = classes.index("yellow_rust")
    rusts = [c for c in ("brown_rust", "black_rust") if c in classes]
    variants = result["variants"]
    lines = [
        "## Does Gemma call every rust yellow_rust?",
        "",
        "yellow_rust precision = share of Gemma's yellow_rust answers whose true label is "
        "yellow_rust. *All sets combined* counts every evaluated image once (5,898 minus the "
        "in-context references); the pooled test split is not added separately because its images "
        "are already in the LOSO sets.",
        "",
    ]
    set_names = list(result["sets"])
    body = []
    for name, v in variants.items():
        cells = []
        for set_name in set_names:
            s = v["sets"].get(set_name)
            cells.append(_share(*precision_of(s["confusion"], y)) if s else "-")
        body.append([name, _share(*precision_of(v["combined"]["confusion"], y)), *cells])
    lines += table(
        ["Gemma variant", "all sets combined", *[_set_title(s) for s in set_names]], body
    )
    lines += [
        "",
        "Share of brown_rust and black_rust images that Gemma labels yellow_rust:",
        "",
    ]
    checks = [(str(s), str(c)) for s, c in cfg.yellow_rust_checks]
    header = [
        "Gemma variant",
        *[f"{_set_title(s)} {c}" for s, c in checks],
        *[f"all sets {c}" for c in rusts],
        "all sets " + " + ".join(rusts),
    ]
    body = []
    combined_share, per_rust = {}, {c: [] for c in rusts}
    for name, v in variants.items():
        row = [name]
        for set_name, c in checks:
            s = v["sets"].get(set_name)
            row.append(_share(*called_as(s["confusion"], classes.index(c), y)) if s else "-")
        m = v["combined"]["confusion"]
        k_all = n_all = 0
        for c in rusts:
            k, n = called_as(m, classes.index(c), y)
            per_rust[c].append(k / n)
            row.append(_share(k, n))
            k_all, n_all = k_all + k, n_all + n
        combined_share[name] = k_all / n_all
        row.append(_share(k_all, n_all))
        body.append(row)
    lines += table(header, body)
    lines.append("")

    yellow_only = [
        _set_title(s)
        for s, info in result["sets"].items()
        if "yellow_rust" in info["classes"] and not set(rusts) & set(info["classes"])
    ]
    precisions = [precision_of(v["combined"]["confusion"], y) for v in variants.values()]
    precisions = [k / n for k, n in precisions if n]
    if variants and all(share > 0.5 for share in combined_share.values()):
        lines += [
            f"**Gemma labels most rust as yellow_rust.** Over all sets combined it calls "
            f"{_range(combined_share.values())} of {' and '.join(rusts)} images yellow_rust "
            + "("
            + ", ".join(f"{c} {_range(per_rust[c])}" for c in rusts)
            + f"), and only {_range(precisions)} of its yellow_rust answers are right. "
            f"{' and '.join(yellow_only)} contain no {' or '.join(rusts)}, so this costs nothing "
            f"there. **Gemma's {' and '.join(yellow_only)} scores are inflated**: its yellow_rust "
            "recall there is what answering yellow_rust for most rust gives on sets where the only "
            "rust is yellow_rust, so it is not evidence that Gemma recognises yellow rust or "
            "generalises better than DINOv2-B. (Its healthy and septoria recall on those sets is "
            "not affected.)",
            "",
        ]
    elif variants:
        lines += [
            f"Gemma does not label most rust as yellow_rust: over all sets combined it calls "
            f"{_range(combined_share.values())} of {' and '.join(rusts)} images yellow_rust.",
            "",
        ]
    return lines


def _confusion_section(cfg, result) -> list[str]:
    from zarai.vision.results import table

    classes = list(cfg.classes)
    columns = [*classes, "unsure", "invalid"]
    lines = [
        "## Confusion matrices",
        "",
        "Image counts. Rows are the true label (classes present in the set), columns Gemma's answer. "
        "In-context variants exclude the reference images.",
        "",
    ]
    for set_name in result["sets"]:
        for name, v in result["variants"].items():
            s = v["sets"].get(set_name)
            if s is None:
                continue
            lines += [f"### {_set_title(set_name)}, Gemma {name} ({s['n']} images)", ""]
            lines += table(
                ["true \\ Gemma", *columns],
                [
                    [c, *s["confusion"][classes.index(c)]]
                    for c in result["sets"][set_name]["classes"]
                ],
            )
            lines.append("")
    return lines


def report(cfg) -> Path:
    result = evaluate(cfg)
    out = Path(cfg.output.doc)
    out.write_text("\n".join(render(cfg, result)), encoding="utf-8")
    Path(cfg.output.dir, "summary.json").write_text(json.dumps(result, indent=1, default=str))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["run", "latency", "dinov2-timing", "report"])
    parser.add_argument("--variant")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args(argv)
    cfg = load_cfg(args.config)
    if args.command == "run":
        print(run(cfg, args.variant, args.limit))
    elif args.command == "latency":
        latency(cfg, args.variant)
    elif args.command == "dinov2-timing":
        dinov2_timing(cfg)
    else:
        print(report(cfg))


if __name__ == "__main__":
    main()
