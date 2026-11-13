"""Hybrid decision rules for DINOv2-B and Gemma 4, on the vlm_zero_shot test sets.

(a) Rule: answer when DINOv2-B and Gemma give the same class, otherwise defer (ask a clarifying
    question). Gemma saying unsure, or an invalid answer, is a disagreement.
(b) Stacker: logistic regression on DINOv2-B's temperature-calibrated probabilities plus Gemma's
    answer (one-hot, with unsure and invalid as their own columns) and its stated confidence.

Nothing is fit on a test set. For held-out source S the DINOv2-B run is LOSO fold S, and its
temperature, the stacker and the deferral thresholds are fit on that run's validation split,
which holds only the other sources. For the pooled test split and NWRD the run is the pooled one,
fit on the pooled validation split (NWRD is never in it). Gemma isn't trained; its answers are the
ones vlm_zero_shot saved. Deferral thresholds for DINOv2-B and the stacker are set on the val
split so they defer as often there as rule (a) does, which makes the three comparable.

    python -m zarai.vision.vlm_hybrid
"""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from zarai.vision import vlm_zero_shot as vz

CONFIG = "configs/vision/vlm_hybrid.yaml"
DEFER = -1


def load_cfg(path: str = CONFIG):
    cfg = OmegaConf.load(path)
    return cfg, vz.load_cfg(cfg.base)


def fold_of(set_name: str) -> tuple[str, str, str]:
    """(DINOv2-B fold, split column whose 'val' rows train the hybrid, logits artifact split)."""
    if set_name == "pooled_test":
        return "pooled", "split_pooled", "heldout"
    if set_name == "nwrd":
        return "pooled", "split_pooled", "nwrd"
    source = set_name.split(":", 1)[1]
    return source, f"split_loso_{source}", "heldout"


def gemma_inputs(records, paths, classes) -> tuple[np.ndarray, np.ndarray]:
    """Gemma's answer column (class, then unsure, then invalid) and stated confidence."""
    cols = np.array([vz.answer_column(records[p], classes) for p in paths])
    conf = np.array(
        [float(records[p]["confidence"]) if records[p]["valid"] else 0.0 for p in paths]
    )
    return cols, conf


def features(probs, cols, conf, n_classes) -> np.ndarray:
    onehot = np.eye(n_classes + 2)[cols]
    return np.hstack([probs, onehot, conf[:, None]])


def rule(dino_pred, gemma_cols) -> np.ndarray:
    """DINOv2-B's class where Gemma gives the same class, otherwise DEFER."""
    return np.where(dino_pred == gemma_cols, dino_pred, DEFER)


def with_threshold(probs, threshold) -> np.ndarray:
    return np.where(probs.max(axis=1) >= threshold, probs.argmax(axis=1), DEFER)


def fit_stacker(x, y, n_classes, stacker_cfg):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=float(stacker_cfg.C), max_iter=int(stacker_cfg.max_iter))
    model.fit(x, y)

    def predict(x_new) -> np.ndarray:
        out = np.zeros((len(x_new), n_classes))
        out[:, model.classes_] = model.predict_proba(x_new)
        return out

    return predict


def out_of_fold(x, y, n_classes, stacker_cfg) -> np.ndarray:
    """Stacker probabilities on its own training rows, each from a model that didn't see it."""
    from sklearn.model_selection import StratifiedKFold

    folds = StratifiedKFold(
        int(stacker_cfg.cv_folds), shuffle=True, random_state=int(stacker_cfg.seed)
    )
    out = np.zeros((len(y), n_classes))
    for train, held in folds.split(x, y):
        out[held] = fit_stacker(x[train], y[train], n_classes, stacker_cfg)(x[held])
    return out


def outcome(labels, answer, classes, recall_classes) -> dict:
    """Coverage, accuracy on answered images and, per class, recall on answered / all images."""
    answered = answer != DEFER
    correct = answer == labels
    out = {
        "n": int(len(labels)),
        "coverage": float(answered.mean()),
        "accuracy": float(correct[answered].mean()) if answered.any() else None,
    }
    for c in recall_classes:
        is_c = labels == classes.index(c)
        out[f"{c}_n"] = int(is_c.sum())
        out[f"{c}_recall_answered"] = (
            float(correct[is_c & answered].mean()) if (is_c & answered).any() else None
        )
        out[f"{c}_recall_all"] = float(correct[is_c].mean()) if is_c.any() else None
    return out


def evaluate(cfg, base) -> dict:
    from zarai.common.io import read_csv
    from zarai.vision.calibrate import apply_temperature, fit_temperature, threshold_for_coverage
    from zarai.vision.tracking import artifact_path, set_tracking_uri

    classes = list(base.classes)
    n = len(classes)
    recall_classes = list(cfg.recall_classes)
    rows = read_csv(base.splits_csv)
    sets = vz.test_sets(rows)
    set_tracking_uri(base.dinov2.tracking_uri)
    runs = vz.dinov2_runs(base)
    variants = {}
    for name in base.variants:
        records = vz.load_predictions(base, name)
        if not records:
            continue
        ref_file = Path(base.output.dir) / f"{name}.references.json"
        refs = {r["path"] for r in json.loads(ref_file.read_text())} if ref_file.exists() else set()
        variants[name] = (records, refs)

    def load(run_id, artifact, set_rows):
        data = np.load(artifact_path(run_id, artifact))
        labels = np.array([classes.index(r["label"]) for r in set_rows])
        if not np.array_equal(data["labels"], labels):
            raise ValueError(f"{run_id}/{artifact} is not in CSV order")
        return data["logits"]

    result = {"sets": {}, "fits": {}}
    for set_name, test_rows in sets.items():
        fold, column, split = fold_of(set_name)
        val_rows = [r for r in rows if r[column] == "val"]
        if {r["source"] for r in val_rows} & {r["source"] for r in test_rows}:
            if set_name != "pooled_test":
                raise AssertionError(f"{set_name}: val split shares a source with the test set")
        if any(r["source"] == "nwrd" for r in val_rows):
            raise AssertionError("NWRD in a val split")
        per_method: dict[str, list] = {}
        fits = []
        for seed, run_id in sorted(runs[fold].items()):
            val_logits = load(run_id, "predictions/val_logits.npz", val_rows)
            test_logits = load(run_id, f"eval/{split}/{split}_logits.npz", test_rows)
            temperature = fit_temperature(
                val_logits, np.array([classes.index(r["label"]) for r in val_rows])
            )
            for name, (records, refs) in variants.items():
                v_keep = [i for i, r in enumerate(val_rows) if r["path"] not in refs]
                t_keep = [i for i, r in enumerate(test_rows) if r["path"] not in refs]
                v_paths = [val_rows[i]["path"] for i in v_keep]
                t_paths = [test_rows[i]["path"] for i in t_keep]
                v_y = np.array([classes.index(val_rows[i]["label"]) for i in v_keep])
                t_y = np.array([classes.index(test_rows[i]["label"]) for i in t_keep])
                v_p = apply_temperature(val_logits[v_keep], temperature)
                t_p = apply_temperature(test_logits[t_keep], temperature)
                v_cols, v_conf = gemma_inputs(records, v_paths, classes)
                t_cols, t_conf = gemma_inputs(records, t_paths, classes)

                # Coverage of rule (a) on val: the target for the thresholded methods.
                target = float((rule(v_p.argmax(1), v_cols) != DEFER).mean())
                v_x = features(v_p, v_cols, v_conf, n)
                t_x = features(t_p, t_cols, t_conf, n)
                stacker = fit_stacker(v_x, v_y, n, cfg.stacker)
                t_s = stacker(t_x)
                s_threshold = threshold_for_coverage(out_of_fold(v_x, v_y, n, cfg.stacker), target)
                d_threshold = threshold_for_coverage(v_p, target)

                answers = {
                    "dino": t_p.argmax(1),
                    f"dino_defer:{name}": with_threshold(t_p, d_threshold),
                    f"gemma:{name}": np.where(t_cols < n, t_cols, DEFER),
                    f"rule:{name}": rule(t_p.argmax(1), t_cols),
                    f"stacker:{name}": t_s.argmax(1),
                    f"stacker_defer:{name}": with_threshold(t_s, s_threshold),
                }
                for key, answer in answers.items():
                    if key == "dino" and name != next(iter(variants)):
                        continue  # DINOv2-B alone doesn't depend on the variant
                    per_method.setdefault(key, []).append(
                        outcome(t_y, answer, classes, recall_classes)
                    )
                fits.append(
                    {
                        "seed": seed,
                        "variant": name,
                        "val_n": len(v_keep),
                        "val_sources": sorted({r["source"] for r in val_rows}),
                        "val_counts": {c: int((v_y == classes.index(c)).sum()) for c in classes},
                        "temperature": temperature,
                        "target_coverage": target,
                        "dino_threshold": d_threshold,
                        "stacker_threshold": s_threshold,
                    }
                )
        result["sets"][set_name] = {
            "n": len(test_rows),
            "classes": sorted({r["label"] for r in test_rows}),
            # Gemma alone is the same for every seed; keep one copy.
            "methods": {k: (v[:1] if k.startswith("gemma:") else v) for k, v in per_method.items()},
        }
        result["fits"][set_name] = fits
    result["variants"] = list(variants)
    return result


def _mean(outcomes, key):
    vals = [o[key] for o in outcomes if o[key] is not None]
    return float(np.mean(vals)) if vals else None


def render(cfg, base, result) -> list[str]:
    from zarai.vision.results import table

    rc = list(cfg.recall_classes)
    lines = [
        "# Hybrid of DINOv2-B and Gemma 4",
        "",
        "Generated by `python -m zarai.vision.vlm_hybrid` from configs/vision/vlm_hybrid.yaml, "
        "using the Gemma answers of experiments/vlm_zero_shot.md and the logits the DINOv2-B runs "
        "logged. All numbers are measured; DINOv2-B rows and every hybrid are means over its 3 "
        "seeds.",
        "",
        "**Methods.**",
        "",
        "- *DINOv2-B alone*: argmax, answers every image.",
        "- *Gemma alone*: answers unless it says unsure or the answer is invalid.",
        "- *(a) rule*: answers DINOv2-B's class when Gemma gives the same class; otherwise "
        "(different class, unsure or invalid) it defers, i.e. would ask a clarifying question.",
        "- *(b) stacker*: multinomial logistic regression (C = "
        f"{cfg.stacker.C}, not tuned) on DINOv2-B's temperature-calibrated probabilities (5), "
        "Gemma's answer one-hot (5 classes, unsure, invalid) and Gemma's stated confidence. "
        "Answers every image.",
        "- *(b) stacker, deferring* and *DINOv2-B, deferring*: answer only when their top "
        "probability is above a threshold. The threshold is set on the val split so that they "
        "answer as often there as rule (a) does with the same Gemma variant (the stacker's val "
        f"probabilities for this are out-of-fold, {cfg.stacker.cv_folds}-fold). On the test set "
        "their coverage can differ from the rule's.",
        "",
        "**What is fit where.** Nothing is fit on a test set. For a held-out source the DINOv2-B "
        "run is that LOSO fold's, and its temperature, the stacker and the thresholds are fit on "
        "that run's validation split, which contains only the other sources. Pooled test and NWRD "
        "use the pooled run and its validation split (NWRD is never in it; the pooled split's val "
        "and test hold the same sources). Gemma itself is not trained. Caveat carried over from "
        "the zero-shot report: in-context references are drawn from the pooled train split, and "
        "4 of the 5 come from kaggle and 1 from cgiar_wheat_rust, so for those two held-out "
        "sources the in-context variant has seen one or more photos from the held-out source in "
        "its prompt. References are left out of every val and test set here.",
        "",
        "**Columns.** *coverage* = share of images answered. *accuracy* = on answered images. "
        "Recall *answered* = correct / answered images of that class; *all* = correct / all "
        "images of that class (deferred counts as not found).",
        "",
    ]
    names = {
        "dino": "DINOv2-B alone",
        "gemma": "Gemma alone",
        "dino_defer": "DINOv2-B, deferring",
        "rule": "(a) rule: agree or defer",
        "stacker": "(b) stacker",
        "stacker_defer": "(b) stacker, deferring",
    }
    order = ["gemma", "dino_defer", "rule", "stacker", "stacker_defer"]
    for set_name, info in result["sets"].items():
        title = {"pooled_test": "Pooled test split", "nwrd": "NWRD field tiles"}.get(set_name)
        title = title or f"LOSO held-out source: {set_name.split(':', 1)[1]}"
        lines += [f"## {title} ({info['n']} images)", ""]
        present = [c for c in rc if c in info["classes"]]
        header = ["method", "Gemma variant", "n", "coverage", "accuracy"]
        for c in present:
            header += [f"{c} recall answered", f"{c} recall all"]
        body = []
        methods = info["methods"]
        keys = ["dino"] + [f"{m}:{v}" for v in result["variants"] for m in order]
        for key in keys:
            if key not in methods:
                continue
            outs = methods[key]
            kind, _, variant = key.partition(":")
            row = [
                names[kind],
                variant or "-",
                outs[0]["n"],
                vz._f(_mean(outs, "coverage"), pct=True),
                vz._f(_mean(outs, "accuracy")),
            ]
            for c in present:
                row += [
                    vz._f(_mean(outs, f"{c}_recall_answered")),
                    vz._f(_mean(outs, f"{c}_recall_all")),
                ]
            body.append(row)
        lines += table(header, body)
        lines.append("")

    lines += [
        "## Fits",
        "",
        "Per test set and Gemma variant, mean over the 3 DINOv2-B seeds: the val split the hybrid "
        "was fit on, rule (a)'s coverage there (the target for the deferring methods) and the "
        "thresholds that gives.",
        "",
    ]
    body = []
    for set_name, fits in result["fits"].items():
        for variant in result["variants"]:
            f = [x for x in fits if x["variant"] == variant]
            body.append(
                [
                    set_name,
                    variant,
                    f[0]["val_n"],
                    ", ".join(f[0]["val_sources"]),
                    ", ".join(f"{c} {k}" for c, k in f[0]["val_counts"].items()),
                    vz._f(float(np.mean([x["temperature"] for x in f])), 2),
                    vz._f(float(np.mean([x["target_coverage"] for x in f])), pct=True),
                    vz._f(float(np.mean([x["dino_threshold"] for x in f])), 3),
                    vz._f(float(np.mean([x["stacker_threshold"] for x in f])), 3),
                ]
            )
    lines += table(
        [
            "test set",
            "Gemma variant",
            "val images",
            "val sources",
            "val images per class",
            "DINOv2-B temperature",
            "rule (a) coverage on val",
            "DINOv2-B threshold",
            "stacker threshold",
        ],
        body,
    )
    lines.append("")
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args(argv)
    cfg, base = load_cfg(args.config)
    result = evaluate(cfg, base)
    Path(cfg.output.summary).write_text(json.dumps(result, indent=1, default=str))
    Path(cfg.output.doc).write_text("\n".join(render(cfg, base, result)), encoding="utf-8")
    print(cfg.output.doc)


if __name__ == "__main__":
    main()
