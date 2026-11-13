"""Coarse evaluation: healthy / rust / septoria, from saved predictions.

DINOv2-B's fine prediction (argmax of its logits) and Gemma's answer are mapped to the coarse
class, so a rust called the wrong rust counts as correct. Three methods per test set:
DINOv2-B alone, Gemma alone (each variant) and the agree-rule (answer when the two give the same
coarse class, otherwise defer). Gemma's unsure and invalid answers are "no label": wrong for
macro-F1 and recall, and not answered for coverage. Nothing is fit.

    python -m zarai.vision.coarse_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from zarai.vision import vlm_zero_shot as vz

CONFIG = "configs/vision/coarse_eval.yaml"
DEFER = -1


def load_cfg(path: str = CONFIG):
    cfg = OmegaConf.load(path)
    return cfg, vz.load_cfg(cfg.base)


def coarse_map(fine_classes, groups, coarse_classes) -> np.ndarray:
    """Fine index -> coarse index; the fine no-label index maps to the coarse no-label index."""
    return np.array([coarse_classes.index(groups[c]) for c in fine_classes] + [len(coarse_classes)])


def outcome(labels, answer, n_coarse, coarse_classes) -> dict:
    """Coverage, accuracy on answered and per-class recall over all images (deferred = missed).
    `answer` uses DEFER for deferred and n_coarse for no label."""
    answered = (answer != DEFER) & (answer != n_coarse)
    correct = answer == labels
    out = {
        "n": int(len(labels)),
        "coverage": float(answered.mean()),
        "accuracy": float(correct[answered].mean()) if answered.any() else None,
    }
    for i, c in enumerate(coarse_classes):
        is_c = labels == i
        out[f"recall_{c}"] = float(correct[is_c].mean()) if is_c.any() else None
    # Rows = true class; columns = coarse classes, then "not answered" (deferred or no label).
    m = np.zeros((n_coarse, n_coarse + 1), dtype=int)
    np.add.at(m, (labels, np.where(answered, answer, n_coarse)), 1)
    out["confusion"] = m.tolist()
    return out


def mean_confusion(outcomes) -> np.ndarray:
    return np.mean([np.array(o["confusion"], dtype=float) for o in outcomes], axis=0)


def rust_stats(m: np.ndarray, coarse_classes) -> dict:
    """From a (mean) confusion matrix: rust answers vs rust images, rust precision, share of
    non-rust images called rust, healthy recall and where septoria images go."""
    r = coarse_classes.index("rust")
    k = len(coarse_classes)
    rust_answers, rust_images = m[:, r].sum(), m[r].sum()
    non_rust = [i for i in range(k) if i != r]
    non_rust_n = m[non_rust].sum()
    out = {
        "rust_answers": float(rust_answers),
        "rust_images": float(rust_images),
        "rust_ratio": float(rust_answers / rust_images) if rust_images else None,
        "rust_precision": float(m[r, r] / rust_answers) if rust_answers else None,
        "non_rust_called_rust": float(m[non_rust, r].sum() / non_rust_n) if non_rust_n else None,
    }
    h = coarse_classes.index("healthy")
    out["healthy_recall"] = float(m[h, h] / m[h].sum()) if m[h].sum() else None
    s = coarse_classes.index("septoria")
    sept = m[s].sum()
    for j, name in enumerate([*coarse_classes, "not_answered"]):
        out[f"septoria_to_{name}"] = float(m[s, j] / sept) if sept else None
    return out


def evaluate(cfg, base) -> dict:
    from zarai.common.io import read_csv

    fine = list(base.classes)
    coarse = list(cfg.coarse_classes)
    k = len(coarse)
    to_coarse = coarse_map(fine, cfg.groups, coarse)
    rows = read_csv(base.splits_csv)
    sets = vz.test_sets(rows)
    dino = vz.dinov2_predictions(base, sets)
    variants = {}
    for name in base.variants:
        records = vz.load_predictions(base, name)
        if records:
            ref_file = Path(base.output.dir) / f"{name}.references.json"
            refs = (
                {r["path"] for r in json.loads(ref_file.read_text())}
                if ref_file.exists()
                else set()
            )
            variants[name] = (records, refs)

    result = {"sets": {}, "variants": list(variants)}
    for set_name, set_rows in sets.items():
        labels = to_coarse[[fine.index(r["label"]) for r in set_rows]]
        methods: dict[str, list] = {"dino": []}
        for preds in dino[set_name].values():
            p = to_coarse[preds]
            methods["dino"].append(
                {**outcome(labels, p, k, coarse), "scores": vz.scores(labels, p, k)}
            )
        for name, (records, refs) in variants.items():
            keep = [i for i, r in enumerate(set_rows) if r["path"] not in refs]
            y = labels[keep]
            g = to_coarse[[vz.gemma_index(records[set_rows[i]["path"]], fine) for i in keep]]
            methods[f"gemma:{name}"] = [{**outcome(y, g, k, coarse), "scores": vz.scores(y, g, k)}]
            methods[f"rule:{name}"] = []
            for preds in dino[set_name].values():
                d = to_coarse[preds][keep]
                answer = np.where((d == g) & (g < k), d, DEFER)
                methods[f"rule:{name}"].append(outcome(y, answer, k, coarse))
        result["sets"][set_name] = {
            "n": len(set_rows),
            "classes": [c for c in coarse if c in {cfg.groups[r["label"]] for r in set_rows}],
            "methods": methods,
        }
    return result


def _mean(outcomes, key):
    vals = []
    for o in outcomes:
        v = o["scores"][key]["value"] if key == "macro_f1" else o[key]
        if v is not None and not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def render(cfg, base, result) -> list[str]:
    from zarai.vision.results import table

    coarse = list(cfg.coarse_classes)
    groups = {}
    for fine_class, c in cfg.groups.items():
        groups.setdefault(c, []).append(fine_class)
    lines = [
        "# Coarse evaluation: healthy / rust / septoria",
        "",
        "Generated by `python -m zarai.vision.coarse_eval` from configs/vision/coarse_eval.yaml, "
        "from the saved predictions of experiments/vlm_zero_shot.md (Gemma answers, DINOv2-B "
        "logits). Nothing is run or fit. All numbers are measured.",
        "",
        "**Coarse classes.** "
        + "; ".join(f"*{c}* = {', '.join(groups[c])}" for c in coarse)
        + ". A rust called the wrong rust counts as correct.",
        "",
        "**Methods.** *DINOv2-B alone*: its fine argmax mapped to the coarse class, mean over 3 "
        "seeds (pooled run for pooled test and NWRD, each LOSO fold's run for its held-out "
        "source). *Gemma alone*: its answer mapped to the coarse class; unsure and invalid "
        "answers are not answered. *Agree-rule*: answers when DINOv2-B and Gemma give the same "
        "coarse class, otherwise defers (Gemma unsure or invalid also defers); mean over "
        "DINOv2-B's 3 seeds.",
        "",
        "**Columns.** *coverage* = share answered. *accuracy* = on answered images. *macro-F1* "
        "averages the coarse classes present, with unanswered images scored as wrong (not given "
        "for the agree-rule, which defers by design). *recall* = correct / all images of that "
        "class, so deferred and unsure count as missed. In-context variants leave out their 5 "
        "reference images.",
        "",
    ]
    names = METHOD_NAMES
    for set_name, info in result["sets"].items():
        title = {"pooled_test": "Pooled test split", "nwrd": "NWRD field tiles"}.get(set_name)
        title = title or f"LOSO held-out source: {set_name.split(':', 1)[1]}"
        lines += [f"## {title} ({info['n']} images)", ""]
        present = info["classes"]
        header = ["method", "Gemma variant", "n", "coverage", "accuracy", "macro-F1"]
        header += [f"recall {c}" for c in present]
        body = []
        methods = info["methods"]
        keys = ["dino"] + [f"{m}:{v}" for v in result["variants"] for m in ("gemma", "rule")]
        for key in keys:
            outs = methods.get(key)
            if not outs:
                continue
            kind, _, variant = key.partition(":")
            body.append(
                [
                    names[kind],
                    variant or "-",
                    outs[0]["n"],
                    vz._f(_mean(outs, "coverage"), pct=True),
                    vz._f(_mean(outs, "accuracy")),
                    vz._f(_mean(outs, "macro_f1")) if kind != "rule" else "-",
                    *[vz._f(_mean(outs, f"recall_{c}")) for c in present],
                ]
            )
        lines += table(header, body)
        lines.append("")
    lines += _rust_section(cfg, result)
    lines += _confusion_section(cfg, result)
    return lines


METHOD_NAMES = {"dino": "DINOv2-B alone", "gemma": "Gemma alone", "rule": "agree-rule"}


def _method_keys(result) -> list[str]:
    return ["dino"] + [f"{m}:{v}" for v in result["variants"] for m in ("gemma", "rule")]


def _title(set_name: str) -> str:
    return {"pooled_test": "pooled test", "nwrd": "NWRD"}.get(set_name) or set_name.split(":", 1)[1]


def _count(x: float) -> str:
    x = round(float(x), 6)
    return f"{x:.0f}" if x.is_integer() else f"{x:.1f}"


def _rust_section(cfg, result) -> list[str]:
    from zarai.vision.results import table

    coarse = list(cfg.coarse_classes)
    lines = [
        "## Is rust over-predicted?",
        "",
        "From the confusion matrices below (DINOv2-B and agree-rule: mean counts over the 3 seeds). "
        "*rust answers / rust images*: how often the method says rust, relative to how many rust "
        "images there are; above 1.00 means it says rust more often than rust occurs. *rust "
        "precision*: share of rust answers that are rust. *non-rust called rust*: share of healthy "
        "and septoria images answered rust. *septoria ->*: where septoria images go (shares of all "
        "septoria images; *not answered* = deferred, unsure or invalid).",
        "",
    ]
    header = [
        "test set",
        "method",
        "Gemma variant",
        "rust answers / rust images",
        "rust precision",
        "non-rust called rust",
        "healthy recall",
        "septoria -> healthy",
        "septoria -> rust",
        "septoria -> septoria",
        "septoria -> not answered",
    ]
    body = []
    over: dict[str, list[str]] = {}
    leak: dict[str, list[str]] = {}
    for set_name, info in result["sets"].items():
        for key in _method_keys(result):
            outs = info["methods"].get(key)
            if not outs:
                continue
            kind, _, variant = key.partition(":")
            st = rust_stats(mean_confusion(outs), coarse)
            who = f"{METHOD_NAMES[kind]}{' ' + variant if variant else ''}"
            if st["rust_ratio"] is not None and st["rust_ratio"] > 1:
                over.setdefault(_title(set_name), []).append(
                    f"{who} {st['rust_ratio']:.2f} ({_count(st['rust_answers'])} rust answers, "
                    f"{_count(st['rust_images'])} rust images; precision "
                    f"{st['rust_precision']:.3f})"
                )
            to_rust, to_sept = st["septoria_to_rust"], st["septoria_to_septoria"]
            if to_rust is not None and to_rust > to_sept:
                leak.setdefault(_title(set_name), []).append(
                    f"{who} (rust {to_rust:.1%}, septoria {to_sept:.1%})"
                )
            body.append(
                [
                    _title(set_name),
                    METHOD_NAMES[kind],
                    variant or "-",
                    f"{_count(st['rust_answers'])} / {_count(st['rust_images'])}"
                    + (f" = {st['rust_ratio']:.2f}" if st["rust_ratio"] is not None else ""),
                    vz._f(st["rust_precision"]),
                    vz._f(st["non_rust_called_rust"], pct=True),
                    vz._f(st["healthy_recall"]),
                    *[
                        vz._f(st[f"septoria_to_{c}"], pct=True)
                        for c in ("healthy", "rust", "septoria", "not_answered")
                    ],
                ]
            )
    lines += table(header, body)
    lines.append("")
    sets = [_title(s) for s in result["sets"]]
    if over:
        lines += [
            "**Rust is over-predicted** (more rust answers than rust images) on "
            + ", ".join(over)
            + ":",
            "",
        ]
        lines += [f"- {name}: " + "; ".join(items) for name, items in over.items()]
        clean = [name for name in sets if name not in over]
        lines += [
            "",
            f"It is not over-predicted by this measure on {', '.join(clean)}." if clean else "",
            "",
        ]
    else:
        lines += [
            "**Rust is not over-predicted anywhere:** every method gives fewer rust answers than "
            "there are rust images on every test set.",
            "",
        ]
    if leak:
        lines += [
            "**Septoria images answered rust more often than septoria:**",
            "",
            *[f"- {name}: " + "; ".join(items) for name, items in leak.items()],
            "",
        ]
    return lines


def _confusion_section(cfg, result) -> list[str]:
    from zarai.vision.results import table

    coarse = list(cfg.coarse_classes)
    lines = [
        "## Coarse confusion matrices",
        "",
        "Rows = true class (present in the set), columns = answer. *not answered* = deferred "
        "(agree-rule) or unsure / invalid (Gemma). DINOv2-B and agree-rule are mean counts over "
        "DINOv2-B's 3 seeds.",
        "",
    ]
    for set_name, info in result["sets"].items():
        for key in _method_keys(result):
            outs = info["methods"].get(key)
            if not outs:
                continue
            kind, _, variant = key.partition(":")
            m = mean_confusion(outs)
            lines += [
                f"### {_title(set_name)}, {METHOD_NAMES[kind]}"
                f"{' ' + variant if variant else ''} ({outs[0]['n']} images)",
                "",
            ]
            lines += table(
                ["true \\ answer", *coarse, "not answered"],
                [[c, *[_count(x) for x in m[coarse.index(c)]]] for c in info["classes"]],
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
