"""Score the frozen wheat vision model on external datasets it never saw.

Each dataset has a label_map onto our 15 classes. Anything unmapped is skipped and reported, and a
missing dataset is reported rather than treated as an error. Results go to the MLflow experiment
wheat_v1_cross_domain and to experiments/wheat_v1_cross_domain.md.
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score

from zarai.common.io import read_csv
from zarai.common.logging import get_logger
from zarai.vision.results import replace_section, table

ROOT = Path("data/raw/images_public")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TILE_SIZE = 384
MAX_HAMMING_DISTANCE = 6
EXPERIMENT = "wheat_v1_cross_domain"
TBD = "[TBD]"
# A plain client name: Mendeley refuses Python's default User-Agent with 403.
USER_AGENT = "zarai-dataset-check/0.1"
KAGGLE_TOKEN_FILE = Path.home() / ".kaggle" / "access_token"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    label_map: dict[str, str]
    page: str
    licence: str
    layout: str
    # (url, filename, hash algorithm, hash) for files that can be fetched without an account.
    files: tuple[tuple[str, str, str, str], ...] = ()
    manual: str = ""
    kaggle: str = ""
    only_under: str | None = None
    uses_masks: bool = False
    notes: str = ""
    expected_images: dict[str, int] = field(default_factory=dict)


EXTERNAL_DATASETS = {
    d.name: d
    for d in (
        ExternalDataset(
            name="mendeley_wheat_leaf_ethiopia",
            label_map={
                "healthy": "healthy",
                "healthy_wheat": "healthy",
                "healthy_wheat_leaf": "healthy",
                "healthy_leaf": "healthy",
                "stripe_rust": "yellow_rust",
                "strip_rust": "yellow_rust",
                "stripe": "yellow_rust",
                "septoria": "septoria",
                "septoria_disease": "septoria",
            },
            page="https://data.mendeley.com/datasets/wgd66f8n6h/1",
            licence="CC BY 4.0",
            layout="<any depth>/{healthy,stripe_rust,septoria}/*.jpg (a train/val/test level is fine)",
            files=(
                (
                    "https://data.mendeley.com/public-files/datasets/wgd66f8n6h/files/"
                    "4ec9266d-d7c1-4fd5-85c9-e5624c09a847/file_downloaded",
                    "wheat_leaf.rar",
                    "sha256",
                    "23653dd88f9f900315a9a0c88a163c891b3bdd00cc89b36d98cc5b3fcc1d7e94",
                ),
            ),
            notes="A .rar archive: it is unpacked if unrar is installed, otherwise use 7z x wheat_leaf.rar.",
            expected_images={"healthy": 102, "yellow_rust": 208, "septoria": 97},
        ),
        ExternalDataset(
            name="mendeley_iari_leaf_rust",
            label_map={
                "diseased": "brown_rust",
                "disease": "brown_rust",
                "control": "healthy",
            },
            page="https://data.mendeley.com/datasets/th422bg4yd/1",
            licence="CC BY 4.0",
            layout="<any depth>/{diseased,control}/*.jpg (a train/val/test level is fine)",
            files=(
                (
                    "https://data.mendeley.com/public-files/datasets/th422bg4yd/files/"
                    "e8dc9987-db3b-4b0e-9363-613d599bf0bd/file_downloaded",
                    "WheatLeafRust.zip",
                    "sha256",
                    "1ad34f9e57d3510d81646cb027aaf5e72db0978d2278b25ca91819a6df9409bb",
                ),
            ),
            notes="Only WheatLeafRust.zip is needed, not Ndeficient.zip. Backgrounds are masked.",
            expected_images={"brown_rust": 368, "healthy": 491},
        ),
        ExternalDataset(
            name="nwrd",
            label_map={"masked": "yellow_rust"},
            page="https://zenodo.org/records/21672928",
            licence="CC BY 4.0",
            layout="{train,test}/images/<id>.jpg with {train,test}/masks/<id>.png",
            files=(
                (
                    "https://zenodo.org/api/records/21672928/files/NWRD.zip/content",
                    "NWRD.zip",
                    "md5",
                    "7b9193efebe1ed8c60b10c9f05d11e14",
                ),
            ),
            uses_masks=True,
            notes="Images and tiles with an empty mask are skipped, not treated as healthy.",
        ),
        ExternalDataset(
            name="cgiar_wheat_rust",
            label_map={
                "healthy_wheat": "healthy",
                "stem_rust": "black_rust",
                "leaf_rust": "brown_rust",
            },
            page="https://www.kaggle.com/datasets/shadabhussain/cgiar-computer-vision-for-crop-disease",
            licence="Listed as Unknown on Kaggle: check it, and the Zindi rules, before reuse",
            layout="<any depth>/train/{healthy_wheat,stem_rust,leaf_rust}/*",
            manual=(
                "kaggle datasets download shadabhussain/cgiar-computer-vision-for-crop-disease "
                "(needs a Kaggle API token), or Train.zip from the competition page, which needs a "
                "login: https://zindi.africa/competitions/"
                "iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/data"
            ),
            kaggle="shadabhussain/cgiar-computer-vision-for-crop-disease",
            only_under="train",
            notes="Part of this data comes from Google Images, so it may overlap with training data.",
        ),
    )
}


def normalise_folder(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith((".", "__MACOSX")) for part in path.relative_to(root).parts)
    )


def collect_folders(root: Path, spec: ExternalDataset) -> tuple[list[dict], Counter]:
    """Label images by their nearest parent folder in the label map. Returns (samples, skipped)."""
    samples, skipped = [], Counter()
    for path in _image_files(root):
        parents = path.relative_to(root).parts[:-1]
        normalised = [normalise_folder(part) for part in parents]
        if spec.only_under and spec.only_under not in normalised:
            skipped[f"outside {spec.only_under}/"] += 1
            continue
        label = next((spec.label_map[n] for n in reversed(normalised) if n in spec.label_map), None)
        if label is None:
            skipped[f"folder {parents[-1] if parents else '.'}"] += 1
            continue
        samples.append({"path": str(path), "label": label, "box": None, "image": str(path)})
    return samples, skipped


def tile_boxes(width: int, height: int, size: int = TILE_SIZE) -> list[tuple[int, int, int, int]]:
    """Non-overlapping full tiles. A leftover strip narrower than a tile is dropped."""
    return [
        (left, top, left + size, top + size)
        for top in range(0, height - size + 1, size)
        for left in range(0, width - size + 1, size)
    ]


def collect_masked(root: Path) -> tuple[list[dict], list[dict], Counter, Counter]:
    """NWRD: label whole images and 384 px tiles by their own mask.

    Returns (image samples, tile samples, skipped images, skipped tiles). Images aren't rotated by
    EXIF here, because the masks line up with the stored pixels.
    """
    from PIL import Image

    images, tiles, skip_images, skip_tiles = [], [], Counter(), Counter()
    label = EXTERNAL_DATASETS["nwrd"].label_map["masked"]
    for path in _image_files(root):
        if path.parent.name != "images":
            continue
        mask_path = path.parent.parent / "masks" / f"{path.stem}.png"
        if not mask_path.exists():
            skip_images["no mask"] += 1
            continue
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        with Image.open(path) as image:
            size = image.size
        if (mask.shape[1], mask.shape[0]) != size:
            skip_images["mask size differs"] += 1
            continue
        if mask.any():
            images.append({"path": str(path), "label": label, "box": None, "image": str(path)})
        else:
            skip_images["empty mask"] += 1
        for box in tile_boxes(*size):
            left, top, right, bottom = box
            if mask[top:bottom, left:right].any():
                tiles.append({"path": str(path), "label": label, "box": box, "image": str(path)})
            else:
                skip_tiles["empty mask"] += 1
    return images, tiles, skip_images, skip_tiles


def check_dataset(root: Path, spec: ExternalDataset) -> dict:
    """Describe one dataset's state. Never raises for a missing or unexpected dataset."""
    folder = root / spec.name
    if not folder.is_dir():
        return {"name": spec.name, "status": "missing", "counts": {}, "skipped": {}}
    if spec.uses_masks:
        images, tiles, skipped, skipped_tiles = collect_masked(folder)
        counts = Counter(s["label"] for s in images)
        result = {"tiles": len(tiles), "skipped_tiles": dict(skipped_tiles)}
    else:
        samples, skipped = collect_folders(folder, spec)
        counts = Counter(s["label"] for s in samples)
        result = {}
    status = "ready" if counts else "empty or unrecognised layout"
    return {
        "name": spec.name,
        "status": status,
        "counts": dict(counts),
        "skipped": dict(skipped),
        **result,
    }


def _verify(path: Path, algorithm: str, expected: str) -> bool:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def safe_extract(archive: Path, target: Path) -> None:
    """Unpack a zip, refusing any entry that would land outside the target folder."""
    resolved = target.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not (resolved / member).resolve().is_relative_to(resolved):
                raise ValueError(f"{archive}: unsafe path in archive: {member}")
        z.extractall(target)


def kaggle_token() -> str | None:
    """A Kaggle API token from KAGGLE_API_TOKEN or ~/.kaggle/access_token, if either is set."""
    import os

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()
    if KAGGLE_TOKEN_FILE.exists():
        return KAGGLE_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def _fetch(url: str, path: Path, attempts: int = 6) -> None:
    """Download to a .part file, resuming after a stall, then rename it into place.

    The .part file is kept if every attempt fails, so the next run resumes it.
    """
    import time

    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, attempts + 1):
        done = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores the range sends everything again, so start over.
                mode = "ab" if done and response.status == 206 else "wb"
                with open(partial, mode) as f:
                    shutil.copyfileobj(response, f, length=2**20)
            partial.replace(path)
            return
        except (OSError, TimeoutError) as error:
            if attempt == attempts:
                raise
            log.warning("%s: attempt %d failed (%s), resuming", path.name, attempt, error)
            time.sleep(min(30, 2**attempt))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def kaggle_download_url(slug: str, token: str) -> str:
    """Ask Kaggle for a dataset's signed download link.

    The token goes only to kaggle.com. urllib would otherwise forward the Authorization header
    to the storage host when it follows the redirect.
    """
    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{slug}",
        headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(request, timeout=60)
    except urllib.error.HTTPError as response:
        if response.code in (301, 302, 303, 307, 308) and response.headers.get("Location"):
            return response.headers["Location"]
        raise
    raise ValueError(f"Kaggle did not redirect to a download for {slug}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack(archive: Path, folder: Path) -> str:
    """Unpack a zip, or a rar with unrar if it is installed. Returns a message."""
    folder.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        safe_extract(archive, folder)
        return f"unpacked {archive.name} into {folder}"
    if archive.suffix == ".rar":
        unrar = shutil.which("unrar")
        if unrar is None:
            return f"saved {archive}. Install unrar or 7z and unpack it into {folder}"
        import subprocess

        # unrar refuses entries with absolute or parent paths unless told otherwise.
        subprocess.run([unrar, "x", "-y", "-o+", "-idq", str(archive), f"{folder}/"], check=True)
        return f"unpacked {archive.name} into {folder}"
    return f"saved {archive}. Unpack it into {folder}"


def download(root: Path, spec: ExternalDataset) -> str:
    """Fetch, check and unpack a dataset's files. Returns a message for the user."""
    folder = root / spec.name
    holding = root / ".downloads" / spec.name
    holding.mkdir(parents=True, exist_ok=True)
    messages = []
    for url, filename, algorithm, expected in spec.files:
        path = holding / filename
        if not (path.exists() and _verify(path, algorithm, expected)):
            _fetch(url, path)
            if not _verify(path, algorithm, expected):
                path.unlink()
                raise ValueError(f"{filename}: {algorithm} does not match, so it was deleted")
        messages.append(f"{spec.name}: {_unpack(path, folder)}")
    if spec.kaggle:
        token = kaggle_token()
        if token is None:
            return f"{spec.name}: needs a Kaggle API token in {KAGGLE_TOKEN_FILE}. {spec.manual}"
        path = holding / f"{spec.kaggle.split('/')[-1]}.zip"
        if not path.exists():
            _fetch(kaggle_download_url(spec.kaggle, token), path)
        # Kaggle publishes no checksum, so check the zip's own CRCs before unpacking.
        with zipfile.ZipFile(path) as archive:
            broken = archive.testzip()
        if broken is not None:
            path.unlink()
            raise ValueError(f"{path.name}: {broken} is corrupt, so the zip was deleted")
        # Kaggle publishes no checksum, so this records one to compare later downloads with.
        (holding / f"{path.name}.sha256").write_text(f"{_sha256(path)}  {path.name}\n")
        messages.append(f"{spec.name}: {_unpack(path, folder)}, sha256 in {path.name}.sha256")
    if not messages:
        return f"{spec.name}: can't be fetched automatically. {spec.manual}"
    return " ".join(messages)


def download_all(root: Path, names: list[str]) -> list[str]:
    """Download each dataset in turn. One failure is reported and doesn't stop the others."""
    messages = []
    for name in names:
        try:
            messages.append(download(root, EXTERNAL_DATASETS[name]))
        except Exception as error:  # Network and archive errors take many forms.
            messages.append(f"{name}: FAILED: {type(error).__name__}: {error}")
    return messages


def build_evaluations(root: Path, names: list[str]) -> tuple[dict, dict]:
    """Return ({evaluation name: samples}, {evaluation name: info}) for datasets that are present."""
    evaluations, info = {}, {}
    for name in names:
        spec = EXTERNAL_DATASETS[name]
        folder = root / name
        if not folder.is_dir():
            continue
        if spec.uses_masks:
            images, tiles, skip_images, skip_tiles = collect_masked(folder)
            found = {f"{name}_image": (images, skip_images), f"{name}_tiles": (tiles, skip_tiles)}
        else:
            samples, skipped = collect_folders(folder, spec)
            found = {name: (samples, skipped)}
        for key, (samples, skipped) in found.items():
            if samples:
                evaluations[key] = samples
                info[key] = {
                    "mapped": sorted(set(spec.label_map.values())),
                    "skipped": dict(skipped),
                    "transpose": not spec.uses_masks,
                }
    return evaluations, info


def evaluation_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    mapped: list[str],
    temperature: float,
    threshold: float,
    overlap: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Scores on one external evaluation. Returns (metrics, confusion matrix over all classes).

    Macro-F1 averages only the mapped classes with images here, but a prediction of any of our 15
    classes counts, so predictions outside the mapped classes are errors.
    """
    indices = [classes.index(name) for name in mapped]
    present = [i for i in indices if (labels == i).any()]
    preds = logits.argmax(axis=1)
    in_set = np.isin(preds, indices)

    def macro(mask: np.ndarray) -> float:
        # A class with no images in this subset has no F1, rather than an F1 of zero.
        scored = [i for i in present if (labels[mask] == i).any()]
        return float(
            f1_score(labels[mask], preds[mask], labels=scored, average="macro", zero_division=0)
        )

    every = np.ones(len(labels), dtype=bool)
    metrics = {
        "n": len(labels),
        "macro_f1": macro(every),
        "accuracy": float((preds == labels).mean()),
        "out_of_set_share": float((~in_set).mean()),
        "overlap_n": int(overlap.sum()),
    }
    if (~overlap).any():
        metrics["macro_f1_no_overlap"] = macro(~overlap)
    confidence = softmax(logits / temperature, axis=1).max(axis=1)
    answered = confidence >= threshold
    metrics["abstain_coverage"] = float(answered.mean())
    if answered.any():
        metrics["abstain_macro_f1"] = macro(answered)
        metrics["abstain_accuracy"] = float((preds[answered] == labels[answered]).mean())
        metrics["abstain_out_of_set_share"] = float((~in_set[answered]).mean())
    for i in present:
        name, members = classes[i], labels == i
        metrics[f"n_{name}"] = int(members.sum())
        metrics[f"recall_{name}"] = float((preds[members] == i).mean())
        metrics[f"abstain_coverage_{name}"] = float(answered[members].mean())
        # An abstention counts as a miss here, so this can't look better just by abstaining.
        metrics[f"abstain_recall_{name}"] = float(
            ((preds[members] == i) & answered[members]).mean()
        )
    return metrics, confusion_matrix(labels, preds, labels=list(range(len(classes))))


def nearest_reference(
    external: list[str | None], reference: list[str], max_distance: int
) -> list[int]:
    """Index of the closest reference pHash within max_distance for each external one, else -1."""
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    table = np.array([list(bytes.fromhex(h)) for h in reference], dtype=np.uint8)
    found = []
    for value in external:
        if value is None or not len(table):
            found.append(-1)
            continue
        distances = bit_counts[table ^ np.frombuffer(bytes.fromhex(value), np.uint8)].sum(axis=1)
        best = int(distances.argmin())
        found.append(best if distances[best] <= max_distance else -1)
    return found


def overlap_with(external: list[str | None], reference: list[str], max_distance: int) -> np.ndarray:
    """True where an external pHash is within max_distance of any reference pHash."""
    return np.array([i >= 0 for i in nearest_reference(external, reference, max_distance)])


def overlap_breakdown(samples: list[dict], matches: dict[str, dict]) -> list[dict]:
    """Count overlapping images by their external label and their wheat_v1 label and split."""
    counts = Counter(
        (s["label"], matches[s["image"]]["label"], matches[s["image"]]["split"])
        for s in {s["image"]: s for s in samples}.values()
        if s["image"] in matches
    )
    return [
        {"external": a, "wheat_v1": b, "split": c, "images": n}
        for (a, b, c), n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


class ExternalImages:
    """External images in order, cropped to a tile if the sample has one."""

    def __init__(self, samples, labels, transform, use_exif):
        self.samples, self.labels = samples, labels
        self.transform, self.use_exif = transform, use_exif

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        from zarai.data.clean_wheat import _open_rgb

        sample = self.samples[index]
        if self.use_exif:
            image, _ = _open_rgb(sample["path"])
        else:
            with Image.open(sample["path"]) as source:
                image = source.convert("RGB")
        if sample["box"] is not None:
            image = image.crop(tuple(sample["box"]))
        return self.transform(image), self.labels[index]


def _safe_phash(path: str) -> str | None:
    from zarai.data.clean_wheat import _hash_image

    try:
        return _hash_image(path)[2]
    except OSError:
        return None


def _plot_confusion(matrix: np.ndarray, classes: list[str], title: str, path: Path) -> Path:
    from zarai.vision.plots import _pyplot

    plt = _pyplot()
    rows = [i for i in range(len(classes)) if matrix[i].sum()]
    shown = matrix[rows]
    shares = shown / shown.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.6 * len(rows) + 1.8)))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(rows)), [classes[i] for i in rows])
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for r in range(len(rows)):
        for c in range(len(classes)):
            if shown[r, c]:
                color = "white" if shares[r, c] > 0.5 else "black"
                ax.text(c, r, int(shown[r, c]), ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_seed(source, evaluations, info, overlaps, workers, work_dir: Path) -> None:
    """Score one frozen checkpoint on every available evaluation and log it to MLflow."""
    import mlflow
    from torch.utils.data import DataLoader

    from zarai.common.io import write_json
    from zarai.vision.eval import load_checkpoint
    from zarai.vision.train import predict
    from zarai.vision.transforms import build_transforms

    model, checkpoint = load_checkpoint(source.info.run_id, "cuda")
    classes = checkpoint["classes"]
    transform = build_transforms(
        checkpoint["image_size"],
        train=False,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        eval_resize=checkpoint["config"]["data"]["eval_resize"],
    )
    temperature = source.data.metrics["temperature"]
    threshold = source.data.metrics["abstain_threshold"]
    seed = source.data.params["seed"]
    with mlflow.start_run(run_name=f"cross_domain_seed{seed}"):
        mlflow.log_params(
            {
                "source_run_id": source.info.run_id,
                "source_seed": seed,
                "temperature": temperature,
                "abstain_threshold": threshold,
            }
        )
        for name, samples in evaluations.items():
            labels = np.array([classes.index(s["label"]) for s in samples])
            dataset = ExternalImages(samples, labels.tolist(), transform, info[name]["transpose"])
            loader = DataLoader(dataset, batch_size=128, num_workers=workers, pin_memory=True)
            logits = predict(model, loader, "cuda")[0]
            metrics, matrix = evaluation_metrics(
                logits,
                labels,
                classes,
                info[name]["mapped"],
                temperature,
                threshold,
                overlaps[name],
            )
            mlflow.log_metrics({f"{name}__{key}": value for key, value in metrics.items()})
            write_json(
                work_dir / f"{name}.json",
                {"classes": classes, "matrix": matrix.tolist(), "skipped": info[name]["skipped"]},
            )
            mlflow.log_artifact(str(work_dir / f"{name}.json"), "confusion")
            title = f"{name}, seed {seed}"
            figure = _plot_confusion(matrix, classes, title, work_dir / f"{name}.png")
            mlflow.log_artifact(str(figure), "confusion")
            log.info("%s seed %s: macro-F1 %.4f", name, seed, metrics["macro_f1"])
        if (work_dir / "overlap.json").exists():
            mlflow.log_artifact(str(work_dir / "overlap.json"), "overlap")


def _mean_sd(values: list[float | None], digits: int = 4) -> str:
    if not values or any(value is None for value in values):
        return TBD
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def evaluation_names() -> list[str]:
    names = []
    for name, spec in EXTERNAL_DATASETS.items():
        names += [f"{name}_image", f"{name}_tiles"] if spec.uses_masks else [name]
    return names


def _spec_for(evaluation: str) -> ExternalDataset:
    return EXTERNAL_DATASETS[re.sub(r"_(image|tiles)$", "", evaluation)]


def render(
    runs: list[dict[str, float]],
    confusion: dict[str, dict],
    status: dict[str, dict],
    frozen_ids: list[str],
    overlap: dict[str, list[dict]] | None = None,
) -> list[str]:
    """Build the generated report. Anything not measured, for every seed, is [TBD]."""

    def values(name: str, key: str) -> list[float | None]:
        return [run.get(f"{name}__{key}") for run in runs]

    def cell(name: str, key: str, digits: int = 4) -> str:
        return _mean_sd(values(name, key), digits)

    def count(name: str, key: str) -> str:
        found = values(name, key)
        return f"{found[0]:.0f}" if found and found[0] is not None else TBD

    names = evaluation_names()
    lines = [
        "Frozen model runs: " + ", ".join(f"`{run_id}`" for run_id in frozen_ids) + ".",
        "",
        f"Results are the mean ± SD over the {len(runs) or 'frozen'} seeds logged in the MLflow "
        f"experiment `{EXPERIMENT}`.",
        "",
        "## Dataset status",
        "",
        *table(
            ["dataset", "status", "images by class", "skipped"],
            [
                [
                    name,
                    status[name]["status"],
                    ", ".join(f"{k} {v}" for k, v in sorted(status[name]["counts"].items())) or "-",
                    ", ".join(f"{k}: {v}" for k, v in sorted(status[name]["skipped"].items()))
                    or "-",
                ]
                for name in EXTERNAL_DATASETS
            ],
        ),
        "",
        "## Results",
        "",
        "Macro-F1 averages the dataset's mapped classes only, and a prediction of any of our 15 "
        'classes counts. "Outside" is the share of predictions in classes the dataset doesn\'t '
        "contain. NWRD has one mapped class, so its macro-F1 depends only on recall.",
        "",
        *table(
            [
                "evaluation",
                "images",
                "macro-F1",
                "accuracy",
                "outside",
                "seen in train or val",
                "macro-F1 without those",
            ],
            [
                [
                    name,
                    count(name, "n"),
                    cell(name, "macro_f1"),
                    cell(name, "accuracy"),
                    cell(name, "out_of_set_share"),
                    count(name, "overlap_n"),
                    cell(name, "macro_f1_no_overlap"),
                ]
                for name in names
            ],
        ),
        "",
        "## With abstention at the val-chosen threshold",
        "",
        *table(
            ["evaluation", "coverage", "macro-F1 on answered", "accuracy on answered", "outside"],
            [
                [
                    name,
                    cell(name, "abstain_coverage"),
                    cell(name, "abstain_macro_f1"),
                    cell(name, "abstain_accuracy"),
                    cell(name, "abstain_out_of_set_share"),
                ]
                for name in names
            ],
        ),
        "",
        "## Per-class recall",
        "",
        "With abstention, an image the model declines counts as a miss.",
        "",
    ]
    rows = []
    for name in names:
        for label in sorted(set(_spec_for(name).label_map.values())):
            rows.append(
                [
                    name,
                    label,
                    count(name, f"n_{label}"),
                    cell(name, f"recall_{label}"),
                    cell(name, f"abstain_coverage_{label}"),
                    cell(name, f"abstain_recall_{label}"),
                ]
            )
    lines += table(
        ["evaluation", "class", "images", "recall", "coverage", "recall with abstention"], rows
    )
    lines += ["", "## Confusion", "", "Mean counts over the seeds, for true labels by predicted."]
    for name in names:
        lines += ["", f"### {name}", ""]
        entry = confusion.get(name)
        if not entry:
            lines.append(TBD)
            continue
        classes, matrix = entry["classes"], np.array(entry["matrix"], dtype=float)
        rows_shown = [i for i in range(len(classes)) if matrix[i].sum()]
        columns = [i for i in range(len(classes)) if matrix[:, i].sum()]
        lines += table(
            ["true \\ predicted", *[classes[c] for c in columns]],
            [[classes[r], *[f"{matrix[r, c]:.1f}" for c in columns]] for r in rows_shown],
        )
    lines += ["", "Per-seed matrices are in each run's `confusion/` artifacts."]
    lines += [
        "",
        "## Overlap with wheat_v1",
        "",
        f"External images within pHash distance {MAX_HAMMING_DISTANCE} of a wheat_v1 image, by "
        'their label in each dataset. Only overlaps with train or val are left out of "macro-F1 '
        'without those". pHash also matches different leaves on plain black backgrounds, so check '
        "these pairs by eye.",
        "",
    ]
    rows = [
        [name, item["external"], item["wheat_v1"], item["split"], item["images"]]
        for name, items in (overlap or {}).items()
        for item in items
    ]
    lines += (
        table(["evaluation", "external label", "wheat_v1 label", "wheat_v1 split", "images"], rows)
        if rows
        else [TBD]
    )
    return lines


def _read_logged(names: list[str], frozen_ids: list[str]) -> tuple[list[dict], dict, dict]:
    """Return the latest logged metrics per frozen seed, and the mean confusion per evaluation."""
    import mlflow

    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        return [], {}, {}
    found = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        output_format="list",
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    latest = {run.data.params["source_run_id"]: run for run in found}
    runs = [latest[run_id] for run_id in frozen_ids if run_id in latest]
    matrices = defaultdict(list)
    classes = {}
    for run in runs:
        for name in names:
            try:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path=f"confusion/{name}.json"
                )
            except Exception:  # MLflow raises several types for a missing artifact.
                continue
            entry = json.loads(Path(path).read_text(encoding="utf-8"))
            matrices[name].append(entry["matrix"])
            classes[name] = entry["classes"]
    confusion = {
        name: {"classes": classes[name], "matrix": np.mean(found, axis=0).tolist()}
        for name, found in matrices.items()
    }
    overlap = {}
    if runs:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=runs[-1].info.run_id, artifact_path="overlap/overlap.json"
            )
            overlap = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # MLflow raises several types for a missing artifact.
            overlap = {}
    return [run.data.metrics for run in runs], confusion, overlap


def main() -> None:
    import mlflow

    from zarai.vision.tracking import set_tracking_uri

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vision/wheat_vision_v1.yaml"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(EXTERNAL_DATASETS))
    parser.add_argument("--doc", type=Path, default=Path("experiments/wheat_v1_cross_domain.md"))
    parser.add_argument("--splits-csv", default="data/splits/wheat_v1.csv")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--work-dir", type=Path, default=Path("data/interim/cross_domain"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    get_logger(__name__)

    cfg = OmegaConf.load(args.config)
    frozen_ids = list(cfg.frozen.runs.values())
    set_tracking_uri(args.tracking_uri or cfg.mlflow.tracking_uri)
    status = {name: check_dataset(args.root, spec) for name, spec in EXTERNAL_DATASETS.items()}
    for name, state in status.items():
        log.info("%s: %s", name, state["status"])

    if not args.report_only:
        evaluations, info = build_evaluations(args.root, args.datasets)
        if evaluations:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            rows = read_csv(args.splits_csv)
            with Pool(args.workers) as pool:
                hashed = pool.map(_safe_phash, [r["path"] for r in rows], chunksize=32)
                kept = [(r, h) for r, h in zip(rows, hashed, strict=True) if h]
                reference = [h for _, h in kept]
                overlaps, breakdown = {}, {}
                for name, samples in evaluations.items():
                    unique = sorted({s["image"] for s in samples})
                    hashes = pool.map(_safe_phash, unique, chunksize=4)
                    nearest = nearest_reference(hashes, reference, MAX_HAMMING_DISTANCE)
                    matches = {
                        path: kept[i][0] for path, i in zip(unique, nearest, strict=True) if i >= 0
                    }
                    # Only train and val could have been learned from, so only they are left out.
                    seen = {p for p, row in matches.items() if row["split"] in ("train", "val")}
                    overlaps[name] = np.array([s["image"] in seen for s in samples])
                    breakdown[name] = overlap_breakdown(samples, matches)
                    log.info("%s: %d images seen in train or val", name, int(overlaps[name].sum()))
            args.work_dir.mkdir(parents=True, exist_ok=True)
            (args.work_dir / "overlap.json").write_text(json.dumps(breakdown, indent=2))
            mlflow.set_experiment(EXPERIMENT)
            client_runs = [mlflow.get_run(run_id) for run_id in frozen_ids]
            for source in client_runs:
                evaluate_seed(source, evaluations, info, overlaps, args.workers, args.work_dir)
        else:
            log.warning("no external dataset is present, so nothing was scored")

    runs, confusion, overlap = _read_logged(evaluation_names(), frozen_ids)
    args.doc.write_text(
        replace_section(
            args.doc.read_text(encoding="utf-8"),
            render(runs, confusion, status, frozen_ids, overlap),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
