"""QLoRA fine-tuning with peft and bitsandbytes (needs a CUDA GPU)."""

from pathlib import Path

from zarai.common.config import parse_config
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run


def train(cfg, run_dir: Path, log) -> None:
    raise NotImplementedError(
        "TODO: BitsAndBytesConfig + prepare_model_for_kbit_training + LoraConfig + Trainer loop"
    )


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, "qlora_train")
    log = get_logger(__name__, run_dir)
    log.info("run directory: %s", run_dir)
    train(cfg, run_dir, log)


if __name__ == "__main__":
    main()
