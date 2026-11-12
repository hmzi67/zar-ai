"""Train and evaluate wheat_v2 models, log them to MLflow and rebuild the results table.

    .venv/bin/python train.py --dry-run --backbone dinov2_vit_b14 --mode lora
    .venv/bin/python train.py --backbone dinov2_vit_b14 --mode lora --aug strong --fold all --seeds 42,0,1

Extra key=value arguments override the config (e.g. train.epochs=5). All logic is in
zarai.vision.train_v2. --help lists the flags.
"""

from zarai.vision.train_v2 import main

if __name__ == "__main__":
    main()
