"""Build wheat_v2 (Part A, via DVC) and check its PyTorch DataLoaders for a protocol and fold.

    .venv/bin/python dataloader.py --protocol loso                 # every LOSO fold
    .venv/bin/python dataloader.py --protocol loso --fold cgiar --aug strong --skip-preprocess

All logic is in zarai.data.wheat_v2 and zarai.vision.data_v2. --help lists the flags.
"""

from zarai.vision.data_v2 import main

if __name__ == "__main__":
    main()
