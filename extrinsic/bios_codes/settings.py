import os
from pathlib import Path

DATA_DIR = Path(os.environ.get('BIAS_DATA_DIR', './data/'))
EXPERIMENTS_DIR = DATA_DIR.joinpath('experiments/')
CACHE_DIR = DATA_DIR.joinpath('cache/')

BIOS_FILENAME = DATA_DIR.joinpath('BIOS_inferred.pkl')
NAMES_RACES_FILENAME = DATA_DIR.joinpath('names_and_stats.pkl')
EMBEDDINGS_FILENAME = DATA_DIR.joinpath('crawl-300d-2M.pickled')

ADULT_DATASET_DIR = DATA_DIR.joinpath('uci_adult/')

try:
    from settings_local import *
except ImportError as e:
    pass
