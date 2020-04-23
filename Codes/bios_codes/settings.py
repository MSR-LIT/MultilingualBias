import os
from pathlib import Path

DATA_DIR = Path(os.environ.get('BIAS_DATA_DIR', './../BiosBias/EN/'))
CACHE_DIR = DATA_DIR.joinpath('cache_gender_balanced/finetune')
# CACHE_DIR = DATA_DIR.joinpath('cache_8l')
MODEL_DIR = Path(
    os.environ.get('BIOS_MODEL_DIR', './../BiosBias/EN/cache_gender_balanced/finetune'))
EMB_DIR = Path(
    os.environ.get('EMB_PATH_DIR', '/local/jyzhao/Github/fastText/alignment/'))
EMBEDDINGS_FILENAME = EMB_DIR.joinpath(
    'data/wiki.en_debias.vec'
    # 'res/wiki.es.align.vec'
)  # data/wiki.en_debias.vec; res/wiki.es-endeb.vec  res/wiki.es.align.vec

EXPERIMENTS_DIR = DATA_DIR.joinpath('experiments/')
BIOS_FILENAME = DATA_DIR.joinpath('BIOS.pkl')
BIOS_FILENAME1 = DATA_DIR.joinpath('bios_raw.pkl')
BIOS_FILENAME2 = DATA_DIR.joinpath('bios_raw_8l.pkl')
NAMES_RACES_FILENAME = DATA_DIR.joinpath('names_and_stats.pkl')

ADULT_DATASET_DIR = DATA_DIR.joinpath('uci_adult/')

try:
    from settings_local import *
except ImportError as e:
    pass
