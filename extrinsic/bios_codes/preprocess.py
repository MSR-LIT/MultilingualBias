import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import save_pickle, load_pickle

import settings
from config import parse_args, PreprocessConfig
from settings import CACHE_DIR


def load_data(filename):
    data = load_pickle(filename)
    print(f'Data: {len(data)}')

    # stratified split
    labels_all = [bio['title'] for bio in data]
    data_train_val, data_test = train_test_split(data, test_size=0.20, random_state=42, stratify=labels_all)

    labels_train_val = [bio['title'] for bio in data_train_val]
    data_train, data_val = train_test_split(data_train_val, test_size=0.25, random_state=42, stratify=labels_train_val)

    return data_train, data_val, data_test


def load_bios_counts(scrubbed):
    filename = 'bios_counts_scrubbed.pkl' if scrubbed else 'bios_counts_no_scrubbed.pkl'

    bios_train, bios_val, bios_test = load_pickle(os.path.join(CACHE_DIR, filename))
    print(f'Bios: {bios_train.shape}, {bios_val.shape}, {bios_test.shape}')

    return bios_train, bios_val, bios_test


def load_bios_seq(scrubbed):
    filename = 'bios_seq_scrubbed.pkl' if scrubbed else 'bios_seq_no_scrubbed.pkl'

    bios_train, bios_val, bios_test = load_pickle(os.path.join(CACHE_DIR, filename))
    print(f'Bios: {bios_train.shape}, {bios_val.shape}, {bios_test.shape}')

    return bios_train, bios_val, bios_test


def load_titles():
    titles_train, titles_val, titles_test = load_pickle(os.path.join(CACHE_DIR, 'titles.pkl'))
    print(f'Titles: {len(titles_train)}, {len(titles_val)}, {len(titles_test)}')

    return titles_train, titles_val, titles_test


def load_vectorizers():
    vectorizer_scrubbed, vectorizer_no_scrubbed, titles_encoder = load_pickle(os.path.join(CACHE_DIR, 'vectorizers.pkl'))
    print(f'Vectorizer scrubbed: {len(vectorizer_scrubbed.get_feature_names())}')
    print(f'Vectorizer no scrubbed: {len(vectorizer_no_scrubbed.get_feature_names())}')
    print(f'Titles: {len(titles_encoder.classes_)}')

    return vectorizer_scrubbed, vectorizer_no_scrubbed, titles_encoder


def load_features_names(scrubbed):
    features_names = None

    vectorizer_scrubbed, vectorizer_no_scrubbed, _ = load_vectorizers()
    vectorizer = vectorizer_scrubbed if scrubbed else vectorizer_no_scrubbed
    features_names = vectorizer.get_feature_names()

    return features_names


def load_vocab(scrubbed):
    vectorizer_scrubbed, vectorizer_no_scrubbed, _ = load_vectorizers()
    vectorizer = vectorizer_scrubbed if scrubbed else vectorizer_no_scrubbed
    vocab = vectorizer.vocabulary_

    return vocab


def create_bios_sequences(bios, vectorizer, config):
    bio_analyzer = vectorizer.build_analyzer()

    bios_tokens = [bio_analyzer(bio) for bio in bios]
    bios_tokens_ids = [
        [vectorizer.vocabulary_[t] + 1 for t in bio if t in vectorizer.vocabulary_]  # +1 for the 0 pad
        for bio in bios_tokens
    ]
    bios_tokens_ids = [bio[:config.seq_max_len] for bio in bios_tokens_ids]

    bios_seq = [
        np.pad(bio, (0, config.seq_max_len - len(bio),), 'constant', constant_values=0)
        for bio in bios_tokens_ids
    ]
    bios_seq = np.stack(bios_seq).astype(np.long)

    return bios_seq


def remove_title(bio):
    bio_cleaned = bio['raw'][bio['start_pos']:].strip()

    return bio_cleaned


def encode_bios(bios_train, bios_val, bios_test, config):
    vectorizer = CountVectorizer(
        max_df=config.counts_max_df, min_df=config.counts_min_df, binary=config.counts_binary
    )
    vectorizer.fit(bios_train)
    print(f'Nb features: {len(vectorizer.get_feature_names())}')

    bios_counts_train, bios_counts_val, bios_counts_test = [
        vectorizer.transform(data) for data in (bios_train, bios_val, bios_test)
    ]
    print(f'Bios counts: {bios_counts_train.shape}, {bios_counts_val.shape}, {bios_counts_test.shape}')

    bios_seq_train, bios_seq_val, bios_seq_test = [
        create_bios_sequences(data, vectorizer, config)
        for data in (bios_train, bios_val, bios_test)
    ]
    print(f'Bios seq: {bios_seq_train.shape}, {bios_seq_val.shape}, {bios_seq_test.shape}')

    bios_counts = (bios_counts_train, bios_counts_val, bios_counts_test,)
    bios_seq = (bios_seq_train, bios_seq_val, bios_seq_test,)
    return vectorizer, bios_counts, bios_seq


def get_token_embedding(token, word_embeddings):
    if token in word_embeddings:
        return word_embeddings[token]

    token_lower = token.lower()
    if token_lower in word_embeddings:
        return word_embeddings[token_lower]

    return None


def main(cfg):
    print(f'Preprocessing started')

    # load data
    data_train, data_val, data_test = load_data(settings.BIOS_FILENAME)
    print(f'Data: {len(data_train)}, {len(data_val)}, {len(data_test)}')

    # create data lists
    bios_scrubbed_train, bios_scrubbed_val, bios_scrubbed_test = [
        [bio['bio'] for bio in data] for data in (data_train, data_val, data_test)
    ]
    bios_no_scrubbed_train, bios_no_scrubbed_val, bios_no_scrubbed_test = [
        [remove_title(bio) for bio in data] for data in (data_train, data_val, data_test)
    ]
    titles_train, titles_val, titles_test = [
        [bio['title'] for bio in data] for data in (data_train, data_val, data_test)
    ]

    # encode bios
    vectorizer_scrubbed, bios_counts_scrubbed, bios_seq_scrubbed = encode_bios(
        bios_scrubbed_train, bios_scrubbed_val, bios_scrubbed_test, cfg
    )
    vectorizer_no_scrubbed, bios_counts_no_scrubbed, bios_seq_no_scrubbed = encode_bios(
        bios_no_scrubbed_train, bios_no_scrubbed_val, bios_no_scrubbed_test, cfg
    )

    # encode titles
    titles_encoder = LabelEncoder()
    titles_encoder.fit(titles_train)
    print(f'Nb titles: {len(titles_encoder.classes_)}')

    titles_train, titles_val, titles_test = [
        titles_encoder.transform(data) for data in (titles_train, titles_val, titles_test)
    ]
    print(f'Titles: {titles_train.shape}, {titles_val.shape}, {titles_test.shape}')

    # create names embeddings
    word_embeddings = load_pickle(settings.EMBEDDINGS_FILENAME)
    print(f'Embeddings: {len(word_embeddings)}')

    save_pickle(os.path.join(CACHE_DIR, 'bios_counts_scrubbed.pkl'), bios_counts_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'bios_counts_no_scrubbed.pkl'), bios_counts_no_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'bios_seq_scrubbed.pkl'), bios_seq_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'bios_seq_no_scrubbed.pkl'), bios_seq_no_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'titles.pkl'), (titles_train, titles_val, titles_test))
    save_pickle(
        os.path.join(CACHE_DIR, 'vectorizers.pkl'), (vectorizer_scrubbed, vectorizer_no_scrubbed, titles_encoder,)
    )

    print(f'Preprocessing finished')


if __name__ == '__main__':
    cfg = parse_args(PreprocessConfig, 'Preprocess')
    main(cfg)
