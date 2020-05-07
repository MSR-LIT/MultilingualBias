import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import save_pickle, load_pickle, load_embeddings

import settings
from config import parse_args, PreprocessConfig
from settings import CACHE_DIR
from collections import Counter, defaultdict
import random

random.seed(1110)

occs = [x.strip() for x in open('filtered_occs_8', 'r').readlines()]


def load_data(filename):
    data = load_pickle(filename)
    data = [bio for bio in data if bio['title'] in occs]
    print(f'Data: {len(data)}')

    # stratified split
    labels_all = [bio['title'] for bio in data]
    data_train_val, data_test = train_test_split(data,
                                                 test_size=0.20,
                                                 random_state=42,
                                                 stratify=labels_all)

    labels_train_val = [bio['title'] for bio in data_train_val]
    data_train, data_val = train_test_split(data_train_val,
                                            test_size=0.25,
                                            random_state=42,
                                            stratify=labels_train_val)

    return data_train, data_val, data_test


def load_bios_counts(scrubbed):
    filename = 'bios_counts_scrubbed.pkl' if scrubbed else 'bios_counts_no_scrubbed.pkl'

    bios_train, bios_val, bios_test = load_pickle(
        os.path.join(CACHE_DIR, filename))
    print(f'Bios: {bios_train.shape}, {bios_val.shape}, {bios_test.shape}')

    return bios_train, bios_val, bios_test


def load_bios_seq(scrubbed):
    filename = 'bios_seq_scrubbed.pkl' if scrubbed else 'bios_seq_no_scrubbed.pkl'

    bios_train, bios_val, bios_test = load_pickle(
        os.path.join(CACHE_DIR, filename))
    print(f'Loading data from {os.path.join(CACHE_DIR, filename)}')
    print(f'Bios: {bios_train.shape}, {bios_val.shape}, {bios_test.shape}')

    return bios_train, bios_val, bios_test

def load_bios_raw():
    filename = 'bios_raw.pkl'
    train, val, test = load_pickle(os.path.join(CACHE_DIR, filename))
    bios_train = [x['bio'] for x in train]
    bios_val = [x['bio'] for x in val]
    bios_test = [x['bio'] for x in test]
    print(f'Loading data from {os.path.join(CACHE_DIR, filename)}')
    print(f'Bios: {len(bios_train)}-train, {len(bios_val)}-val, {len(bios_test)}-test')
    return bios_train, bios_val, bios_test
    

def load_titles():
    titles_train, titles_val, titles_test = load_pickle(
        os.path.join(CACHE_DIR, 'titles.pkl'))
    print(
        f'Titles: {len(titles_train)}, {len(titles_val)}, {len(titles_test)}')

    return titles_train, titles_val, titles_test


def load_vectorizers():
    vectorizer_scrubbed, vectorizer_no_scrubbed, titles_encoder = load_pickle(
        os.path.join(CACHE_DIR, 'vectorizers.pkl'))
    print(
        f'Vectorizer scrubbed: {len(vectorizer_scrubbed.get_feature_names())}')
    print(
        f'Vectorizer no scrubbed: {len(vectorizer_no_scrubbed.get_feature_names())}'
    )
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
        # +1 for the 0 pad
        [
            vectorizer.vocabulary_[t] + 1 for t in bio
            if t in vectorizer.vocabulary_
        ] for bio in bios_tokens
    ]
    bios_tokens_ids = [bio[:config.seq_max_len] for bio in bios_tokens_ids]

    bios_seq = [
        np.pad(bio, (
            0,
            config.seq_max_len - len(bio),
        ),
               'constant',
               constant_values=0) for bio in bios_tokens_ids
    ]
    bios_seq = np.stack(bios_seq).astype(np.long)

    return bios_seq


def remove_title(bio):
    bio_cleaned = bio['raw'][bio['start_pos']:].strip()

    return bio_cleaned


def encode_bios(bios_train, bios_val, bios_test, config, vectorizer):
    # vectorizer = CountVectorizer(
    #     max_df=config.counts_max_df, min_df=config.counts_min_df, binary=config.counts_binary
    # )
    # vectorizer.fit(bios_train)
    # print(f'Nb features: {len(vectorizer.get_feature_names())}')

    bios_counts_train, bios_counts_val, bios_counts_test = [
        vectorizer.transform(data)
        for data in (bios_train, bios_val, bios_test)
    ]

    bios_seq_train, bios_seq_val, bios_seq_test = [
        create_bios_sequences(data, vectorizer, config)
        for data in (bios_train, bios_val, bios_test)
    ]
    print(
        f'Bios seq: {bios_seq_train.shape}, {bios_seq_val.shape}, {bios_seq_test.shape}'
    )

    bios_counts = (
        bios_counts_train,
        bios_counts_val,
        bios_counts_test,
    )
    bios_seq = (
        bios_seq_train,
        bios_seq_val,
        bios_seq_test,
    )
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
    # if 'EN' in str(settings.BIOS_FILENAME):
    #     data_train, data_val, data_test = load_pickle(
    #         './../BiosBias/EN/cache_7l/bios_raw.pkl'
    #     )  #load data from splitted dataset.
    # elif 'FR' in str(settings.BIOS_FILENAME):
    #    data_train, data_val, data_test = load_pickle(
    #         './../BiosBias/FR/cache_8l/bios_raw.pkl'
    #     )  #load data from splitted dataset.
    # else:
    #     print("Wrong dataset")
    #     return 0
    print(f'Data: {len(data_train)}, {len(data_val)}, {len(data_test)}')

    for x in data_train:
        if 'bio' not in x.keys():
            print(x)

    #blance the occ  and keep the original gender ratio
    def _balance_occ(dataset, train_flag='train'):
        new = []
        cbocc = dict()
        for x in dataset:
            occ = x['title']
            if occ not in cbocc:
                cbocc[occ] = defaultdict(list)
            if x['gender'] == 'M':
                cbocc[occ]['M'].append(x)
            elif x['gender'] == 'F':
                cbocc[occ]['F'].append(x)
            else:
                print("Wrong gender")
                break

        if 'ES' in str(settings.BIOS_FILENAME):
            count_train = 1500
            count_val = 400
        elif 'EN' in str(settings.BIOS_FILENAME):
            count_train = 2000
            count_val = 700
        if train_flag == 'train':
            count = count_train
        else:
            count = count_val
        for occ in cbocc:
            m = len(cbocc[occ]['M'])
            f = len(cbocc[occ]['F'])
            if m + f < count:
                new.extend(cbocc[occ]['M'] + cbocc[occ]['F'])
                m_toadd = int((count - (m + f)) * (m / (m + f)))
                f_toadd = int((count - (m + f)) * (f / (m + f)))
                if f_toadd // f != 0:
                    new.extend(cbocc[occ]['F'] * (f_toadd // f))
                new.extend(random.sample(cbocc[occ]['F'], f_toadd % f))
                if m_toadd // m != 0:
                    new.extend(cbocc[occ]['M'] * (m_toadd // m))
                new.extend(random.sample(cbocc[occ]['M'], m_toadd % m))
            else:
                new.extend(
                    random.sample(cbocc[occ]['M'], int(count * m / (m + f))) +
                    random.sample(cbocc[occ]['F'], int(count * f / (m + f))))
        random.shuffle(new)
        return new

    #balance the dataset -- gender
    # For EN_7l, add some tricks to change the dataset distribution
    def _balance_gender(data_set, flag='test'):
        print("Balancing the {} gender".format(flag))
        cbocc = dict()
        for x in data_set:
            occ = x['title']
            if occ not in cbocc:
                cbocc[occ] = defaultdict(list)
            if x['gender'] == 'M':
                cbocc[occ]['M'].append(x)
            elif x['gender'] == 'F':
                cbocc[occ]['F'].append(x)
            else:
                print("Wrong gender")
                break
        all_females = sum([len(cbocc[x]['F']) for x in cbocc])
        all_males = sum([len(cbocc[x]['M']) for x in cbocc])
        print("In total, {} females and {} males".format(all_females, all_males))
        ratio = {
            x: (len(cbocc[x]['M']) + len(cbocc[x]['F'])) /
            (all_males + all_females)
            for x in cbocc
        }
        print(ratio)
        if flag == 'test':
            dataset = []
            if 'ES' in str(settings.BIOS_FILENAME):
                count = 50
            elif 'EN' in str(settings.BIOS_FILENAME):
                count = 150
            else:
                print("wrong dataset")
            for occ in cbocc:
                print(occ, ratio[occ] * all_females, len(cbocc[occ]['F']),
                      len(cbocc[occ]['M']))
                dataset.extend(
                    random.sample(
                        cbocc[occ]['M'],
                        min(int(ratio[occ] *
                                all_females), len(cbocc[occ]['F']), len(cbocc[occ]['M']))) +
                    random.sample(
                        cbocc[occ]['F'],
                        min(int(ratio[occ] * all_females), len(
                            cbocc[occ]['F']), len(cbocc[occ]['M']))))
            random.shuffle(dataset)
        else:
            dataset = []
            for occ in cbocc:
                m = len(cbocc[occ]['M'])
                f = len(cbocc[occ]['F'])
                # if occ == 'model' and flag=='train':
                #     f -= 600
                # if occ == 'architect' and flag == 'train':
                #     m -= 230
                # if occ == 'poet' and flag == 'train':
                #     m -= 580
                #     f -= 450
                # if occ == 'model' and flag == 'dev':
                #     f -= 240
                # print(occ, m, f)
                if m > f:
                    toadd = m - f
                    if toadd // f != 0:
                        dataset.extend(random.sample(cbocc[occ]['F'],f) * (toadd // f ))
                    dataset.extend(random.sample(cbocc[occ]['F'], toadd % f))
                    dataset.extend(random.sample(cbocc[occ]['M'], m))
                    dataset.extend(random.sample(cbocc[occ]['F'],f))
                else:
                    toadd = f - m
                    if toadd // m != 0:
                        dataset.extend(random.sample(cbocc[occ]['M'],m) * (toadd // m ))
                    dataset.extend(random.sample(cbocc[occ]['M'], toadd % m))
                    dataset.extend(random.sample(cbocc[occ]['F'], f))
                    dataset.extend(random.sample(cbocc[occ]['M'],m))
            random.shuffle(dataset)
        return dataset

    #balance both occ and gender
    def _balance_occ_gender(dataset, train_flag):
        new = []
        cbocc = dict()
        for x in dataset:
            occ = x['title']
            if occ not in cbocc:
                cbocc[occ] = defaultdict(list)
            if x['gender'] == 'M':
                cbocc[occ]['M'].append(x)
            elif x['gender'] == 'F':
                cbocc[occ]['F'].append(x)
            else:
                print("Wrong gender")
                break

        if 'ES' in str(settings.BIOS_FILENAME):
            count_train = 500
            count_val = 100
        elif 'EN' in str(settings.BIOS_FILENAME):
            count_train = 1500
            count_val = 300
        if train_flag == 'train':
            count = count_train
        else:
            count = count_val
        for occ in cbocc:
            m = len(cbocc[occ]['M'])
            f = len(cbocc[occ]['F'])
            if m < count:
                new.extend(cbocc[occ]['M'] * (count // m))
                new.extend(random.sample(cbocc[occ]['M'], count % m))
            else:
                new.extend(random.sample(cbocc[occ]['M'], count))
            if f < count:
                new.extend(cbocc[occ]['F'] * (count // f))
                new.extend(random.sample(cbocc[occ]['F'], count % f))
            else:
                new.extend(random.sample(cbocc[occ]['F'], count))
        random.shuffle(new)
        return new

    data_train = _balance_gender(data_train, 'train')
    data_val = _balance_gender(data_val, 'dev')
    print(Counter([x['title'] for x in data_val]))
    print(Counter([x['gender'] for x in data_val]))
    # data_test = _balance_gender(data_test, 'test')
    # data_train = _balance_occ(data_train, 'train')
    # data_val = _balance_occ(data_val, 'val')
    # data_train = _balance_occ_gender(data_train, 'train')
    # data_val = _balance_occ_gender(data_val, 'val')
    print(Counter([x['title'] for x in data_train]))
    print(Counter([x['gender'] for x in data_train]))
    # create data lists
    bios_scrubbed_train, bios_scrubbed_val, bios_scrubbed_test = [[
        bio['bio'] for bio in data
    ] for data in (data_train, data_val, data_test)]
    bios_no_scrubbed_train, bios_no_scrubbed_val, bios_no_scrubbed_test = [[
        remove_title(bio) for bio in data
    ] for data in (data_train, data_val, data_test)]
    titles_train, titles_val, titles_test = [[bio['title'] for bio in data]
                                             for data in (data_train, data_val,
                                                          data_test)]

    vectorizer = CountVectorizer(max_df=cfg.counts_max_df,
     min_df=cfg.counts_min_df,
     binary=cfg.counts_binary)
    vectorizer.fit(bios_no_scrubbed_train)

    #load vectorizer
    # if 'ES' in str(settings.BIOS_FILENAME):
    #     vectorizer, _, _ = load_pickle(
    #         '/home/jyzhao/git7/Gender_bias_in-Cross_Lingual_Transfer_Learning/BiosBias/ES/cache_gender_balanced/vectorizers.pkl'
    #     )

    # elif 'EN' in str(settings.BIOS_FILENAME):
    #     vectorizer, _, _ = load_pickle(
    #         '/home/jyzhao/git7/Gender_bias_in-Cross_Lingual_Transfer_Learning/BiosBias/EN/cache_gender_balanced/vectorizers.pkl'
    #     )
    # else:
    #     print('Wrong vectorizer.pkl')
    #     return 0
    print(f'Nb features: {len(vectorizer.get_feature_names())}')

    # encode bios
    vectorizer_scrubbed, bios_counts_scrubbed, bios_seq_scrubbed = encode_bios(
        bios_scrubbed_train, bios_scrubbed_val, bios_scrubbed_test, cfg,
        vectorizer)
    vectorizer_no_scrubbed, bios_counts_no_scrubbed, bios_seq_no_scrubbed = encode_bios(
        bios_no_scrubbed_train, bios_no_scrubbed_val, bios_no_scrubbed_test,
        cfg, vectorizer)

    # encode titles
    titles_encoder = LabelEncoder()
    titles_encoder.fit(occs)
    print(f'Nb titles: {len(titles_encoder.classes_)}')

    titles_train, titles_val, titles_test = [
        titles_encoder.transform(data)
        for data in (titles_train, titles_val, titles_test)
    ]
    print(
        f'Titles: {titles_train.shape}, {titles_val.shape}, {titles_test.shape}'
    )
    print(
        f'Train Data Distribution: {Counter(titles_train)}\nDev distribution: {Counter(titles_val)}\nTest data distribution: {Counter(titles_test)}'
    )
    # create names embeddings
    # word_embeddings = load_embeddings(settings.EMBEDDINGS_FILENAME)
    # print(f'Embeddings: {len(word_embeddings)}')

    # save_pickle(os.path.join(CACHE_DIR, 'bios_counts_scrubbed.pkl'),
    #             bios_counts_scrubbed)
    # save_pickle(os.path.join(
    #     CACHE_DIR, 'bios_counts_no_scrubbed.pkl'), bios_counts_no_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'bios_seq_scrubbed.pkl'),
                bios_seq_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'bios_seq_no_scrubbed.pkl'),
                bios_seq_no_scrubbed)
    save_pickle(os.path.join(CACHE_DIR, 'titles.pkl'),
                (titles_train, titles_val, titles_test))
    save_pickle(os.path.join(CACHE_DIR, 'vectorizers.pkl'), (
        vectorizer_scrubbed,
        vectorizer_no_scrubbed,
        titles_encoder,
    ))
    save_pickle(os.path.join(CACHE_DIR, 'bios_raw.pkl'), (
        data_train,
        data_val,
        data_test,
    ))

    print(f'Preprocessing finished')


if __name__ == '__main__':
    cfg = parse_args(PreprocessConfig, 'Preprocess')
    main(cfg)
