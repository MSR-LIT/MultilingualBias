import numpy as np
import torch.utils.data


class BiosBaseDataset(torch.utils.data.Dataset):
    def __init__(self, bios, titles, **kwargs):
        super().__init__()

        self.bios = bios
        self.titles = titles

        if len(kwargs) != 0:
            self.extra_targets = kwargs
        else:
            self.extra_targets = None

        self.nb_classes = len(set(self.titles))

    def __getitem__(self, index):
        bio = self.bios[index]
        title = self.titles[index]

        if self.extra_targets is None:
            targets = title
        else:
            extra_targets = [self.extra_targets[k][index] for k in self.extra_targets.keys()]
            targets = (title, *extra_targets)

        return bio, targets

    def __len__(self):
        return len(self.titles)


class BiosCountsDataset(BiosBaseDataset):
    def __init__(self, feature_names, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_names = feature_names
        self.nb_features = len(feature_names)

    def __getitem__(self, index):
        bio, targets = super().__getitem__(index)

        bio = bio.astype(np.float32)
        if not isinstance(bio, np.ndarray):
            bio = bio.toarray().squeeze()

        return bio, targets


class BiosSeqDataset(BiosBaseDataset):
    def __init__(self, vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab = vocab
        self.vocab_size = len(vocab) + 1  # +1 for 0 pad
        self.max_len = self.bios.shape[1]
