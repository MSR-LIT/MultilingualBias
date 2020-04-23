import pickle
import io

import numpy as np
import torch
from tqdm import tqdm


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device=device) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj


def load_embeddings(fname):
    print(f"Loading embeddings from {fname}")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking).long()

    lengths = masks.sum(dim=dim)

    return lengths


def softmax_masked(inputs, mask, dim=1, epsilon=0.000001):
    inputs_exp = torch.exp(inputs)
    inputs_exp = inputs_exp * mask.float()
    inputs_exp_sum = inputs_exp.sum(dim=dim)
    inputs_attention = inputs_exp / (inputs_exp_sum.unsqueeze(dim) + epsilon)

    return inputs_attention


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=1):
    pin_memory = torch.cuda.is_available()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader


def get_trainable_parameters(parameters):
    parameters = list(parameters)
    nb_params_before = sum(p.nelement() for p in parameters)

    parameters = [p for p in parameters if p.requires_grad]
    nb_params_after = sum(p.nelement() for p in parameters)

    print(f'Parameters: {nb_params_before} -> {nb_params_after}')
    return parameters


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def load_pickle(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)
    except EOFError:
        print(f'Cannot load: {filename}')
        obj = None

    return obj


def save_pickle(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)


def restore_weights(model, filename):
    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        def map_location(storage, loc): return storage

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)

    print(f'Model restored: {filename}')


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))

    # print(f'Model saved: {filename}')
