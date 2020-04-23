import os, sys

import numpy as np
import torch
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter
from sklearn.utils import compute_class_weight

from settings import CACHE_DIR, EMBEDDINGS_FILENAME, MODEL_DIR
from config import parse_args, TrainConfig
from utils import create_data_loader, to_device, get_trainable_parameters, save_weights, load_pickle, init_weights, restore_weights, save_pickle
from preprocess import load_bios_counts, load_bios_seq, load_titles, load_features_names, load_vocab, load_embeddings
from models import CountsModel, RNNModel, HANModel
from datasets import BiosSeqDataset, BiosCountsDataset
from collections import Counter


class BiosLoss(torch.nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()

        self.config = config
        self.weight = weight

        self.criterion_xent = torch.nn.CrossEntropyLoss(weight=weight)

        self.epoch = 1

    def epoch_complete(self):
        self.epoch += 1

    def forward(self, inputs, targets):
        if isinstance(targets, (tuple, list)):
            labels, *extra_targets = targets
        else:
            labels = targets

        loss_xent = self.criterion_xent(inputs, labels)

        return loss_xent


def create_embeddings(word_embeddings, vocab):
    embeddings_size = len(list(next(iter(word_embeddings.values()))))
    vocab_size = len(vocab)

    # W_emb = np.zeros((vocab_size + 1, embeddings_size))  # +1 for the 0 pad
    W_emb = np.zeros((10000, embeddings_size))  # +1 for the 0 pad

    nb_unk = 0
    for token, i in vocab.items():
        token_idx = i + 1  # +1 for 0 pad
        if token in word_embeddings:
            W_emb[token_idx] = word_embeddings[token]
        else:
            W_emb[token_idx] = np.random.uniform(-0.3, 0.3, embeddings_size)
            nb_unk += 1

    print(f'Unknown tokens: {nb_unk}')
    return W_emb


def create_model(config, dataset, create_W_emb=False):
    model_class = None
    model_params = {}

    if config.model == 'counts':
        model_class = CountsModel
        model_params.update(
            dict(
                input_size=dataset.nb_features,
                output_size=dataset.nb_classes,
                dropout=config.dropout,
            ))

    elif config.model == 'han' or config.model == 'rnn':
        if create_W_emb:
            word_embeddings = load_embeddings(EMBEDDINGS_FILENAME)
            print(f'Embeddings: {len(word_embeddings)}')

            W_emb = create_embeddings(word_embeddings, dataset.vocab)
        else:
            W_emb = None

        model_class = RNNModel
        model_params.update(
            dict(
                # vocab_size=dataset.vocab_size,
                vocab_size=10000,  # set this number to a fixed one
                trainable_embeddings=config.trainable_embeddings,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                output_size=dataset.nb_classes,
                W_emb=W_emb,
                embedding_size=300,
                padding_idx=0))

        if config.model == 'han':
            model_class = HANModel
            model_params.update(dict(attention_size=config.attention_size, ))

    else:
        raise ValueError(f'Model {config.model} is unknown')

    model = model_class(**model_params)
    init_weights(model)
    model = to_device(model)

    print(f'Model: {model.__class__.__name__}')

    return model


def create_dataset(config):
    dataset_class = None
    dataset_params_train = {}
    dataset_params_val = {}
    dataset_params_test = {}

    def add_dataset_param(name, value_train, value_dev=None, value_test=None):
        dataset_params_train[name] = value_train
        dataset_params_val[
            name] = value_dev if value_dev is not None else value_train
        dataset_params_test[
            name] = value_test if value_test is not None else value_train

    titles_train, titles_val, titles_test = load_titles()
    add_dataset_param('titles', titles_train, titles_val, titles_test)
    print("Data statistics:")
    print("\tTraining: ", Counter(titles_train))
    print("\tVal: ", Counter(titles_val))
    print("\tTest: ", Counter(titles_test))

    if config.model == 'counts' or config.model == 'counts_deep':
        dataset_class = BiosCountsDataset

        bios_train, bios_val, bios_test = load_bios_counts(config.scrubbed)
        add_dataset_param('bios', bios_train, bios_val, bios_test)

        features_names = load_features_names(config.scrubbed)
        add_dataset_param('feature_names', features_names)
    elif config.model == 'han' or config.model == 'rnn':
        dataset_class = BiosSeqDataset

        bios_train, bios_val, bios_test = load_bios_seq(config.scrubbed)
        add_dataset_param('bios', bios_train, bios_val, bios_test)

        vocab = load_vocab(config.scrubbed)
        add_dataset_param('vocab', vocab)
    else:
        raise ValueError(f'Dataset for the model {config.model} is unknown')

    dataset_train = dataset_class(**dataset_params_train)
    dataset_val = dataset_class(**dataset_params_val)
    dataset_test = dataset_class(**dataset_params_test)

    print(
        f'Dataset: {type(dataset_train).__name__} - {len(dataset_train)}, {len(dataset_val)}, {len(dataset_test)}'
    )
    return dataset_train, dataset_val, dataset_test


def main(cfg):
    print(f'Training stated: {cfg.model}')

    dataset_train, dataset_val, dataset_test = create_dataset(cfg)
    data_loader_train = create_data_loader(dataset_train,
                                           cfg.batch_size,
                                           shuffle=True,
                                           num_workers=3)
    data_loader_dev = create_data_loader(dataset_val,
                                         cfg.batch_size,
                                         shuffle=False,
                                         num_workers=3)
    data_loader_test = create_data_loader(dataset_test,
                                          cfg.batch_size,
                                          shuffle=False,
                                          num_workers=3)
    print(f'Data loader: {len(data_loader_train)}, {len(data_loader_dev)}')

    # weighted loss
    if cfg.use_class_weight:
        classes_weights = compute_class_weight('balanced',
                                               np.unique(dataset_train.titles),
                                               dataset_train.titles)
        classes_weights = to_device(classes_weights).float()
        print(f'Class weight: yes')
    else:
        classes_weights = None
        print(f'Class weight: no')

    # create model
    scrub_flag = 'no_scrub'
    if cfg.scrubbed == True:
        scrub_flag = 'scrub'

    model = create_model(cfg, dataset_train, create_W_emb=True)
    if cfg.resume != "":
        restore_weights(model, cfg.resume)

    model_parameters = get_trainable_parameters(model.parameters())
    optimizer = torch.optim.Adam(model_parameters,
                                 cfg.learning_rate,
                                 weight_decay=cfg.weight_decay,
                                 amsgrad=True)

    criterion = BiosLoss(cfg, classes_weights)

    def update_function(engine, batch):
        if engine.state.epoch % 10 == 0: 
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 1
            
        model.train()
        optimizer.zero_grad()

        inputs, targets = to_device(batch)

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, cfg.max_grad_norm)
        optimizer.step()

        return loss.item()

    def inference_function(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = to_device(batch)

            logits = model(inputs)

            return logits, targets

    trainer = Engine(update_function)
    evaluator = Engine(inference_function)

    metrics = [
        ('loss', Loss(criterion)),
        ('accuracy', Accuracy()),
    ]
    for name, metric in metrics:
        metric.attach(evaluator, name)

    best_val_loss = np.inf
    nb_epoch_no_improvement = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def loss_step(engine):
        criterion.epoch_complete()

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_model(engine):
        nonlocal best_val_loss, nb_epoch_no_improvement

        def log_progress(mode, metrics_values):
            metrics_str = ', '.join([
                f'{metric_name} {metrics_values[metric_name]:.3f}'
                for metric_name, _ in metrics
            ])
            print(f'{mode}: {metrics_str}', end=' | ')

        # evaluator.run(data_loader_train)
        # metrics_train = evaluator.state.metrics.copy()

        evaluator.run(data_loader_dev)
        metrics_dev = evaluator.state.metrics.copy()

        print(f'Epoch {engine.state.epoch:>3}', end=' | ')
        # log_progress('train', metrics_train)
        log_progress('dev', metrics_dev)

        if best_val_loss > metrics_dev[
                'loss'] or engine.state.epoch <= cfg.min_epochs:
            best_val_loss = metrics_dev['loss']
            nb_epoch_no_improvement = 0

            save_weights(
                model,
                os.path.join(MODEL_DIR,
                             f'model_{cfg.model_flag}_{scrub_flag}.pt'))
            print('Model saved', end=' ')
        else:
            nb_epoch_no_improvement += 1

        if cfg.early_stopping_patience != 0 and nb_epoch_no_improvement > cfg.early_stopping_patience:
            trainer.terminate()

        print()

    trainer.run(data_loader_train, max_epochs=cfg.nb_epochs)
    print(f'Training finished')
    print(
        f"Saving model at {os.path.join(MODEL_DIR, f'model_{cfg.model_flag}_{scrub_flag}.pt')}"
    )
    evaluator.run(data_loader_test)
    metrics_test = evaluator.state.metrics.copy()
    print(f"acc on {CACHE_DIR} test:", metrics_test['accuracy'])


def predict(cfg):
    _, dataset_val, dataset_test = create_dataset(cfg)
    data_loader_dev = create_data_loader(dataset_val,
                                         cfg.batch_size,
                                         shuffle=False,
                                         num_workers=3)
    data_loader_test = create_data_loader(dataset_test,
                                          cfg.batch_size,
                                          shuffle=False,
                                          num_workers=3)
    print(f'Data loader: {len(data_loader_test)}')
    print("Test Data Statistics:", Counter(dataset_test.titles))
    scrub_flag = 'no_scrub'
    if cfg.scrubbed == True:
        scrub_flag = 'scrub'

    preds = []

    def inference_function(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = to_device(batch)

            logits = model(inputs)
            y_pred = np.argmax(logits.cpu().numpy(), axis=1)
            preds.extend(y_pred)
            return logits, targets

    metrics = [
        ('accuracy', Accuracy()),
    ]
    predictor = Engine(inference_function)

    for name, metric in metrics:
        metric.attach(predictor, name)

    # create model
    print(f'Start Testing')

    model = create_model(cfg, dataset_val, create_W_emb=True)
    restore_weights(
        model,
        os.path.join(MODEL_DIR, f'model_{cfg.model_flag}_{scrub_flag}.pt'))

    # for name, param in model.state_dict().items():
    #     if name == 'dropout':
    #         print(name, param)
    print(model.dropout)
    predictor.run(data_loader_test)
    metrics_test = predictor.state.metrics.copy()
    print(metrics_test['accuracy'])
    # mf = str(MODEL_DIR).strip().split('/')[-1]
    print("Saving predictions to {}".format(
        os.path.join(
            CACHE_DIR, "pred-" + scrub_flag + '-' + cfg.model_flag + '-T-' +
            cfg.pred_flag + ".pkl")))
    save_pickle(
        os.path.join(
            CACHE_DIR, "pred-" + scrub_flag + '-' + cfg.model_flag + '-T-' +
            cfg.pred_flag + ".pkl"), preds)


if __name__ == '__main__':
    cfg = parse_args(TrainConfig, 'Train model')
    print(cfg)
    if cfg.train_flag != 0:
        main(cfg)
    if cfg.reload_flag != 0:
        predict(cfg)
