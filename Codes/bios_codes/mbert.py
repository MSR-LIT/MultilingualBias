import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

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
from utils import create_data_loader, to_device, get_trainable_parameters, save_weights, load_pickle, init_weights, restore_weights, save_pickle, get_sequences_lengths, softmax_masked
from preprocess import  load_bios_raw, load_titles, load_features_names, load_vocab, load_embeddings

from rnn_encoder import GRUEncoder
from models import  RNNModel, HANModel
from datasets import BiosSeqDataset, BiosCountsDataset
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
mbertmodel = BertModel.from_pretrained('bert-base-multilingual-cased')
mbertmodel.eval()
mbertmodel.to('cuda')

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

class HANModel(torch.nn.Module):
    def __init__(self, embedding_size, trainable_embeddings, hidden_size, attention_size, output_size, dropout,
                 padding_idx=0):
        super().__init__()

        self.padding_idx = padding_idx
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.encoder_sentences = GRUEncoder(embedding_size, hidden_size, bidirectional=True,
                                            return_sequence=True)
        self.att_sentences = torch.nn.Linear(hidden_size * 2, attention_size)
        self.att_reduce = torch.nn.Linear(attention_size, 1, bias=False)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, output_size),
        )

    def zero_state(self, batch_size):
        state_shape = (2, batch_size, self.hidden_size)

        # will work on both GPU and CPU in contrast to just Variable(*state_shape)
        h = to_device(torch.zeros(*state_shape))
        return h

    def encode_sentences(self, inputs, inputs_emb):
        mask = inputs != self.padding_idx
        inputs_len = get_sequences_lengths(inputs)

        # inputs_emb = self.embedding(inputs)
        inputs_enc = self.encoder_sentences(inputs_emb, inputs_len)
        inputs_enc = F.dropout(inputs_enc, self.dropout, self.training)

        mask = mask[:, :inputs_enc.size(1)]
    
        att_vec = self.att_sentences(inputs_enc)
        att_weights = self.att_reduce(att_vec)
        # print("input len:", inputs_len)
        # print(inputs_emb.size())
        # print(mask.size())
        # print(att_weights.size())
        att = softmax_masked(att_weights, mask.unsqueeze(-1))

        inputs_att = torch.sum(inputs_enc * att, dim=1)
        inputs_att = F.dropout(inputs_att, self.dropout, self.training)

        return inputs_att, att

    def get_logits(self, inputs_att):
        logits = self.out(inputs_att)
        return logits

    def forward(self, inputs, inputs_emb):
        inputs_att, att = self.encode_sentences(inputs, inputs_emb)
        logits = self.get_logits(inputs_att)

        return logits

def create_model(config, dataset, create_W_emb=False):
    model_class = None
    model_params = {}

    if config.model == 'han' or config.model == 'rnn':
        model_class = RNNModel
        model_params.update(
            dict(
                trainable_embeddings=config.trainable_embeddings,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                output_size=dataset.nb_classes,
                embedding_size=768,
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
    add_dataset_param('vocab', tokenizer.vocab)
    def _truncate_seq(tokens, max_length=120):
        """Truncates a sequence in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if len(tokens) <= max_length:
            return tokens
        else:
            return tokens[:max_length]
       

    if config.model == 'han' or config.model == 'rnn':
        dataset_class = BiosSeqDataset

        bios_train, bios_val, bios_test = load_bios_raw()
        bios_train =  list(map(lambda t: _truncate_seq(tokenizer.tokenize(t)), bios_train)) 
        bios_val =  list(map(lambda t:  _truncate_seq(tokenizer.tokenize(t)), bios_val))
        bios_test =  list(map(lambda t: _truncate_seq(tokenizer.tokenize(t)), bios_test))
        bios_train = list(map(lambda t: ["[CLS]"] + t + ["[SEP]"], bios_train))
        bios_val = list(map(lambda t: ["[CLS]"] + t + ["[SEP]"], bios_val))
        bios_test = list(map(lambda t: ["[CLS]"] + t + ["[SEP]"], bios_test))
        bios_train = list(map(tokenizer.convert_tokens_to_ids, bios_train))
        bios_val = list(map(tokenizer.convert_tokens_to_ids, bios_val))
        bios_test = list(map(tokenizer.convert_tokens_to_ids, bios_test))

        input_masks = [[1] * len(x) for x in bios_train]
        segment_ids = [[0] * len(x) for x in bios_train]
        padding = [[0] * (150 - len(x)) for x in bios_train]
        bios_train = np.array([bios_train[x] + padding[x] for x in range(len(bios_train))])
        input_masks = np.array([input_masks[idx] + padding[idx] for idx in range(len(bios_train))])
        segment_ids = np.array([segment_ids[idx] + padding[idx] for idx in range(len(bios_train))])
        bios_train = np.stack((bios_train, input_masks, segment_ids), axis = 1)

        input_masks = [[1] * len(x) for x in bios_val]
        segment_ids = [[0] * len(x) for x in bios_val]
        padding = [[0] * (150 - len(x)) for x in bios_val]
        bios_val = np.array([bios_val[x] + padding[x] for x in range(len(bios_val))])
        input_masks = np.array([input_masks[idx] + padding[idx] for idx in range(len(bios_val))])
        segment_ids = np.array([segment_ids[idx] + padding[idx] for idx in range(len(bios_val))])
        bios_val = np.stack((bios_val, input_masks, segment_ids), axis = 1)

        input_masks = [[1] * len(x) for x in bios_test]
        segment_ids = [[0] * len(x) for x in bios_test]
        padding = [[0] * (150 - len(x)) for x in bios_test]
        bios_test = np.array([bios_test[x] + padding[x] for x in range(len(bios_test))])
        input_masks = np.array([input_masks[idx] + padding[idx] for idx in range(len(bios_test))])
        segment_ids = np.array([segment_ids[idx] + padding[idx] for idx in range(len(bios_test))])
        bios_test = np.stack((bios_test, input_masks, segment_ids), axis = 1)
        
        add_dataset_param('bios', bios_train, bios_val, bios_test)

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
    #get the input
    bios_train, bios_val, bios_test = load_bios_raw()
    titles_train, titles_val, titles_test = load_titles()

    dataset_train, dataset_val, dataset_test = create_dataset(cfg)

    data_loader_train = create_data_loader(dataset_train,
                                           cfg.batch_size,
                                           shuffle=True,
                                           num_workers=8)
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

    model = create_model(cfg, dataset_train, create_W_emb=False)
    if cfg.resume != "":
        restore_weights(model, cfg.resume)

    model_parameters = get_trainable_parameters(model.parameters())
    optimizer = torch.optim.Adam(model_parameters,
                                 cfg.learning_rate,
                                 weight_decay=cfg.weight_decay,
                                 amsgrad=True)

    criterion = BiosLoss(cfg, classes_weights)

    def update_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        inputs, targets = to_device(batch)
        
        with torch.no_grad():
            inputs_bert_emb, _ = mbertmodel(inputs[:,0,:], \
                token_type_ids = inputs[:,2,:], attention_mask= inputs[:,1,:])
        logits = model(inputs[:,0,:], inputs_bert_emb)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, cfg.max_grad_norm)
        optimizer.step()

        return loss.item()

    def inference_function(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = to_device(batch)
            inputs_bert_emb, _ = mbertmodel(inputs[:,0,:], \
                token_type_ids = inputs[:,2,:], attention_mask= inputs[:,1,:])

            logits = model(inputs[:,0,:], inputs_bert_emb)

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
    _, _, dataset_test = create_dataset(cfg)
    
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
            inputs_bert_emb, _ = mbertmodel(inputs[:,0,:], \
                token_type_ids = inputs[:,2,:], attention_mask= inputs[:,1,:])
            logits = model(inputs[:,0,:], inputs_bert_emb)

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

    model = create_model(cfg, dataset_test, create_W_emb=False)
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
            MODEL_DIR, "pred-" + scrub_flag + '-' + cfg.model_flag + '-T-' +
            cfg.pred_flag + ".pkl")))
    save_pickle(
        os.path.join(
            MODEL_DIR, "pred-" + scrub_flag + '-' + cfg.model_flag + '-T-' +
            cfg.pred_flag + ".pkl"), preds)

if __name__ == '__main__':
    cfg = parse_args(TrainConfig, 'Train model')
    print(cfg)
    if cfg.train_flag == 1:
        main(cfg)
    predict(cfg)