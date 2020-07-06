import torch
import torch.nn.functional as F

from rnn_encoder import GRUEncoder
from utils import get_sequences_lengths, softmax_masked, to_device

class CountsModel(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.classifier = torch.nn.Linear(input_size, output_size)

    def forward(self, inputs):
        if self.dropout != 0:
            inputs = F.dropout(inputs, self.dropout, self.training)

        logits = self.classifier(inputs)
        return logits



class RNNModel(torch.nn.Module):
    def __init__(self, embedding_size, vocab_size, trainable_embeddings, hidden_size, output_size, dropout, W_emb=None,
                 padding_idx=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder_sentences = GRUEncoder(embedding_size, hidden_size, bidirectional=True, return_sequence=False)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, output_size),
        )

    def zero_state(self, batch_size):
        state_shape = (2, batch_size, self.hidden_size)

        # will work on both GPU and CPU in contrast to just Variable(*state_shape)
        h = to_device(torch.zeros(*state_shape))
        return h

    def encode(self, inputs):
        inputs_len = get_sequences_lengths(inputs)

        inputs_emb = self.embedding(inputs)
        inputs_enc = self.encoder_sentences(inputs_emb, inputs_len)
        inputs_enc = F.dropout(inputs_enc, self.dropout, self.training)

        return inputs_enc

    def get_logits(self, inputs_att):
        logits = self.out(inputs_att)
        return logits

    def forward(self, inputs):
        inputs_enc = self.encode(inputs)
        logits = self.get_logits(inputs_enc)

        return logits



class HANModel(torch.nn.Module):
    def __init__(self, embedding_size, vocab_size, trainable_embeddings, hidden_size, attention_size, output_size, dropout, W_emb=None,
                 padding_idx=0):
        super().__init__()

        self.padding_idx = padding_idx
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

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

    def encode_sentences(self, inputs):
        mask = inputs != self.padding_idx
        inputs_len = get_sequences_lengths(inputs)

        inputs_emb = self.embedding(inputs)
        inputs_enc = self.encoder_sentences(inputs_emb, inputs_len)
        inputs_enc = F.dropout(inputs_enc, self.dropout, self.training)

        mask = mask[:, :inputs_enc.size(1)]

        att_vec = self.att_sentences(inputs_enc)
        att_weights = self.att_reduce(att_vec)
        att = softmax_masked(att_weights, mask.unsqueeze(-1))

        inputs_att = torch.sum(inputs_enc * att, dim=1)
        inputs_att = F.dropout(inputs_att, self.dropout, self.training)

        return inputs_att, att

    def get_logits(self, inputs_att):
        logits = self.out(inputs_att)
        return logits

    def forward(self, inputs):
        inputs_att, att = self.encode_sentences(inputs)
        logits = self.get_logits(inputs_att)

        return logits
