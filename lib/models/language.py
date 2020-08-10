import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_numpy, to_torch


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True, pretrain=False):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        if pretrain is True:
            embedding_mat = np.load('./data/word_embedding/embed_matrix.npy')
            self.embedding = nn.Embedding.from_pretrained(to_torch(embedding_mat).cuda(), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)

            # make ixs
            input_lengths_list = to_numpy(input_lengths).tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            max_length = sorted_input_lengths_list[0]
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long().cuda()
            recover_ixs = input_labels.data.new(recover_ixs).long().cuda()

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs, 0:max_length].long().cuda()
            assert max(input_lengths_list) == input_labels.size(1)

        # embed
        embedded = self.embedding(input_labels)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_embedding_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]
            hidden = hidden[:, recover_ixs, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)

        return output, hidden, embedded, max_length


class ModuleInputAttention(nn.Module):
    def __init__(self, input_dim):
        super(ModuleInputAttention, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, input_labels):
        # context(bs, num_seq, max_length, dim_cxt); input_labels(bs, num_seq, max_length)
        bs, num_seq, max_length = context.size(0), context.size(1), context.size(2)
        context = context.view(bs*num_seq, max_length, -1)
        embedded = embedded.view(bs*num_seq, max_length, -1)
        input_labels = input_labels.view(bs*num_seq, max_length)

        cxt_scores = self.fc(context).squeeze(2)
        attn = F.softmax(cxt_scores, dim=1)

        # mask zeros
        is_not_zero = (input_labels != 0).float()
        attn = attn * is_not_zero
        attn_sum = attn.sum(1).unsqueeze(1).expand(attn.size(0), attn.size(1))
        attn[attn_sum!=0] = attn[attn_sum!=0] / attn_sum[attn_sum!=0]

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)
        weighted_emb = weighted_emb.squeeze(1)

        weighted_emb = weighted_emb.view(bs, num_seq, -1)

        return weighted_emb, attn