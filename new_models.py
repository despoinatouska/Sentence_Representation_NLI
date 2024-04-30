import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
        
class AWESentenceEncoder(nn.Module):
    def __init__(self, config):
        super(AWESentenceEncoder, self).__init__()
        self.out_dim = config.word_emb_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, sent_tuple):
        input, lens = sent_tuple # input: max_len, batch_size, embedding_dim
        summed = input.sum(0) # batch_size, embedding_dim
        avg = torch.div(summed.transpose(0, -1), lens) #embedding_dim, batch_size
        return avg.transpose(0, -1) # batch_size, embedding_dim

class BLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BLSTMEncoder, self).__init__()
        self.bsize = config.batch_size
        self.word_emb_dim = config.word_emb_dim
        self.enc_lstm_dim = config.enc_lstm_dim
        self.pool_type = config.pool_type
        self.dpout_model = config.dpout_model
        self.out_dim = 2 * self.enc_lstm_dim

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple # sent: seqlen, batch, worddim
        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = torch.sort(sent_len, descending=True)
        idx_unsort = torch.argsort(idx_sort)
        sent = torch.index_select(sent, 1, Variable(idx_sort))

        # Handling padding 
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, list(sent_len_sorted))
        sent_output, (h_n, c_n) = self.enc_lstm(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0] # output: batch, seqlen, 2*hid
        sent_output = torch.index_select(sent_output, 0, Variable(idx_unsort))

        if self.pool_type == "last":
            fwd_hidden = h_n[0] # batch, hid
            bwd_hidden = h_n[1] # batch, hid
            if len(list(fwd_hidden.shape)) > 2:
                fwd_hidden = fwd_hidden.squeeze(dim=0)
                bwd_hidden = bwd_hidden.squeeze(dim=0)
            emb = torch.cat([fwd_hidden, bwd_hidden], dim=-1) # batch, 2*hid
            emb = torch.index_select(emb, 0, Variable(idx_unsort))

        elif self.pool_type == "max":
            sent_output = [x[:l] for x, l in zip(sent_output, sent_len)] # batch, seqlen (varies based on seq len), 2*hid
            emb = [torch.max(x, 0)[0] for x in sent_output] # find max over the sequence. batch, 2*hid
            emb = torch.stack(emb, 0)

        return emb

class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config.batch_size
        self.word_emb_dim = config.word_emb_dim
        self.enc_lstm_dim = config.enc_lstm_dim
        self.pool_type = config.pool_type
        self.dpout_model = config.dpout_model
        self.out_dim = self.enc_lstm_dim

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=False, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        self.init_lstm = Variable(torch.FloatTensor(1, bsize, self.out_dim).zero_()).to(sent.device)

        sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent = torch.index_select(sent, 1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, list(sent_len))
        sent_output = self.enc_lstm(sent_packed, (self.init_lstm,
                      self.init_lstm))[1][0].squeeze(0)  # batch x 2*nhid
        
        # Un-sort by length
        idx_unsort = torch.argsort(idx_sort)
        emb = torch.index_select(sent_output, 0, idx_unsort)

        return emb

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
        self.nonlinear_fc = config.nonlinear_fc
        self.fc_dim = config.fc_dim
        self.n_classes = 3
        self.enc_lstm_dim = config.enc_lstm_dim
        self.encoder_type = config.encoder_type
        self.dpout_fc = config.dpout_fc
        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4 * self.encoder.out_dim

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1) # bsx4x300 AWE, bsx4x2*hid BLSTM, bsx4xhid BLSTM
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
