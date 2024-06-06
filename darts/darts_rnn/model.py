import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.darts_rnn.genotypes import STEPS
from darts.darts_rnn.utils import mask2d
from darts.darts_rnn.utils import LockedDropout
from darts.darts_rnn.utils import embedded_dropout
from torch.autograd import Variable

from model.spectral_rnn.cgRNN import RNNLayer

INITRANGE = 0.04
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DARTSCell(nn.Module):

  def __init__(self,config_layer,fix_weight,ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.fix_weight = fix_weight
    # genotype is None when doing arch search
    steps = STEPS

    if config_layer.rnn_layer_config.use_gated:
        cell = torch.nn.GRUCell
    self.layer = nn.ModuleList([
        RNNLayer(cell, config_layer) for i in range(steps)
    ])

  def forward(self, inputs,srnn_arch_weights):
    T, B = inputs.size(0), inputs.size(1)

    #if self.training:
    #  x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
    #  h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    #else:
    #  x_mask = h_mask = None

    #hidden = hidden[0]
    hiddens = []
    #for t in range(T):
    hidden = self.cell(inputs,srnn_arch_weights)
    hiddens.append(hidden)
    #hiddens = torch.stack(hiddens)

    return hiddens, hidden #hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.recurrent):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      #fn = self._get_activation(name)

      if name == "tanh":
          h = torch.tanh(h)
      elif name == "relu":
          h = torch.relu(h)
      elif name == "sigmoid":
          h = torch.sigmoid(h)
      elif name == "identity":
          h = h
      else:
          raise NotImplementedError

      #h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,stft,config_layer, ntoken, ninp, nhid, nhidlast,
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                 cell_cls=DARTSCell, genotype=None):
        super(RNNModel, self).__init__()
        #self.lockdrop = LockedDropout()
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.stft = stft
        self.config = config_layer
        self.fix_weight = False

        if config_layer.use_only_ts_input:
            self.f_in = nn.ModuleList([self.stft])

        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(config_layer,self.fix_weight,ninp, nhid, dropouth, dropoutx, genotype)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(config_layer,self.fix_weight,ninp, nhid, dropouth, dropoutx)]

        steps = STEPS

        self.rnns = torch.nn.ModuleList(self.rnns)

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls
        self.amt_prediction_samples = None
        self.amt_prediction_windows = None

    def init_weights(self):
        #self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        #self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        return

    def forward(self, x_in, y_in, srnn_arch_weights, return_coefficients=True):
        #batch_size = x_in.size(0)
        #emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.lockdrop(emb, self.dropouti)

        if self.config.use_only_ts_input:
            x = [self.f_in[0](x_in[:, :, -1])]
        else:
            x = [f_in_(x_in[:, :, i].long()) if i in [1, 2] else (
                f_in_(x_in[:, :, i:i + 1].float() / 100.) if i == 0 else f_in_(x_in[:, :, i])) for i, f_in_ in
                 enumerate(self.f_in)]

        stft_len = x[-1].shape[1]

        x[-1] = x[-1].swapaxes(-2, -1)

        encoder_in = x[-1]
        if self.amt_prediction_samples is None or self.amt_prediction_windows is None:
            self.amt_prediction_samples = y_in.shape[1]
            self.amt_prediction_windows = self.stft(y_in).shape[-1]

        decoder_in = torch.zeros([encoder_in.shape[0], self.amt_prediction_windows, encoder_in.shape[-1]]).to(device)

        gru_in = torch.cat([encoder_in, decoder_in], dim=-2).type(torch.complex64) \
            .to(device)

        gru_in_extended = gru_in.clone()

        if not self.config.rnn_layer_config.use_cg_cell:
            gru_in_extended = torch.cat([gru_in_extended.real, gru_in_extended.imag], dim=-1)

        raw_output = gru_in_extended#gru_in_extended.clone()
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output,srnn_arch_weights)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden
        gru_out = hidden[0]#hidden[0].clone()

        decoder_out = gru_out[:, -self.amt_prediction_windows:].clone()
        if not self.config.rnn_layer_config.use_cg_cell:
            decoder_out = torch.complex(decoder_out[:, :, :stft_len], decoder_out[:, :, stft_len:])

        if self.config.use_add_linear:
            decoder_out = self.add_pre_act(decoder_out)
            decoder_out = self.add_linear(decoder_out)

        decoder_out_ = decoder_out.swapaxes(-2, -1)

        out = self.stft(decoder_out_, reverse=True)[:, -self.amt_prediction_samples:]

        if return_coefficients:
            return out, decoder_out
        else:
            return out

        #output = self.lockdrop(raw_output, self.dropout)
        #outputs.append(output)

        #logit = self.decoder(output.view(-1, self.ninp))
        #log_prob = nn.functional.log_softmax(logit, dim=-1)
        #model_output = log_prob
        #model_output = model_output.view(-1, batch_size, self.ntoken)

        #if return_coefficients:
        #    return model_output, hidden, raw_outputs, outputs
        #return model_output, hidden

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]

