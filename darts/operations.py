import torch
import torch.nn as nn

#from model import Model
from model.spectral_rnn.spectral_rnn import SpectralRNN
from model.transformer.transformer_predictor import TransformerPredictor
from model.cwspn.cwspn import CWSPN
from model.wein.wein import WEin
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model.cwspn.weight_nn import WeigthNN
from model.wein.EinsumNetwork.EinsumNetwork import EinsumNetwork
from model.transformer.transformer_predictor import TransformerNet
from model.spectral_rnn.spectral_rnn import SpectralRNNNet

import pickle
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
PWN_OPS = {
  'spectral_transformer' : lambda config,config_t, config_w, config_c: createSTrans(config_t),
  'spectral_rnn' : lambda config,config_t, config_w, config_c: createSRNN(config,config_c),
  'cwspn' : lambda config,config_t, config_w, config_c: createCWSPN(config_c),
  'wein' : lambda config,config_t, config_w, config_c: createWEin(config_w),
}

def createSRNN(config, config_c):

  SRNN = SpectralRNN(config)
  SRNN.build_net()
  config_c.stft_module = SRNN.net.stft
  return SRNN.net

def createSTrans(config):

  STrans = TransformerPredictor(config)
  STrans.build_net()

  return STrans.net

def createCWSPN(config_c):

  CW_SPN = CWSPN(config_c)
  CW_SPN.stft_module = config_c.stft_module
  x, y = CW_SPN.prepare_input(config_c.x_, config_c.y_)
  CW_SPN.input_sizes = x.shape[1], y.shape[1]
  CW_SPN.create_net()

  return CW_SPN.net

def createWEin(config_w):

  W_Ein = WEin(config_w)
  W_Ein.create_net()

  return W_Ein.net

# ################################################## Variante 2 ##########################################################
# # Dont forget sending net to device!
#
# def buildSRNN(config):
#
#   net = SpectralRNNNet(config)
#
#   return net
#
# def buildSTrans(config):
#
#   in_factor = 1 if complex else 2
#   net = TransformerNet(config, config.input_dim * in_factor, config.hidden_dim,
#                                   config.input_dim * in_factor, config.q, config.k, config.heads,
#                                   config.num_enc_dec, attention_size=config.attention_size,
#                                   dropout=config.dropout, chunk_mode=config.chunk_mode, pe=config.pe,
#                                   complex=config.is_complex, native_complex=config.native_complex)
#
#   return net
#
#
#
# def buildCWSPN(config):
#
#   input_sizes = 0 #based on x and y      input input_sizes = x.shape[1], y.shape[1] after prepare input of concate (x_ ; y_)
#   inp_size = input_sizes[0] * (2 if config.use_stft else 1)
#   num_sum_params = config.num_sum_paarams
#   num_leaf_params = config.num_leaf_params
#
#   #WeigthNN(self.input_sizes[0] * (2 if self.use_stft else 1), self.num_sum_params,
#   #         self.num_leaf_params, self.config.use_rationals)
#
#   net = WeigthNN(inp_size, num_sum_params, num_leaf_params, use_rationals=config.use_rationals)
#
#   return net
#
#
#
# def buildWEin():
#
#   net = EinsumNetwork(graph, args=None)
#
#   return
#
# ########################################################################################################################

# OPS = {
#   'none' : lambda C, stride, affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#   'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#   'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#   'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#   'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#   'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#     nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#     ),
# }

# class ReLUConvBN(nn.Module):
#
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(ReLUConvBN, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine)
#     )
#
#   def forward(self, x):
#     return self.op(x)
#
# class DilConv(nn.Module):
#
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#     super(DilConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )
#
#   def forward(self, x):
#     return self.op(x)
#
#
# class SepConv(nn.Module):
#
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(SepConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_in, affine=affine),
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )
#
#   def forward(self, x):
#     return self.op(x)
#
#
# class Identity(nn.Module):
#
#   def __init__(self):
#     super(Identity, self).__init__()
#
#   def forward(self, x):
#     return x
#
#
# class Zero(nn.Module):
#
#   def __init__(self, stride):
#     super(Zero, self).__init__()
#     self.stride = stride
#
#   def forward(self, x):
#     if self.stride == 1:
#       return x.mul(0.)
#     return x[:,:,::self.stride,::self.stride].mul(0.)
#
#
# class FactorizedReduce(nn.Module):
#
#   def __init__(self, C_in, C_out, affine=True):
#     super(FactorizedReduce, self).__init__()
#     assert C_out % 2 == 0
#     self.relu = nn.ReLU(inplace=False)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.bn = nn.BatchNorm2d(C_out, affine=affine)
#
#   def forward(self, x):
#     x = self.relu(x)
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.bn(out)
#     return out
#
