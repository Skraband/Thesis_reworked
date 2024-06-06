import torch
import torch.nn as nn
import torch.nn.functional as F
#from operations import *
from torch.autograd import Variable
#from genotypes import PRIMITIVES
#from genotypes import Genotype
from darts.darts_cnn.operations import *
from darts.darts_cnn.genotypes import PRIMITIVES
from darts.darts_cnn.genotypes import Genotype
class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      #if 'pool' in primitive:
      #  op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps,in_seq_length):
    super(Cell, self).__init__()
    self._steps = steps
    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):

      op = MixedOp(C=1, stride=1)
      self._ops.append(op)

  def forward(self, input, weights):

    states = [input]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[i](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.mean(torch.cat(states[-4:], dim=1), dim=1)


class Network(nn.Module):

  def __init__(self,in_seq_length, output_length,sum_params, layers, steps=4):
    super(Network, self).__init__()
    self.output_length = output_length
    self.in_seq_length = in_seq_length
    self._layers = layers
    self._steps = steps
    self._sum_params = sum_params
    self.cells = nn.ModuleList()

    for i in range(layers):
      cell = Cell(steps, in_seq_length)
      self.cells += [cell]

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.lin_layer = nn.Linear(in_seq_length, output_length)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    input_aux = input.unsqueeze(1)
    for i, cell in enumerate(self.cells):
      weights = F.softmax(self.alphas_normal, dim=-1)
      out_aux = cell(input_aux, weights)
    #out = self.global_pooling(s1)
    #logits = self.classifier(out.view(out.size(0),-1))

    #out_aux_2 = out_aux.squeez(1)
    out = self.lin_layer(out_aux)
    sum_params = out[:, :self._sum_params]
    leaf_params = out[:, self._sum_params:]
    return sum_params , leaf_params

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

