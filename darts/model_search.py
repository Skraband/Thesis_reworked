import torch
import torch.nn as nn
import torch.nn.functional as F

import model.wein.EinsumNetwork.EinsumNetwork
from .operations import *
from torch.autograd import Variable
from .genotypes import PWN_PRIMITIVES, PRIMITIVES
from .genotypes import Genotype
from model.wein.EinsumNetwork import EinsumNetwork

class MixedOp(nn.Module):

    def __init__(self, spectral_transfromer, srnn, transformer, wein, cwspn):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        #self._isSpectralElement = spectral_transfromer
        numberOfModuls = 2
        self.numberOfModuls = numberOfModuls

        for i in range(numberOfModuls):

            if spectral_transfromer:
                if i == 0:
                    op = transformer  # PWN_OPS['spectral_transformer'](config,config_t,config_w,config_c)
                elif i == 1:
                    op = srnn  # PWN_OPS['spectral_rnn'](config,config_t,config_w,config_c)
            else:
                if i == 0:
                    op = cwspn  # PWN_OPS['cwspn'](config,config_t,config_w,config_c)
                elif i == 1:
                    op = wein  # PWN_OPS['wein'](config,config_t,config_w,config_c)

            self._ops.append(op)

    def forward(self, batch_x, batch_y, weights,srnn_arch_weights, spectral_transfomer, spn, spn_2,turn_off_wein):
        out_1 = 0
        out_2 = 0

        if spectral_transfomer:
            pred_trans, f_c_trans = self._ops[0](batch_x, batch_y, return_coefficients=True)

            if srnn_arch_weights is None:
                pred_srnn, f_c_srnn = self._ops[1](batch_x, batch_y, return_coefficients=True)
            else:
                pred_srnn, f_c_srnn = self._ops[1](batch_x, batch_y,srnn_arch_weights, return_coefficients=True)

            out_1 = pred_trans * weights[0][0] + pred_srnn * weights[0][1]
            out_2 = f_c_trans * weights[0][0] + f_c_srnn * weights[0][1]

        else:
            ##### cwspn ######
            y_ = torch.stack([batch_y.real, batch_y.imag], dim=-1) if torch.is_complex(batch_y) else batch_y
            sum_params, leaf_params = self._ops[0](batch_x.reshape((batch_x.shape[0], batch_x.shape[1] *
                                                                    (2 if True else 1),)))
            spn.westimator.args.param_provider.sum_params = sum_params
            spn.westimator.args.param_provider.leaf_params = leaf_params

            predicition_ll_cwspn, w_in_cwspn = spn.westimator.spn(y_), y_

            ##### WEin #######
            y_ = torch.stack([batch_y.real, batch_y.imag], dim=-1) if torch.is_complex(batch_y) else batch_y

            if spn_2 == []:

                val_in = torch.cat([batch_x, y_], dim=1)
                ll_joint = self._ops[1](val_in)
                prediction_ll_wein = EinsumNetwork.log_likelihoods(ll_joint)

                if turn_off_wein:
                    prediction_ll_wein = prediction_ll_wein * 0


            else:
                sum_params_2, leaf_params_2 = self._ops[1](batch_x.reshape((batch_x.shape[0], batch_x.shape[1] *
                                                                        (2 if True else 1),)))
                spn_2.westimator.args.param_provider.sum_params = sum_params_2
                spn_2.westimator.args.param_provider.leaf_params = leaf_params_2

                prediction_ll_wein, w_in_cwspn_2 = spn_2.westimator.spn(y_), y_

            out_1 = predicition_ll_cwspn * weights[0][0] + prediction_ll_wein * weights[0][1]
            out_2 = 0

        return out_1, out_2


class Cell(nn.Module):

    def __init__(self, steps, spectral_transfromer, srnn, transformer, wein, wein_nn, cwspn):
        super(Cell, self).__init__()
        self.spectral_transfromer = spectral_transfromer
        self._steps = steps

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._steps):
            op = MixedOp(spectral_transfromer, srnn, transformer, wein_nn, cwspn)
            self._ops.append(op)

    def forward(self, batch_x, batch_y, weights,srnn_arch_weights, spectral_transfomer, spn, spn_2,turn_off_wein):

        out_1 = 0
        out_2 = 0

        if spectral_transfomer:
            out_1, out_2 = self._ops[0](batch_x, batch_y, weights,srnn_arch_weights, spectral_transfomer, spn, spn_2,turn_off_wein)
        else:
            srnn_arch_weights = None
            out_1, out_2 = self._ops[0](batch_x, batch_y, weights,srnn_arch_weights, spectral_transfomer, spn, spn_2,turn_off_wein)

        return out_1, out_2


class Network(nn.Module):

    def __init__(self, srnn, transformer, wein,wein_nn, cwspn, cwspn_spn,turn_off_wein, layers, criterion,search_srnn=False,search_cwspn=False,smape_target=False, steps=1, multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.cwspn_nn = cwspn_spn
        self.smape_target = smape_target
        self.srnn_arch_weights = None
        self.cwspn_arch_weights = None
        self.search_srnn = search_srnn
        self.search_cwspn = search_cwspn
        self.wein_nn = wein
        self.turn_off_wein = turn_off_wein
        self.weight_mse_by_ll = None

        self.ll_weight = 0.5
        self.ll_weight_inc_dur = 20
        self.train_rnn_w_ll = False

        if search_srnn:
            self.srnn_arch_weights = srnn.weights#srnn.weights.clone()
            #STEPS = 8
            #k = sum(i for i in range(1, STEPS + 1))
            #weights_data = torch.randn(k, 5, requires_grad=True).cuda() #1e-3 *
            #weights_data = Variable(1e-3 * torch.randn(k, 5).cuda(), requires_grad=True)
            #self.srnn_arch_weights = weights_data
        if search_cwspn:
            self.cwspn_arch_weights = cwspn.alphas_normal


        if self.train_rnn_w_ll:
            self.current_ll_weight = 0
            self.ll_weight_history = []
            self.ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        self.cells = nn.ModuleList()
        self.srnn = srnn
        reduction_prev = False

        for i in range(layers):
            if i == 0:
                spectral_transfromer = True
            else:
                spectral_transfromer = False

            # cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            cell = Cell(steps, spectral_transfromer, srnn, transformer, wein, wein_nn, cwspn)

            # reduction_prev = reduction
            self.cells += [cell]
            # C_prev_prev, C_prev = C_prev, multiplier*C_curr

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, batch_x, batch_y, batch_westimator_x_val, batch_westimator_y_val):

        pred = 0
        f_c = 0
        prediction_ll = 0
        w_in = 0
        spn = self.cwspn_nn
        spn_wein = self.wein_nn
        my_f_c = None
        srnn_arch_weights = self.srnn_arch_weights
        for i, cell in enumerate(self.cells):

            if cell.spectral_transfromer:
                weights = F.softmax(self.alphas_normal, dim=-1)
                prediction, f_c = cell(batch_x, batch_y, weights,srnn_arch_weights, spectral_transfomer=True, spn=spn, spn_2 = spn_wein,turn_off_wein = self.turn_off_wein)
                my_f_c = f_c.reshape((f_c.shape[0], -1)).detach()
            else:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                prediction_ll, _ = cell(batch_westimator_x_val, my_f_c, weights,srnn_arch_weights, spectral_transfomer=False, spn=spn, spn_2 = spn_wein,turn_off_wein = self.turn_off_wein)

        return prediction, prediction_ll

    def _loss(self, batch_x, batch_y, batch_westimator_x_val, batch_westimator_y_val):
        # Add smape target crit as soon as everything else runs
        if self.smape_target:
            smape_adjust = 2  # Move all values into the positive space
            p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                                  (torch.abs(out + smape_adjust) +
                                                   torch.abs(label + smape_adjust))).mean(axis=1)
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()

        prediction, prediction_ll = self(batch_x, batch_y, batch_westimator_x_val, batch_westimator_y_val)

        error = p_base_loss(prediction, batch_y)
        p_loss = prediction_loss(error)

        if self.train_rnn_w_ll:
            l_loss = ll_loss(prediction_ll)
            self.ll_weight_history.append(self.current_ll_weight)

            if self.weight_mse_by_ll is None:
                srnn_loss = (1 - self.current_ll_weight) * p_loss + self.current_ll_weight * l_loss
            else:
                local_ll = torch.logsumexp(prediction_ll, dim=1)
                local_ll = local_ll - local_ll.max()  # From 0 to -inf
                local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
                local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

                if self.weight_mse_by_ll == 'het':
                    # Het: low LL is 0, high LL is 1
                    local_ll = local_ll.max() - local_ll

                srnn_loss = p_loss * (self.ll_weight - self.current_ll_weight) + \
                            self.current_ll_weight * (error * local_ll).mean()
        else:
            srnn_loss = p_loss
            l_loss = 0


        westimator_loss = Network.ll_loss_pred(prediction_ll, error.detach())

        return srnn_loss, westimator_loss

    def ll_loss_pred(out, error):
        return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

    def _initialize_alphas(self):
        # k = sum(1 for i in range(self._steps) for n in range(2+i))
        k = sum(1 for i in range(self._steps))
        num_ops = 2  # len(PWN_PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        #self.alphas_reduce.requires_grad = False
        #self.alphas_reduce[0, 0] += 10
        #self.alphas_normal.requires_grad = False
        #self.alphas_normal[0, 1] += 10
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        arch_params = self._arch_parameters
        if self.search_srnn:
            arch_params = arch_params + [self.srnn_arch_weights]
        if self.search_cwspn:
            arch_params = arch_params + [self.cwspn_arch_weights]

        return arch_params

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
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

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
