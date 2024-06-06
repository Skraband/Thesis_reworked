from .spectral_rnn.spectral_rnn import SpectralRNN, SpectralRNNConfig
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig
from .maf import MAFEstimator
from .cwspn import CWSPN, CWSPNConfig
from .model import Model

import numpy as np
import torch
import torch.nn as nn
#from darts.darts_cnn.model_search import Network as CWSPNModelSearch
#from darts.darts_rnn.model_search import RNNModelSearch
# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_arch_params(srnn=True):

    if srnn:
        arch_params_raw = np.array([[0.13259780406951904, 0.24076713621616364, 0.2534853518009186, 0.11249035596847534, 0.2606593370437622], [0.1440565586090088, 0.25148487091064453, 0.2283712923526764, 0.11011575162410736, 0.2659715712070465], [0.14229190349578857, 0.23250094056129456, 0.2703188359737396, 0.10851601511240005, 0.2463722676038742], [0.18138743937015533, 0.22439733147621155, 0.22608043253421783, 0.12540064752101898, 0.2427341789007187], [0.17366157472133636, 0.20776717364788055, 0.27516642212867737, 0.12462981045246124, 0.2187749594449997], [0.15872301161289215, 0.2027873545885086, 0.2700975239276886, 0.12379473447799683, 0.24459731578826904], [0.20973490178585052, 0.2041788399219513, 0.2237374484539032, 0.14984506368637085, 0.21250379085540771], [0.202859565615654, 0.1950104534626007, 0.2561758756637573, 0.14783424139022827, 0.1981198638677597], [0.18905547261238098, 0.19409416615962982, 0.2546370029449463, 0.1461944729089737, 0.2160188853740692], [0.18173353374004364, 0.17546316981315613, 0.2516404390335083, 0.14580832421779633, 0.2453545480966568], [0.20544281601905823, 0.20635414123535156, 0.21132592856884003, 0.16771581768989563, 0.20916123688220978], [0.2035631686449051, 0.20132912695407867, 0.2265666425228119, 0.16620858013629913, 0.2023324817419052], [0.19599156081676483, 0.2021075338125229, 0.22765564918518066, 0.16463622450828552, 0.2096090316772461], [0.18935491144657135, 0.18910448253154755, 0.23065787553787231, 0.16415654122829437, 0.2267262488603592], [0.1890873908996582, 0.1611195057630539, 0.24223291873931885, 0.16403093934059143, 0.24352917075157166], [0.20131433010101318, 0.20656262338161469, 0.2042701691389084, 0.18064041435718536, 0.20721252262592316], [0.1989413946866989, 0.20374609529972076, 0.21427693963050842, 0.1789049506187439, 0.20413054525852203], [0.19395166635513306, 0.2041616588830948, 0.21656383574008942, 0.17762456834316254, 0.2076983004808426], [0.19034822285175323, 0.19616955518722534, 0.21852216124534607, 0.17832164466381073, 0.21663837134838104], [0.1879197657108307, 0.1743212342262268, 0.23005910217761993, 0.17745517194271088, 0.2302446961402893], [0.1866140067577362, 0.15288281440734863, 0.24549749493598938, 0.16948345303535461, 0.24552226066589355], [0.20201094448566437, 0.20353685319423676, 0.20422334969043732, 0.18655401468276978, 0.20367483794689178], [0.20031118392944336, 0.20291019976139069, 0.20830510556697845, 0.1853429079055786, 0.20313063263893127], [0.19729267060756683, 0.20349645614624023, 0.2094917744398117, 0.18428833782672882, 0.2054307758808136], [0.19602423906326294, 0.19844990968704224, 0.21085995435714722, 0.18465517461299896, 0.21001076698303223], [0.1963476687669754, 0.18369664251804352, 0.21694990992546082, 0.1859038919210434, 0.21710191667079926], [0.19308514893054962, 0.16640199720859528, 0.2299496829509735, 0.18042123317718506, 0.23014190793037415], [0.18596242368221283, 0.1584322154521942, 0.24344603717327118, 0.16855596005916595, 0.24360328912734985], [0.20251137018203735, 0.2021455019712448, 0.20280678570270538, 0.1903916299343109, 0.20214466750621796], [0.2013077586889267, 0.2022128850221634, 0.20459626615047455, 0.18957774341106415, 0.2023053616285324], [0.19943907856941223, 0.2027737945318222, 0.20541439950466156, 0.18858157098293304, 0.20379118621349335], [0.19946148991584778, 0.19958913326263428, 0.2064257711172104, 0.1885538250207901, 0.20596981048583984], [0.20225463807582855, 0.1896560937166214, 0.20870020985603333, 0.19067268073558807, 0.20871639251708984], [0.20221146941184998, 0.17783591151237488, 0.21531134843826294, 0.18932873010635376, 0.21531252562999725], [0.19495221972465515, 0.17052757740020752, 0.22705817222595215, 0.18026337027549744, 0.22719866037368774], [0.19058695435523987, 0.17226488888263702, 0.2315262109041214, 0.1740020513534546, 0.23161986470222473]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    else:
        arch_params_raw = np.array([[0.12599501013755798, 0.32322800159454346, 0.11491930484771729, 0.1612580418586731, 0.1243358924984932, 0.15026375651359558], [0.1389445960521698, 0.28792625665664673, 0.10612446069717407, 0.1558292806148529, 0.16506774723529816, 0.14610762894153595], [0.16202902793884277, 0.22440101206302643, 0.13319498300552368, 0.1553414911031723, 0.16785161197185516, 0.15718185901641846], [0.15680372714996338, 0.26380687952041626, 0.13456235826015472, 0.15701892971992493, 0.14242208003997803, 0.14538602530956268], [0.16321887075901031, 0.2002309411764145, 0.1338645964860916, 0.17844876646995544, 0.1609017252922058, 0.1633351445198059], [0.166190043091774, 0.2158687561750412, 0.13191263377666473, 0.17276370525360107, 0.155690535902977, 0.15757431089878082], [0.16268931329250336, 0.23585893213748932, 0.18545731902122498, 0.15871010720729828, 0.08522351086139679, 0.17206092178821564], [0.16955330967903137, 0.18751516938209534, 0.18044252693653107, 0.16271042823791504, 0.11133356392383575, 0.18844495713710785], [0.16884009540081024, 0.19716575741767883, 0.18785230815410614, 0.1649021953344345, 0.09780135750770569, 0.18343821167945862], [0.16892416775226593, 0.22190886735916138, 0.18126562237739563, 0.15215392410755157, 0.09399005025625229, 0.18175740540027618], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.16666540503501892, 0.16666540503501892, 0.16667300462722778, 0.16666540503501892, 0.16666540503501892, 0.16666540503501892], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    return arch_params_torch

class PWN(Model):

    def __init__(self, s_config: SpectralRNNConfig, c_config: CWSPNConfig, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False, use_maf=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        self.srnn = SpectralRNN(s_config) if not use_transformer else TransformerPredictor(
            TransformerConfig(normalize_fft=True, window_size=s_config.window_size,
                              fft_compression=s_config.fft_compression))
        self.westimator = CWSPN(c_config) if not use_maf else MAFEstimator()

        self.train_spn_on_gt = train_spn_on_gt
        self.train_spn_on_prediction = train_spn_on_prediction
        self.train_rnn_w_ll = train_rnn_w_ll
        self.weight_mse_by_ll = weight_mse_by_ll
        self.always_detach = always_detach

        self.westimator_early_stopping = westimator_early_stopping
        self.westimator_stop_threshold = westimator_stop_threshold
        self.westimator_final_learn = westimator_final_learn
        self.ll_weight = ll_weight
        self.ll_weight_inc_dur = ll_weight_inc_dur
        self.step_increase = step_increase
        self.use_transformer = use_transformer
        self.use_maf = use_maf
        self.smape_target = smape_target

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=70, lr=0.004, lr_decay=0.97):

        # TODO: Adjustment for complex optimization, needs documentation
        if type(self.srnn) == TransformerPredictor:
            lr /= 10
        elif self.srnn.config.rnn_layer_config.use_cg_cell:
            lr /= 4

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        if self.srnn.final_amt_pred_samples is None:
            self.srnn.final_amt_pred_samples = next(iter(y_in.values())).shape[1]

        # Init SRNN
        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)
        self.srnn.config.embedding_sizes = embedding_sizes
        self.srnn.build_net()

        self.westimator.stft_module = self.srnn.net.stft

        if self.srnn.config.use_searched_srnn:
            from darts.darts_rnn.model_search import RNNModelSearch
            search_stft = self.srnn.net.stft
            emsize = 300
            nhid = 300
            nhidlast = 300
            ntokens = 10000
            dropout = 0
            dropouth = 0
            dropouti = 0
            dropoute = 0
            dropoutx = 0
            config_layer = self.srnn.config
            srnn_search = RNNModelSearch(search_stft, config_layer, ntokens, emsize, nhid, nhidlast,
                                     dropout, dropouth, dropoutx, dropouti, dropoute).cuda()
            # set arch weights according to searched cell
            arch_params = load_arch_params(True)
            srnn_search.fix_arch_params(arch_params)
            # set searched srnn arch as srnn net
            self.srnn.net = srnn_search
        # Init westimator
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(x[:1, :, -1], y[:1])
        self.westimator.input_sizes = westimator_x_prototype.shape[1], westimator_y_prototype.shape[1]
        self.westimator.create_net()

        if self.westimator.config.use_searched_cwspn:
            from darts.darts_cnn.model_search import Network as CWSPNModelSearch
            in_seq_length = self.westimator.input_sizes[0] * (
                2 if self.westimator.use_stft else 1)  # input sequence length into the WeightNN
            output_length = self.westimator.num_sum_params + self.westimator.num_leaf_params  # combined length of sum and leaf params
            sum_params = self.westimator.num_sum_params
            cwspn_weight_nn_search = CWSPNModelSearch(in_seq_length, output_length, sum_params, layers=1, steps=4).cuda()
            self.westimator.weight_nn = cwspn_weight_nn_search


        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        if self.smape_target:
            smape_adjust = 2  # Move all values into the positive space
            p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                                  (torch.abs(out + smape_adjust) +
                                                   torch.abs(label + smape_adjust))).mean(axis=1)
        # MSE target
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        srnn_parameters = list(self.srnn.net.parameters())
        westimator_parameters = self.westimator.parameters()

        amt_param = sum([p.numel() for p in self.srnn.net.parameters()])
        amt_param_w = sum([p.numel() for p in self.westimator.parameters()])

        srnn_optimizer = ManifoldOptimizer(srnn_parameters, lr, torch.optim.RMSprop, alpha=0.9) \
            if type(self.srnn.config) == SpectralRNNConfig and self.srnn.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(srnn_parameters, lr=lr, alpha=0.9)
        westimator_optimizer = torch.optim.Adam(westimator_parameters, lr=1e-4)

        if self.train_rnn_w_ll:
            current_ll_weight = 0
            ll_weight_history = []
            ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        elif self.train_spn_on_prediction:
            def ll_loss_pred(out, error):
                return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=srnn_optimizer, gamma=lr_decay)

        westimator_losses = []
        srnn_losses = []
        srnn_losses_p = []
        srnn_losses_ll = []

        stop_cspn_training = False
        westimator_patience_counter = 0
        westimator_losses_epoch = []

        self.srnn.net.train()

        if hasattr(self.westimator, 'spn'):
            self.westimator.spn.train()
            self.westimator.weight_nn.train()
        else:
            self.westimator.model.train()

        if self.srnn.config.use_cached_predictions:
            import pickle
            with open('srnn_train.pkl', 'rb') as f:
                all_predictions, all_f_cs = pickle.load(f)

                all_predictions = torch.from_numpy(all_predictions).to(device)
                all_f_cs = torch.from_numpy(all_f_cs).to(device)

        val_errors = []
        val_smape_errors = []
        print(f'Starting Training of {self.identifier} model')
        for epoch in range(epochs):
            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            #if epoch % 2:
            #    model_base_path = 'res/models/'
            #    model_name = f'{self.identifier}-{str(epoch)}'
            #    model_filepath = f'{model_base_path}{model_name}'
            #    self.save(model_filepath)

            if self.train_rnn_w_ll:
                ll_weight_history.append(current_ll_weight)

            srnn_loss_p_e = 0
            srnn_loss_ll_e = 0

            pi = torch.tensor(np.pi)
            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x[:, :, -1], batch_y)

                if self.srnn.config.use_cached_predictions:
                    batch_p = all_predictions.detach().clone()[idx]
                    batch_fc = all_f_cs.detach().clone()[idx]

                if self.train_spn_on_gt:
                    westimator_optimizer.zero_grad()
                    if not stop_cspn_training or epoch >= epochs - self.westimator_final_learn:
                        out_w, _ = self.call_westimator(batch_westimator_x, batch_westimator_y)

                        if hasattr(self.westimator, 'spn'):
                            gt_ll = out_w
                            westimator_loss = ll_loss(gt_ll)
                            westimator_loss.backward()
                            westimator_optimizer.step()
                        else:
                            if self.westimator.use_made:
                                raise NotImplementedError  # MADE not implemented here

                            u, log_det = out_w

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss)

                            negloglik_loss.backward()
                            westimator_loss = negloglik_loss.item()
                            westimator_optimizer.step()
                            westimator_optimizer.zero_grad()

                    else:
                        westimator_loss = westimator_losses_epoch[-1]

                    westimator_loss_e += westimator_loss.detach()

                # Also zero grads for westimator, s.t. old grads dont influence the optimization
                srnn_optimizer.zero_grad()
                westimator_optimizer.zero_grad()

                if not self.srnn.config.use_cached_predictions:
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                        return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                else:
                    prediction, f_c = batch_p, batch_fc

                if self.westimator.use_stft:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape((f_c.shape[0], -1))
                                        if self.train_rnn_w_ll and not self.always_detach else f_c.reshape((f_c.shape[0], -1)).detach())
                else:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, prediction
                                        if self.train_rnn_w_ll and not self.always_detach else prediction.detach())

                #if type(self.srnn) == TransformerPredictor:
                #    num_to_cut = int(prediction_raw.shape[1]/3)
                #    prediction = prediction_raw[:,num_to_cut:]
                #else:
                #    prediction = prediction_raw

                error = p_base_loss(prediction, batch_y)
                p_loss = prediction_loss(error)

                if self.train_rnn_w_ll:
                    l_loss = ll_loss(prediction_ll)

                    if self.weight_mse_by_ll is None:
                        srnn_loss = (1 - current_ll_weight) * p_loss + current_ll_weight * l_loss
                    else:
                        local_ll = torch.logsumexp(prediction_ll, dim=1)
                        local_ll = local_ll - local_ll.max()  # From 0 to -inf
                        local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
                        local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

                        if self.weight_mse_by_ll == 'het':
                            # Het: low LL is 0, high LL is 1
                            local_ll = local_ll.max() - local_ll

                        srnn_loss = p_loss * (self.ll_weight - current_ll_weight) + \
                                    current_ll_weight * (error * local_ll).mean()
                else:
                    srnn_loss = p_loss
                    l_loss = 0

                if not self.srnn.config.use_cached_predictions:
                    srnn_loss.backward()

                    if self.srnn.config.clip_gradient_value > 0:
                        clip_grad_value_complex_(srnn_parameters, self.srnn.config.clip_gradient_value)

                    srnn_optimizer.step()

                if self.train_spn_on_prediction:
                    if hasattr(self.westimator, 'spn'):
                        westimator_loss = ll_loss_pred(prediction_ll, error.detach())

                        westimator_loss.backward()
                        westimator_optimizer.step()
                    else:
                        if type(prediction_ll) == tuple:
                            u, log_det = prediction_ll

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss * (error ** -2)) * 1e-4

                        else:
                            mu, logp = torch.chunk(prediction_ll, 2, dim=1)
                            u = (w_in - mu) * torch.exp(0.5 * logp)

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * w_in.shape[1] * np.log(2 * pi)
                            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                            negloglik_loss = torch.mean(negloglik_loss)

                        negloglik_loss.backward()
                        westimator_loss = negloglik_loss.item()
                        westimator_optimizer.step()

                    westimator_loss_e += westimator_loss.detach()

                l_loss = l_loss.detach() if not type(l_loss) == int else l_loss
                srnn_loss_p_e += p_loss.item()
                srnn_loss_ll_e += l_loss
                srnn_loss_e += srnn_loss.item()

                westimator_losses.append(westimator_loss.detach().cpu().numpy())
                srnn_losses.append(srnn_loss.detach().cpu().numpy())
                srnn_losses_p.append(p_loss.detach().cpu().numpy())
                srnn_losses_ll.append(l_loss)

                if (i + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(idx_batches)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            lr_scheduler.step()

            if epoch < self.ll_weight_inc_dur and self.train_rnn_w_ll:
                if self.step_increase:
                    current_ll_weight = 0
                else:
                    current_ll_weight += ll_weight_increase
            elif self.train_rnn_w_ll:
                current_ll_weight = self.ll_weight

            westimator_loss_epoch = westimator_loss_e / len(idx_batches)
            srnn_loss_epoch = srnn_loss_e / len(idx_batches)
            print(f'Epoch {epoch + 1} / {epochs} done.'
                  f'Avg. WCSPN Loss: {westimator_loss_epoch} '
                  f'Avg. SRNN Loss: {srnn_loss_epoch}')

            print(f'Avg. SRNN-Prediction-Loss: {srnn_loss_p_e / len(idx_batches)}')
            print(f'Avg. SRNN-LL-Loss: {srnn_loss_ll_e / len(idx_batches)}')

            if len(westimator_losses_epoch) > 0 and not stop_cspn_training and \
                    not westimator_loss_epoch < westimator_losses_epoch[-1] - self.westimator_stop_threshold and not \
                    self.train_spn_on_prediction:
                westimator_patience_counter += 1

                print(f'Increasing patience counter to {westimator_patience_counter}')

                if westimator_patience_counter >= self.westimator_early_stopping:
                    stop_cspn_training = True
                    print('WCSPN training stopped!')

            else:
                westimator_patience_counter = 0

            westimator_losses_epoch.append(westimator_loss_epoch)

            if True and epoch % 100 == 0:
                pred_val, _ = self.predict({key: x for key, x in val_x.items() if len(x) > 0}, mpe=False)
                self.srnn.net.train()
                self.westimator.spn.train()
                self.westimator.weight_nn.train()

                val_mse = np.mean([((p - val_y[key][:, :, -1]) ** 2).mean() for key, p in pred_val.items()])
                val_smape = np.mean([2 * (np.abs(p - val_y[key][:, :, -1]) / (np.abs(p + 2) + np.abs(val_y[key][:, :, -1] + 2))).mean() for key, p in pred_val.items()])

                val_errors.append(val_mse)
                val_smape_errors.append(val_smape)
                print('MSE Validation Error: '+str(val_mse))
                print('SMAPE Validation Error: ' + str(val_smape))

        if not self.srnn.config.use_cached_predictions:
            predictions = []
            f_cs = []

            with torch.no_grad():
                print(x.shape[0] // batch_size + 1)
                for i in range(x.shape[0] // batch_size + 1):
                    batch_x, batch_y = x[i * batch_size:(i + 1) * batch_size], \
                                       y[i * batch_size:(i + 1) * batch_size]
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                        return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                    #prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)

                    predictions.append(prediction)
                    f_cs.append(f_c)

            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            f_cs = torch.cat(f_cs, dim=0).detach().cpu().numpy()

            import pickle
            with open('srnn_train.pkl', 'wb') as f:
                pickle.dump((predictions, f_cs), f)

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 48, 'figure.figsize': (60, 40)})
        index = list(range(len(westimator_losses)))
        plt.ylabel('LL')
        plt.plot(index, westimator_losses, label='WCSPN-Loss (Negative LL)', color='blue')
        plt.plot(index, srnn_losses_ll, label='SRNN-Loss (Negative LL)', color='green')
        plt.legend(loc='upper right')

        ax2 = plt.twinx()
        ax2.set_ylabel('MSE', color='red')
        ax2.plot(index, srnn_losses, label='SRNN-Loss Total', color='magenta')
        ax2.plot(index, srnn_losses_p, label='SRNN-Loss Prediction', color='red')
        ax2.legend(loc='upper left')

        plt.savefig('res/plots/0_PWN_Training_losses')

        plt.clf()
        plt.plot(val_errors)
        plt.savefig('res/plots/0_PWN_Val_MSE')

        plt.clf()
        plt.plot(val_smape_errors)
        plt.savefig('res/plots/0_PWN_Val_SMAPE')
        np.savetxt('res/plots/0_PWN_Val_MSE.txt',val_errors)
        np.savetxt('res/plots/0_PWN_Val_SMAPE.txt', val_smape_errors)
        print(val_errors)
        print(val_smape_errors)

        if self.train_rnn_w_ll:
            plt.clf()
            plt.plot(ll_weight_history)
            plt.ylabel('SRNN LL-Loss Weight (percentage of total loss)')
            plt.title('LL Weight Warmup')
            plt.savefig('res/plots/0_PWN_LLWeightWarmup')

    @torch.no_grad()
    def predict(self, x, batch_size=1024, pred_label='', mpe=False):

        predictions, f_c = self.srnn.predict(x, batch_size, pred_label=pred_label, return_coefficients=True)

        x_ = {key: x_[:, :, -1] for key, x_ in x.items()}

        f_c_ = {key: f.reshape((f.shape[0], -1)) for key, f in f_c.items()}

        if self.westimator.use_stft:
            ll = self.westimator.predict(x_, f_c_, stft_y=False, batch_size=batch_size)
        else:
            ll = self.westimator.predict(x_, predictions, batch_size=batch_size)

        if mpe:
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.net.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.westimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.westimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                              stft_y=False, batch_size=batch_size)

            # predictions, likelihoods, likelihoods_mpe, predictions_mpe
            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    def save(self, filepath):
        self.srnn.save(filepath)
        self.westimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.westimator.load(filepath)
        self.westimator.stft_module = self.srnn.net.stft

    def call_westimator(self, x, y):

        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        if hasattr(self.westimator, 'spn'):
            sum_params, leaf_params = self.westimator.weight_nn(x.reshape((x.shape[0], x.shape[1] *
                                                                           (2 if self.westimator.use_stft else 1),)))
            self.westimator.args.param_provider.sum_params = sum_params
            self.westimator.args.param_provider.leaf_params = leaf_params

            return self.westimator.spn(y_), y_
        else:
            val_in = torch.cat([x, y_], dim=1).reshape((x.shape[0], -1))
            return self.westimator.model(val_in), val_in
