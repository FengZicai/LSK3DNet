from __future__ import print_function
import torch
import math
import copy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils.funcs import redistribution_funcs, growth_funcs, prune_funcs


class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1, init_step=0):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)
        if init_step!=0:
            for i in range(init_step):
                self.cosine_stepper.step()
    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']


class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)
    """
    def __init__(self, optimizer, scaler, spatial_partition, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', 
                growth_mode='random',  redistribution_mode='momentum', fp16=False, update_frequency=None, z_spatial_partition=None,
                sparsity=None, sparse_init=None, device=None, distributed=False, stop_iter = 60000):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))
        self.device = torch.device(device)
        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer
        self.scaler = scaler
        self.baseline_nonzero = None

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}
        self.prune_rate = prune_rate
        self.steps = 0
        self.half = fp16
        self.name_to_32bit = {}

        self.update_frequency = update_frequency
        self.stop_iter = stop_iter
        self.sparsity = sparsity
        self.sparse_init = sparse_init

        self.distributed = distributed

        # self.kernel_size = 9 # [0,3,0,3,0,3] k1 k2 k3
        self.group_num = len(spatial_partition) - 1
        self.start_end_map = {}
        for i in range(self.group_num):
            self.start_end_map.update({i:[spatial_partition[i],spatial_partition[i+1]]}) 
        if z_spatial_partition != None:
            self.z_start_end_map = {}
            for i in range(self.group_num):
                self.z_start_end_map.update({i:[z_spatial_partition[i],z_spatial_partition[i+1]]}) 
        else:
            self.z_start_end_map = self.start_end_map

        self.group_map = torch.zeros((int(self.group_num)**3), 6) - 1
        for ik,iv in self.z_start_end_map.items():
            for jk,jv in self.start_end_map.items():
                for kk,kv in self.start_end_map.items():
                    index = ik * self.group_num * self.group_num + jk * self.group_num + kk
                    self.group_map[index] = torch.tensor([iv[0],iv[1],jv[0],jv[1],kv[0],kv[1]])

        self.group_map = self.group_map.int().cuda()


    def add_module(self, module, pre_masks=None):
        self.modules.append(module)
        self.module = module
        if pre_masks == None:
            for name, tensor in module.named_parameters():
                if len(tensor.size()) == 5:
                    if '.layers.' in name:
                        # print(tensor.shape)
                        self.names.append(name)
                        one_kernel_mask_map_dict = {}

                        for ind in range(len(self.group_map)):
                            index_ten = self.group_map[ind]
                            one_kernel_mask_map_dict[ind] = torch.zeros_like(tensor[index_ten[0]:index_ten[1], 
                                                                    index_ten[2]:index_ten[3],
                                                                    index_ten[4]:index_ten[5]], 
                                                            dtype=torch.float32, 
                                                            requires_grad=False).to(self.device)

                        self.masks[name] = one_kernel_mask_map_dict
        else:
            # try:
            self.masks = pre_masks
            # except:
                # print('mask error')

        self.init(mode=self.sparse_init, density=1-self.sparsity, pre_masks=pre_masks)


    # def init_optimizer(self):
    #     import pdb
    #     pdb.set_trace()
    #     if 'fp32_from_fp16' in self.optimizer.state_dict():
    #         print('fp32_from_fp16')
    #         for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
    #             self.name_to_32bit[name] = tensor2
    #         self.half = True

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0, pre_masks=None):
        self.init_growth_prune_and_redist()
        # self.init_optimizer()
        self.density = density

        if pre_masks==None:
            print('initialize by fixed_ERK')
            total_params = 0
            self.baseline_nonzero = 0
            for name, spatial_group in self.masks.items():
                for weight in spatial_group.values():
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel() * density
            is_epsilon_valid = False

            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name_1, spatial_mask in self.masks.items():
                    for name_2, mask in spatial_mask.items():
                        name = name_1 + '/' + str(name_2)
                        n_param = np.prod(mask.shape)
                        n_zeros = n_param * (1 - density)
                        n_ones = n_param * density

                        if name in dense_layers:
                            rhs -= n_zeros

                        else:
                            rhs += n_ones
                            raw_probabilities[name] = (
                                                                np.sum(mask.shape) / np.prod(mask.shape)
                                                        ) ** erk_power_scale

                            divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            # for name, mask in self.masks.items():
            for name_1, spatial_mask in self.masks.items():
                for name_2, mask in spatial_mask.items():
                    name = name_1 + '/' + str(name_2)
                    n_param = np.prod(mask.shape)
                    if name in dense_layers:
                        density_dict[name] = 1.0
                    else:
                        probability_one = epsilon * raw_probabilities[name]
                        density_dict[name] = probability_one
                    print(
                        f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                    )
                    self.masks[name_1][name_2][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

                    total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

            total_size = 0
            sparse_size = 0
            dense_layers = []
            # for name, weight in self.masks.items():
            for name_1, spatial_mask in self.masks.items():
                for name_2, weight in spatial_mask.items():
                    name = name_1 + '/' + str(name_2)
                    dense_weight_num = weight.numel()
                    sparse_weight_num = (weight != 0).sum().int().item()
                    total_size += dense_weight_num
                    sparse_size += sparse_weight_num
                    layer_density = sparse_weight_num / dense_weight_num
                    if layer_density >= 0.99: dense_layers.append(name)
                    print(f'Density of layer {name} with tensor {weight.size()} is {layer_density}')
            print('Final sparsity level of {0}: {1}'.format(1-self.density, 1 - sparse_size / total_size))

            # masks of layers with density=1 are removed
            for name in dense_layers:
                name_1, name_2 = name.split('/')
                name_2 = int(name_2)
                # import pdb
                # pdb.set_trace()
                self.masks[name_1].pop(name_2)
                print(f"pop out layer {name}")
        else:
            self.masks = pre_masks

        self.apply_mask()

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')


    def step(self):
        if self.half:
            self.apply_mask()
        else:
            self.optimizer.step()
            self.apply_mask()

        # decay the adaptation rate for better results
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        self.steps += 1
        if self.steps < self.stop_iter:
            if self.steps % self.update_frequency == 0:
                print('*********************************Dynamic Sparsity********************************')
                self.truncate_weights()
                self.print_nonzero_counts()


    def apply_mask(self):

        # synchronism masks
        if self.distributed:
            self.synchronism_masks()

        for module in self.modules:
            for name_1, tensor in module.named_parameters():
                if name_1 in self.masks:
                    for name_2 in self.masks[name_1].keys():
                        index_ten = self.group_map[name_2]
                        # if not self.half:
                        tensor.data[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]] = \
                            tensor.data[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]]*self.masks[name_1][name_2]
                        if 'momentum_buffer' in self.optimizer.state[tensor]:
                            self.optimizer.state[tensor]['momentum_buffer'][index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]] = \
                                self.optimizer.state[tensor]['momentum_buffer'][index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]]*self.masks[name_1][name_2]
                        # else:
                        #     import pdb
                        #     pdb.set_trace()
                        #     # tensor.data = tensor.data*self.masks[name].half()
                        #     tensor.data[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]] = \
                        #         tensor.data[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]]*self.masks[name_1][name_2].half()
                            # import pdb
                            # pdb.set_trace()
                            # if name_1 in self.name_to_32bit:
                            #     tensor2 = self.name_to_32bit[name_1]
                            #     tensor2.data = tensor2.data*self.masks[name_1]

    def truncate_weights(self):

        for module in self.modules:
            for name_1, weight in module.named_parameters():
                if name_1 not in self.masks: continue
                for name_2 in self.masks[name_1].keys():
                    name = name_1 + '/' + str(name_2)
                    index_ten = self.group_map[name_2]

                    mask = self.masks[name_1][name_2]
                    self.name2nonzeros[name] = mask.sum().item()
                    self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                    # prune
                    new_mask = self.prune_func(self, mask, weight[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]].contiguous(), name)
                    removed = self.name2nonzeros[name] - new_mask.sum().item()
                    self.name2removed[name] = removed
                    self.masks[name_1][name_2][:] = new_mask

        for module in self.modules:
            for name_1, weight in module.named_parameters():
                if name_1 not in self.masks: continue
                for name_2 in self.masks[name_1].keys():
                    name = name_1 + '/' + str(name_2)
                    index_ten = self.group_map[name_2]

                    new_mask = self.masks[name_1][name_2].data.byte()
                    # growth
                    new_mask = self.growth_func(self, name, new_mask, math.floor(self.name2removed[name]), weight[index_ten[0]:index_ten[1], index_ten[2]:index_ten[3], index_ten[4]:index_ten[5]].contiguous())
                    self.masks[name_1][name_2][:] = new_mask.float()

        self.apply_mask()

    '''
                UTILITY
    '''
    # def get_momentum_for_weight(self, weight):
    #     if 'exp_avg' in self.optimizer.state[weight]:
    #         adam_m1 = self.optimizer.state[weight]['exp_avg']
    #         adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
    #         grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
    #     elif 'momentum_buffer' in self.optimizer.state[weight]:
    #         grad = self.optimizer.state[weight]['momentum_buffer']

    #     return grad

    # def get_gradient_for_weights(self, weight):
    #     grad = weight.grad.clone()
    #     return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name_1, tensor in module.named_parameters():
                if name_1 not in self.masks: continue
                for name_2 in self.masks[name_1].keys():
                    name = name_1 + '/' + str(name_2)
                    mask = self.masks[name_1][name_2]
                    num_nonzeros = (mask != 0).sum().item()
                    val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                                num_nonzeros / float(mask.numel()))
                    print(val)

        print('Prune rate: {0}\n'.format(self.prune_rate))

    # def fired_masks_update(self):
    #     ntotal_fired_weights = 0.0
    #     ntotal_weights = 0.0
    #     layer_fired_weights = {}
    #     for module in self.modules:
    #         for name, weight in module.named_parameters():
    #             if name not in self.masks: continue
    #             self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
    #             ntotal_fired_weights += float(self.fired_masks[name].sum().item())
    #             ntotal_weights += float(self.fired_masks[name].numel())
    #             layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
    #             # print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
    #     total_fired_weights = ntotal_fired_weights/ntotal_weights
    #     print('The percentage of the total fired weights is:', total_fired_weights)
    #     return layer_fired_weights, total_fired_weights

    def synchronism_masks(self):
        for name_1, spatial_mask in self.masks.items():
            for name_2 in spatial_mask.keys():
                torch.distributed.broadcast(self.masks[name_1][name_2], src=0, async_op=False)
