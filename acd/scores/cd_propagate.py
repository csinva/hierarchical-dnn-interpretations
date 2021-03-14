from copy import deepcopy

import numpy as np
import torch
from scipy.special import expit as sigmoid
from torch import tanh


def propagate_conv_linear(relevant, irrelevant, module):
    '''Propagate convolutional or linear layer
    Apply linear part to both pieces
    Split bias based on the ratio of the absolute sums
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel) + 1e-20  # add a small constant so we don't divide by 0
    prop_irrel = torch.abs(irrel) + 1e-20  # add a small constant so we don't divide by 0
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


def propagate_batchnorm2d(relevant, irrelevant, module):
    '''Propagate batchnorm2d operation
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_rel[torch.isnan(prop_rel)] = 0
    rel = rel + torch.mul(prop_rel, bias)
    irrel = module(relevant + irrelevant) - rel
    return rel, irrel


def propagate_pooling(relevant, irrelevant, pooler):
    '''propagate pooling operation
    '''
    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)

    # unpooling function
    def unpool(tensor, indices):
        '''Unpool tensor given indices for pooling
        '''
        batch_size, in_channels, H, W = indices.shape
        output = torch.ones_like(indices, dtype=torch.float)
        for i in range(batch_size):
            for j in range(in_channels):
                output[i, j] = tensor[i, j].flatten()[indices[i, j].flatten()].reshape(H, W)
        return output

    rel, irrel = unpool(relevant, both_ind), unpool(irrelevant, both_ind)
    return rel, irrel


def propagate_independent(relevant, irrelevant, module):
    '''use for things which operate independently
    ex. avgpool, layer_norm, dropout
    '''
    return module(relevant), module(irrelevant)


def propagate_relu(relevant, irrelevant, activation):
    '''propagate ReLu nonlinearity
    '''
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


def propagate_three(a, b, c, activation):
    '''Propagate a three-part nonlinearity
    '''
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


def propagate_tanh_two(a, b):
    '''propagate tanh nonlinearity
    '''
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))


def propagate_basic_block(rel, irrel, module):
    '''Propagate a BasicBlock (used in the ResNet architectures)
    This is what the forward pass of the basic block looks like
    identity = x

    out = self.conv1(x) # 1
    out = self.bn1(out) # 2
    out = self.relu(out) # 3
    out = self.conv2(out) # 4
    out = self.bn2(out) # 5

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    '''
    from .cd import cd_generic
    #     for mod in module.modules():
    #         print('\tm', mod)
    rel_identity, irrel_identity = deepcopy(rel), deepcopy(irrel)
    rel, irrel = cd_generic(list(module.modules())[1:6], rel, irrel)

    if module.downsample is not None:
        rel_identity, irrel_identity = cd_generic(module.downsample.modules(), rel_identity, irrel_identity)

    rel += rel_identity
    irrel += irrel_identity
    rel, irrel = propagate_relu(rel, irrel, module.relu)

    return rel, irrel


def propagate_lstm(x, module, start: int, stop: int, my_device=0):
    '''module is an lstm layer
    
    Params
    ------
    module: lstm layer
    x: torch.Tensor
        (batch_size, seq_len, num_channels)
        warning: default lstm uses shape (seq_len, batch_size, num_channels)
    start: int
        start of relevant sequence
    stop: int
        end of relevant sequence
        
    Returns
    -------
    rel, irrel: torch.Tensor
        (batch_size, num_channels, num_hidden_lstm)
    '''

    # extract out weights
    W_ii, W_if, W_ig, W_io = torch.chunk(module.weight_ih_l0, 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(module.weight_hh_l0, 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(module.bias_ih_l0 + module.bias_hh_l0, 4)

    # prepare input x
    # x_orig = deepcopy(x)
    x = x.permute(1, 2, 0)  # convert to (seq_len, num_channels, batch_size)
    seq_len = x.shape[0]
    batch_size = x.shape[2]
    output_dim = W_ho.shape[1]
    relevant_h = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    irrelevant_h = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    prev_rel = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    prev_irrel = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    for i in range(seq_len):
        prev_rel_h = relevant_h
        prev_irrel_h = irrelevant_h
        rel_i = torch.matmul(W_hi, prev_rel_h)
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)

        if i >= start and i <= stop:
            rel_i = rel_i + torch.matmul(W_ii, x[i])
            rel_g = rel_g + torch.matmul(W_ig, x[i])
            rel_f = rel_f + torch.matmul(W_if, x[i])
            # rel_o = rel_o + torch.matmul(W_io, x[i])
        else:
            irrel_i = irrel_i + torch.matmul(W_ii, x[i])
            irrel_g = irrel_g + torch.matmul(W_ig, x[i])
            irrel_f = irrel_f + torch.matmul(W_if, x[i])
            # irrel_o = irrel_o + torch.matmul(W_io, x[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i[:, None], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g[:, None], tanh)

        relevant = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (
                rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if i >= start and i < stop:
            relevant = relevant + bias_contrib_i * bias_contrib_g
        else:
            irrelevant = irrelevant + bias_contrib_i * bias_contrib_g

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f[:, None], sigmoid)
            relevant = relevant + (rel_contrib_f + bias_contrib_f) * prev_rel
            irrelevant = irrelevant + (
                    rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_irrel + irrel_contrib_f * prev_rel

        o = sigmoid(torch.matmul(W_io, x[i]) + torch.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o[:, None])
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant, irrelevant)

        relevant_h = o * new_rel_h
        irrelevant_h = o * new_irrel_h
        prev_rel = relevant
        prev_irrel = irrelevant

    #     outputs, (h1, c1) = module(x_orig)
    #     assert np.allclose((relevant_h + irrelevant_h).detach().numpy().flatten(),
    #                        h1.detach().numpy().flatten(), rtol=0.01)

    # reshape output
    rel_h = relevant_h.transpose(0, 1).unsqueeze(1)
    irrel_h = irrelevant_h.transpose(0, 1).unsqueeze(1)
    return rel_h, irrel_h

def propagate_lstm_block(x_rel, x_irrel, module, start: int, stop: int, my_device=0):
    '''module is an lstm layer. This function still experimental
    
    Params
    ------
    module: lstm layer
    x_rel: torch.Tensor
        (batch_size, seq_len, num_channels)
        warning: default lstm uses shape (seq_len, batch_size, num_channels)
    x_irrel: torch.Tensor
        (batch_size, seq_len, num_channels)
    start: int
        start of relevant sequence
    stop: int
        end of relevant sequence
    weights: torch.Tensor
        (seq_len)
        
    Returns
    -------
    rel, irrel: torch.Tensor
        (batch_size, num_channels, num_hidden_lstm)
    '''

    # ex_reltract out weights
    W_ii, W_if, W_ig, W_io = torch.chunk(module.weight_ih_l0, 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(module.weight_hh_l0, 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(module.bias_ih_l0 + module.bias_hh_l0, 4)

    # prepare input x
    # x_orig = deepcopy(x)
    x_rel = x_rel.permute(1, 2, 0)  # convert to (seq_len, num_channels, batch_size)
    x_irrel = x_irrel.permute(1, 2, 0)  # convert to (seq_len, num_channels, batch_size)
    x = x_rel + x_irrel
    # print('shapes', x_rel.shape, x_irrel.shape, x.shape)
    seq_len = x_rel.shape[0]
    batch_size = x_rel.shape[2]
    output_dim = W_ho.shape[1]
    relevant_h = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    irrelevant_h = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    prev_rel = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    prev_irrel = torch.zeros((output_dim, batch_size), device=torch.device(my_device), requires_grad=False)
    for i in range(seq_len):
        prev_rel_h = relevant_h
        prev_irrel_h = irrelevant_h
        rel_i = torch.matmul(W_hi, prev_rel_h)
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)

        # relevant parts
        rel_i = rel_i + torch.matmul(W_ii, x_rel[i])
        rel_g = rel_g + torch.matmul(W_ig, x_rel[i])
        rel_f = rel_f + torch.matmul(W_if, x_rel[i])
        # rel_o = rel_o + torch.matmul(W_io, x[i])
        
        # irrelevant parts
        irrel_i = irrel_i + torch.matmul(W_ii, x_irrel[i])
        irrel_g = irrel_g + torch.matmul(W_ig, x_irrel[i])
        irrel_f = irrel_f + torch.matmul(W_if, x_irrel[i])
        # irrel_o = irrel_o + torch.matmul(W_io, x[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i[:, None], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g[:, None], tanh)

        relevant = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + \
            bias_contrib_i * rel_contrib_g
        irrelevant = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + \
            (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        # if i >= start and i < stop:
        relevant = relevant + bias_contrib_i * bias_contrib_g
        # else:
        irrelevant = irrelevant + bias_contrib_i * bias_contrib_g

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f[:, None], sigmoid)
            relevant = relevant + (rel_contrib_f + bias_contrib_f) * prev_rel
            irrelevant = irrelevant + (
                    rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_irrel + irrel_contrib_f * prev_rel

        o = sigmoid(torch.matmul(W_io, x[i]) + torch.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o[:, None])
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant, irrelevant)

        relevant_h = o * new_rel_h
        irrelevant_h = o * new_irrel_h
        prev_rel = relevant
        prev_irrel = irrelevant

    #     outputs, (h1, c1) = module(x_orig)
    #     assert np.allclose((relevant_h + irrelevant_h).detach().numpy().flatten(),
    #                        h1.detach().numpy().flatten(), rtol=0.01)

    # reshape output
    rel_h = relevant_h.transpose(0, 1).unsqueeze(1)
    irrel_h = irrelevant_h.transpose(0, 1).unsqueeze(1)
    return rel_h, irrel_h