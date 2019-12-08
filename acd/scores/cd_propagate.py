import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid


def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    '''Propagate convolutional or linear layer
    Apply linear part to both pieces
    Split bias based on the ratio of the absolute sums
    '''
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


def propagate_batchnorm2d(relevant, irrelevant, module, device='cuda'):
    '''Propagate batchnorm2d operation
    '''
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
    
    # pooling function
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


def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    '''propagate ReLu nonlinearity
    '''
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    zeros = torch.zeros(relevant.size()).to(device)
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


def propagate_dropout(relevant, irrelevant, dropout):
    '''propagate dropout operation
    just applies dropout to each piece separately
    '''
    return dropout(relevant), dropout(irrelevant)


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