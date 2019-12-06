import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid
from .cd_propagate import *
from .cd_architecture_specific import *

def cd(im_torch: torch.Tensor, model, mask=None, model_type=None, device='cuda', transform=None):
    '''Get contextual decomposition scores for blob
    
    Params
    ------
    mask: array_like (values in {0, 1})
        required unless transform is supplied
        array with 1s marking the locations of relevant pixels, 0s marking the background
        shape should match the shape of im_torch or just H x W
    im_torch: torch.Tensor
        example to interpret - usually has shape (batch_size, num_channels, height, width)
    model_type: str, optional
        usually should just leave this blank
        if this is == 'mnist', uses CD for a specific mnist model
        if this is == 'resnet18', uses resnet18 model
    device: str, optional
    transform: function
        transform should be a function which transforms the original image
        only used if mask is not passed
        
    Returns
    -------
    relevant: torch.Tensor
        class-wise scores for relevant mask
    irrelevant: torch.Tensor
        class-wise scores for everything but the relevant mask 
    '''
    
    # set up model
    model.eval()
    im_torch = im_torch.to(device)
    
    # set up masks
    if not mask is None:
        mask = torch.FloatTensor(mask).to(device)
        relevant = mask * im_torch
        irrelevant = (1 - mask) * im_torch
    elif not transform is None:
        relevant = transform(im_torch).to(device)
        if len(relevant.shape) < 4:
            relevant = relevant.reshape(1, 1, relevant.shape[0], relevant.shape[1])
        irrelevant = im_torch - relevant
    else:
        print('invalid arguments')
    relevant = relevant.to(device)
    irrelevant = irrelevant.to(device)

    if model_type == 'mnist':
        return cd_propagate_mnist(relevant, irrelevant, model)
    elif model_type == 'resnet18':
        return cd_propagate_resnet(relevant, irrelevant, model)
    
    mods = list(model.modules())
    relevant, irrelevant = cd_generic(mods, relevant, irrelevant)
    return relevant, irrelevant

def cd_generic(mods, relevant, irrelevant):
    for i, mod in enumerate(mods):
        t = str(type(mod))
        if 'Conv2d' in t:
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'Linear' in t:
            relevant = relevant.reshape(relevant.shape[0], -1)
            irrelevant = irrelevant.reshape(irrelevant.shape[0], -1)
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'ReLU' in t:
            relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
        elif 'Pool' in t:
            relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod)
        elif 'Dropout' in t:
            relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)
        elif 'BatchNorm2d' in t:
            relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mod)
    return relevant, irrelevant


def cd_text(batch, model, start, stop, return_irrel_scores=False):
    '''Get contextual decomposition scores for substring of a text sequence
    
    Params
    ------
        batch: torchtext batch
            really only requires that batch.text is the string input to be interpreted
        start: int
            beginning index of substring to be interpreted (inclusive)
        stop: int
            ending index of substring to be interpreted (inclusive)

    Returns
    -------
        scores: torch.Tensor
            class-wise scores for relevant substring
    '''
    weights = model.lstm.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'], 4, 0)
    b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
    word_vecs = model.embed(batch.text)[:, 0].data
    T = word_vecs.size(0)
    relevant = np.zeros((T, model.hidden_dim))
    irrelevant = np.zeros((T, model.hidden_dim))
    relevant_h = np.zeros((T, model.hidden_dim))
    irrelevant_h = np.zeros((T, model.hidden_dim))
    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]
        else:
            prev_rel_h = np.zeros(model.hidden_dim)
            prev_irrel_h = np.zeros(model.hidden_dim)

        rel_i = np.dot(W_hi, prev_rel_h)
        rel_g = np.dot(W_hg, prev_rel_h)
        rel_f = np.dot(W_hf, prev_rel_h)
        rel_o = np.dot(W_ho, prev_rel_h)
        irrel_i = np.dot(W_hi, prev_irrel_h)
        irrel_g = np.dot(W_hg, prev_irrel_h)
        irrel_f = np.dot(W_hf, prev_irrel_h)
        irrel_o = np.dot(W_ho, prev_irrel_h)

        if i >= start and i <= stop:
            rel_i = rel_i + np.dot(W_ii, word_vecs[i])
            rel_g = rel_g + np.dot(W_ig, word_vecs[i])
            rel_f = rel_f + np.dot(W_if, word_vecs[i])
            rel_o = rel_o + np.dot(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
            irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
            irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
            irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g, np.tanh)

        relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if i >= start and i <= stop:
            relevant[i] += bias_contrib_i * bias_contrib_g
        else:
            irrelevant[i] += bias_contrib_i * bias_contrib_g

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f, sigmoid)
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * \
                                                                                                      relevant[i - 1]

        o = sigmoid(np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o)
        rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])
        # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
        # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

    W_out = model.hidden_to_label.weight.data

    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    scores = np.dot(W_out, relevant_h[T - 1])
    irrel_scores = np.dot(W_out, irrelevant_h[T - 1])

    if return_irrel_scores:
        return scores, irrel_scores
    
    return scores