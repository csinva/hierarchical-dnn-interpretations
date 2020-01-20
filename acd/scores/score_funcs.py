import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sys
from ..util.conv2dnp import conv2dnp
import copy
from .cd import cd, cd_text
from tqdm import tqdm


def gradient_times_input_scores(im: np.ndarray, ind: int, model, device='cuda'):
    '''
    Params
    ------
    im: np.ndarray
        Image to get scores with respect to
    ind: int
        Which class to take gradient with respect to
    '''
    ind = torch.LongTensor([np.int(ind)]).to(device)
    if im.grad is not None:
        im.grad.data.zero_()
    pred = model(im)
    crit = nn.NLLLoss()
    loss = crit(pred, ind)
    loss.backward()
    res = im.grad * im
    return res.data.cpu().numpy()[0, 0]




def ig_scores_2d(model, im_torch, num_classes=10, im_size=28, sweep_dim=1, ind=None, device='cuda'):
    '''Compute integrated gradients scores (2D input)
    '''
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
            
    # What class to produce explanations for
    output = np.zeros((im_size * im_size // (sweep_dim * sweep_dim), num_classes))
    
    if ind is None:
        ind = range(num_classes)
    for class_to_explain in ind:
        #         _, class_to_explain = model(im_torch).max(1); class_to_explain = class_to_explain.data[0]

        M = 100
        criterion = torch.nn.L1Loss(size_average=False)
        mult_grid = np.array(range(M)) / (M - 1)

        baseline = torch.zeros(im_torch.shape).to(device)

        input_vecs = torch.empty((M, baseline.shape[1],  baseline.shape[2], baseline.shape[3]), 
                                  dtype=torch.float32,
                                  device=device, requires_grad=False)
        '''
        input_vecs = torch.Tensor(M, baseline.size(1), 
                                  baseline.size(2), baseline.size(3)).to(device)
        input_vecs.requires_grad = True
        '''
        for i, prop in enumerate(mult_grid):
            input_vecs[i].data = baseline + (prop * (im_torch.to(device) - baseline))
        input_vecs.requires_grad=True

#         input_vecs = input_vecs

        out = F.softmax(model(input_vecs))[:, class_to_explain]
        loss = criterion(out, torch.zeros(M).to(device))
        loss.backward()

        imps = input_vecs.grad.mean(0).data.cpu() * (im_torch.data.cpu() - baseline.cpu())
        ig_scores = imps.sum(1)

        # Sanity check: this should be small-ish
        #         print((out[-1] - out[0]).data[0] - ig_scores.sum())
        scores = ig_scores.cpu().numpy().reshape((1, im_size, im_size, 1))
        kernel = np.ones(shape=(sweep_dim, sweep_dim, 1, 1))
        scores_convd = conv2dnp(scores, kernel, stride=(sweep_dim, sweep_dim))
        output[:, class_to_explain] = scores_convd.flatten()
    return output


def ig_scores_1d(batch, model, inputs, device='cuda'):
    '''Compute integrated gradients scores (1D input)
    '''
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    M = 1000
    criterion = torch.nn.L1Loss(size_average=False)
    mult_grid = np.array(range(M)) / (M - 1)
    word_vecs = model.embed(batch.text).data
    baseline_text = copy.deepcopy(batch.text)
    baseline_text.data[:, :] = inputs.vocab.stoi['.']
    baseline = model.embed(baseline_text).data
    input_vecs = torch.Tensor(baseline.size(0), M, baseline.size(2)).to(device)
    for i, prop in enumerate(mult_grid):
        input_vecs[:, i, :] = baseline + (prop * (word_vecs - baseline)).to(device)

    input_vecs = input_vecs

    hidden = (torch.zeros(1, M, model.hidden_dim).to(device),
              torch.zeros(1, M, model.hidden_dim).to(device))
    lstm_out, hidden = model.lstm(input_vecs, hidden)
    logits = F.softmax(model.hidden_to_label(lstm_out[-1]))[:, 0]
    loss = criterion(logits, torch.zeros(M).to(device))
    loss.backward()
    imps = input_vecs.grad.mean(1).data * (word_vecs[:, 0] - baseline[:, 0])
    zero_pred = logits[0]
    scores = imps.sum(1)
    #     for i in range(sent_len):
    #         print(ig_scores[i], text_orig[i])
    # Sanity check: this should be small-ish
    #     print((logits[-1] - zero_pred) - ig_scores.sum())
    return scores.cpu().numpy()


def get_scores_1d(batch, model, method, label, only_one, score_orig, text_orig, subtract=False, device='cuda'):
    '''Return attribution scores for 1D input
    Params
    ------
    method: str
        What type of method to use for attribution (e.g. cd, occlusion)
        
    Returns
    -------
    scores: np.ndarray
        Higher scores are more important
    '''
    # calculate scores
    if method == 'cd':
        if only_one:
            num_words = batch.text.data.cpu().numpy().shape[0]
            scores = np.expand_dims(cd_text(batch, model, start=0, stop=num_words), axis=0)
        else:
            starts, stops = tiles_to_cd(batch)
            batch.text.data = torch.LongTensor(text_orig).to(device)
            scores = np.array([cd_text(batch, model, start=starts[i], stop=stops[i])
                               for i in range(len(starts))])
    else:
        scores = model(batch).data.cpu().numpy()
        if method == 'occlusion' and not only_one:
            scores = score_orig - scores

    # get score for other class
    if subtract:
        return scores[:, label] - scores[:, int(1 - label)]
    else:
        return scores[:, label]

def get_scores_2d(model, method, ims, im_torch=None, pred_ims=None, model_type=None, device='cuda'):
    '''Return attribution scores for 2D input
    Params
    ------
    method: str
        What type of method to use for attribution (e.g. cd, occlusion)
    ims: np.ndarray (1 x C x H x W )
        Tiles to pass as masks to cd
        
    Returns
    -------
    scores: np.ndarray
        Higher scores are more important
    '''
    scores = []
    if method == 'cd':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(cd(im_torch, model, np.expand_dims(ims[i], 0), model_type, 
                             device=device)[0].data.cpu().numpy())
        scores = np.squeeze(np.array(scores))
    elif method == 'build_up':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(pred_ims(model, ims[i])[0])
        scores = np.squeeze(np.array(scores))
    elif method == 'occlusion':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(pred_ims(model, ims[i])[0])
        scores = -1 * np.squeeze(np.array(scores))
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    return scores



def tiles_to_cd(batch):
    '''Converts build up tiles into indices for cd
    Cd requires batch of [start, stop) with unigrams working
    build up tiles are of the form [0, 0, 12, 35, 0, 0]
    return a list of starts and indices
    '''
    starts, stops = [], []
    tiles = batch.text.data.cpu().numpy()
    L = tiles.shape[0]
    for c in range(tiles.shape[1]):
        text = tiles[:, c]
        start = 0
        stop = L - 1
        while text[start] == 0:
            start += 1
        while text[stop] == 0:
            stop -= 1
        starts.append(start)
        stops.append(stop)
    return starts, stops
