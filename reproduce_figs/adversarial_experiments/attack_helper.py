import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import sys
import random, pickle
from os.path import join as oj
from tqdm import tqdm
sys.path.insert(1, oj(sys.path[0], 'mnist'))
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy
cs_div = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
sns.set_style(style='white')
sns.set_palette("husl")

import torch
import foolbox
import mnist.mnist as dset
from mnist.mnist import pred_ims
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from conv2dnp import conv2dnp
import visualize as viz
import tiling
import agglomerate
from cd import cd
import scores

# load the model
model = dset.Net().cuda()
model.load_state_dict(torch.load('mnist/mnist.model'))
model.eval()

def attack_im_num(im_num, attack_type):
    # seed
    np.random.seed(13)
    random.seed(13)
    torch.manual_seed(13)
    
    
    im_torch, im_orig, label = dset.get_im_and_label(im_num)
    pix_min = np.min(im_torch.data.cpu().numpy())
    pix_max = np.max(im_torch.data.cpu().numpy())
    fmodel = foolbox.models.PyTorchModel(model, bounds=(pix_min-1, pix_max+1), 
                                         num_classes=10, channel_axis=1)
    if attack_type == 'saliency':
        attack = foolbox.attacks.SaliencyMapAttack(fmodel) # saliency map attack has some good stuff
    elif attack_type == 'fgsm':
        attack = foolbox.attacks.FGSM(fmodel) # saliency map attack has some good stuff
    elif attack_type == 'gradientattack':
        attack = foolbox.attacks.GradientAttack(fmodel) # saliency map attack has some good stuff
    elif attack_type == 'deepfoolattack':
        attack = foolbox.attacks.DeepFoolAttack(fmodel) # saliency map attack has some good stuff
    elif attack_type == 'boundaryattack':
        attack = foolbox.attacks.BoundaryAttack(fmodel) # saliency map attack has some good stuff

    if attack_type == 'boundaryattack':
        im_orig_adv = attack(im_torch.data.cpu().numpy()[0], label, log_every_n_steps=10000)
    else:
        im_orig_adv = attack(im_torch.data.cpu().numpy()[0], label)

    # # set up vars
    im_torch_adv = Variable(torch.from_numpy(np.expand_dims(im_orig_adv, 0)).cuda(), requires_grad=True)
    im_orig_adv = im_orig_adv[0] # make this 28 x 28
    
    # see preds
    pred_orig = model(im_torch).exp().data.cpu().numpy().flatten()
    pred_adv = model(im_torch_adv).exp().data.cpu().numpy().flatten()
    targets = np.argsort(pred_adv)
    target = targets[-1]
    if target == label:
        target = targets[-2]
    
    return im_orig, im_torch, im_orig_adv, im_torch_adv, label, target, pred_orig, pred_adv

def plot_attack(im_orig, im_orig_adv, label, target, pred_orig, pred_adv):
    plt.figure(figsize=(6, 2))
    plt.subplot(131)
    plt.imshow(im_orig, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(im_orig_adv - im_orig, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(im_orig_adv, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.show()

    print('\tlabel:', label, pred_orig[label], '->', pred_adv[label])
    print('\ttarget:', target, pred_orig[target], '->', pred_adv[target])
    
def plot_lists(lists, method='cd'):
    num_iters = len(lists['scores_list'])
    
     # visualize
    plt.figure(figsize=(num_iters, 3), facecolor='white', dpi=150)
    rows = 4
    viz.visualize_ims_list(lists['scores_list'], 
                           cmap_new='redwhiteblue',
                           title='Refined importance scores',
                           subplot_row=0, subplot_rows=rows, colorbar=False)
    viz.visualize_ims_list(lists['comps_list'],
                          title='Different components of chosen pixels',
                          subplot_row=1, subplot_rows=rows, colorbar=False)
    viz.visualize_dict_list(lists['comp_scores_raw_list'], method,
                           subplot_row=2, subplot_rows=rows)
    viz.visualize_arr_list(lists['comp_scores_raw_combined_list'], method, 
                       subplot_row=3, subplot_rows=rows)

def agg_and_plot(im_torch, lab_num, im_orig, model, pred_ims, percentile_include, 
             method, sweep_dim, layer, use_abs, num_iters, plot=True):
    lists = agglomerate.agglomerate(model, pred_ims, percentile_include, method, sweep_dim, layer, im_orig, 
                                    lab_num, use_abs, num_iters=num_iters, im_torch=im_torch, model_type='mnist')    
    if plot:
        plot_lists(lists)
        
    return lists

def agglomerate_lists(im_orig, im_torch, im_orig_adv, im_torch_adv, label, target, method='cd'):
    # agg params
    percentile_include = 98
    sweep_dim = 1
    layer = 'softmax'
    use_abs = False
    num_iters = 35

    # agglomerate lists
    lists_orig_lab = agg_and_plot(im_torch, label, im_orig, model, dset.pred_ims, percentile_include, 
             method, sweep_dim, layer, use_abs, num_iters=num_iters, plot=False)
    lists_orig_targ = agg_and_plot(im_torch, target, im_orig, model, dset.pred_ims, percentile_include, 
             method, sweep_dim, layer, use_abs, num_iters=num_iters, plot=False)

    lists_adv_lab = agg_and_plot(im_torch_adv, label, im_orig_adv, model, dset.pred_ims, percentile_include, 
             method, sweep_dim, layer, use_abs, num_iters=num_iters, plot=False)
    lists_adv_targ = agg_and_plot(im_torch_adv, target, im_orig_adv, model, dset.pred_ims, percentile_include, 
             method, sweep_dim, layer, use_abs, num_iters=num_iters, plot=False)
    return [lists_orig_lab, lists_orig_targ, lists_adv_lab, lists_adv_targ]