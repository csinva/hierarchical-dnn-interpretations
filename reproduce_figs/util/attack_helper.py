import numpy as np
import matplotlib.pyplot as plt
import torch
import foolbox
from torch.autograd import Variable
import random

def attack_im_num(dset, model, im_num, attack_type):
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