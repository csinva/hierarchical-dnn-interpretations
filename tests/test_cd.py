import numpy as np
import torch
import sys
sys.path.append('..')
sys.path.append('../acd/util')
sys.path.append('../acd/scores')
import cd
import score_funcs
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


def test_sst(device='cpu'):
    
    # load the model and data
    sys.path.append('../dsets/sst')
    from dsets.sst.model import LSTMSentiment
    sst_pkl = pkl.load(open('../dsets/sst/sst_vocab.pkl', 'rb'))
    model = torch.load('../dsets/sst/sst.model', map_location=device)
    model.device = device
    
    # text and label
    sentence = ['a', 'great', 'ensemble', 'cast', 'ca', 'n\'t', 'lift', 'this', 'heartfelt', 'enterprise', 'out', 'of', 'the', 'familiar', '.'] # note this is a real example from the dataset

    def batch_from_str_list(s):
        # form class to hold data
        class B:
            text = torch.zeros(1).to(device)

        batch = B()
        nums = np.expand_dims(np.array([sst_pkl['stoi'][x] for x in s]).transpose(), axis=1)
        batch.text = torch.LongTensor(nums).to(device) #cuda()
        return batch

    # prepare inputs
    batch = batch_from_str_list(sentence)
    preds = model(batch).data.cpu().numpy()[0] # predict

    # check that full sentence = prediction
    preds = preds - model.hidden_to_label.bias.detach().numpy()
    cd_score, irrel_scores = cd.cd_text(batch, model, start=0, stop=len(sentence), return_irrel_scores=True)
    assert(np.allclose(cd_score, preds, atol=1e-2))
    assert(np.allclose(irrel_scores, irrel_scores * 0, atol=1e-2))

    # check that rel + irrel = prediction for another subset
    cd_score, irrel_scores = cd.cd_text(batch, model, start=3, stop=len(sentence), return_irrel_scores=True)
    assert(np.allclose(cd_score + irrel_scores, preds, atol=1e-2))
    
def test_mnist(device='cuda'):
    # load the dataset
    sys.path.append('../dsets/mnist')
    import dsets.mnist.model
    device = 'cuda'
    im_torch = torch.randn(1, 1, 28, 28).to(device)

    # load the model
    model = dsets.mnist.model.Net().to(device)
    model.load_state_dict(torch.load('../dsets/mnist/mnist.model', map_location=device))
    model = model.eval()
    
    # check that full image mask = prediction
    preds = model.logits(im_torch).cpu().detach().numpy()
    cd_score, irrel_scores = cd.cd(np.ones((1, 1, 28, 28)), im_torch, model, model_type='mnist', device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score, preds, atol=1e-2))
    assert(np.allclose(irrel_scores, irrel_scores * 0, atol=1e-2))

    # check that rel + irrel = prediction for another subset
    # preds = preds - model.hidden_to_label.bias.detach().numpy()
    mask = np.zeros((28, 28))
    mask[:14] = 1
    cd_score, irrel_scores = cd.cd(mask, im_torch, model, model_type='mnist', device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score + irrel_scores, preds, atol=1e-2))
    
def test_imagenet_vgg(device='cuda', arch='vgg'):
    # get dataset
    from torchvision import models
    imnet_dict = pkl.load(open('../dsets/imagenet/imnet_dict.pkl', 'rb')) # contains 6 images (keys: 9, 10, 34, 20, 36, 32)

    # get model and image
    if arch == 'vgg':
        model = models.vgg16(pretrained=True).to(device).eval()
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True).to(device).eval()
    im_torch = torch.randn(1, 3, 224, 224).to(device)
    
    # check that full image mask = prediction
    preds = model(im_torch).cpu().detach().numpy()
    cd_score, irrel_scores = cd.cd(np.ones((1, 3, 224, 224)), im_torch, model, device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score, preds, atol=1e-2))
    assert(np.allclose(irrel_scores, irrel_scores * 0, atol=1e-2))

    # check that rel + irrel = prediction for another subset
    # preds = preds - model.hidden_to_label.bias.detach().numpy()
    mask = np.ones((1, 3, 224, 224))
    mask[:, :, :14] = 1
    cd_score, irrel_scores = cd.cd(mask, im_torch, model, device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score + irrel_scores, preds, atol=1e-2))
    
if __name__ == '__main__':
    print('testing sst...')
    test_sst()
    print('testing mnist...')
    test_mnist()
    print('testing imagenet vgg...')
    test_imagenet_vgg(arch='vgg')
    print('testing imagenet alexnet...')
    test_imagenet_vgg(arch='alexnet')
    print('all tests passed!')
    
    # loop over device types?