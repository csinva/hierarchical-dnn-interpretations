import torch
import torch.nn.functional as F

from .cd_propagate import *


def cd_propagate_resnet(rel, irrel, model):
    '''Propagate a resnet architecture
    each BasicBlock passes its input through to its output (might need to downsample)
    note: the bigger resnets use BottleNeck instead of BasicBlock
    '''
    mods = list(model.modules())
    from .cd import cd_generic
    '''
    # mods[1:5]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    # mods[5, 18, 34, 50]
    x = self.layer1(x)  
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    '''

    rel, irrel = cd_generic(mods[1:5], rel, irrel)

    lay_nums = [5, 18, 34, 50]
    for lay_num in lay_nums:
        for basic_block in mods[lay_num]:
            rel, irrel = propagate_basic_block(rel, irrel, basic_block)

    # final things after BasicBlocks
    rel, irrel = cd_generic(mods[-2:], rel, irrel)
    return rel, irrel


def cd_propagate_mnist(relevant, irrelevant, model):
    '''Propagate a specific mnist architecture
    The reason we can't automatically get this score with cd_generic is because
    the model.modules() is missing some things like self.maxpool, and self.Relu
    because the model file only defined these things in the forward method
    '''
    mods = list(model.modules())[1:]
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                             lambda x: F.max_pool2d(x, 2, return_indices=True))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                             lambda x: F.max_pool2d(x, 2, return_indices=True))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

    relevant = relevant.view(-1, 320)
    irrelevant = irrelevant.view(-1, 320)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[4])

    return relevant, irrelevant


def cd_track_vgg(blob, im_torch, model, model_type='vgg'):
    '''This implementation of cd is very long so that we can view CD at intermediate layers
    In reality, one should use the loop contained in the above cd function
    '''
    # set up model
    model.eval()

    # set up blobs
    blob = torch.cuda.FloatTensor(blob)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    mods = list(model.modules())[2:]
    scores = []
    #         (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace)
    #         (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (3): ReLU(inplace)
    #         (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[4])

    #         (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (6): ReLU(inplace)
    #         (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (8): ReLU(inplace)
    #         (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[5])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[6])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[7])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[8])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[9])

    #         (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (11): ReLU(inplace)
    #         (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (13): ReLU(inplace)
    #         (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (15): ReLU(inplace)
    #         (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[10])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[11])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[12])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[13])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[14])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[15])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[16])
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (18): ReLU(inplace)
    #         (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (20): ReLU(inplace)
    #         (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (22): ReLU(inplace)
    #         (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[17])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[18])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[19])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[20])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[21])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[22])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[23])
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (25): ReLU(inplace)
    #         (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (27): ReLU(inplace)
    #         (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (29): ReLU(inplace)
    #         (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[24])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[25])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[26])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[27])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[28])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[29])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[30])
    #         scores.append((relevant.clone(), irrelevant.clone()))

    relevant = relevant.view(relevant.size(0), -1)
    irrelevant = irrelevant.view(irrelevant.size(0), -1)

    #       (classifier): Sequential(
    #         (0): Linear(in_features=25088, out_features=4096)
    #         (1): ReLU(inplace)
    #         (2): Dropout(p=0.5)
    #         (3): Linear(in_features=4096, out_features=4096)
    #         (4): ReLU(inplace)
    #         (5): Dropout(p=0.5)
    #         (6): Linear(in_features=4096, out_features=1000)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[32])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[33])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[34])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[35])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[36])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[37])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[38])

    return relevant, irrelevant, scores
