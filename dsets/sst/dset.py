import torch
import numpy as np
from torchtext import data, datasets, vocab
import random
import os
import pickle as pkl
import sys
path_to_file = os.path.dirname(__file__)

# deal with different torchtext versions
try:
    vocab._default_unk_index
except AttributeError:
    def _default_unk_index():
        return 0
vocab._default_unk_index = _default_unk_index


# set up data loaders
def get_sst():
    inputs = data.Field(lower='preserve-case')
    answers = data.Field(sequential=False, unk_token=None)

    # build with subtrees so inputs are right
    train_s, dev_s, test_s = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=True,
                                                 filter_pred=lambda ex: ex.label != 'neutral')
    inputs.build_vocab(train_s, dev_s, test_s)
    answers.build_vocab(train_s)

    # rebuild without subtrees to get longer sentences
    train, dev, test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=False,
                                           filter_pred=lambda ex: ex.label != 'neutral')

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=1, device=0)

    return inputs, answers, train_iter, dev_iter


# get specific batches
def get_batches(batch_nums, train_iterator, dev_iterator, dset='dev'):
    print('getting batches...')
    np.random.seed(13)
    random.seed(13)

    # pick data_iterator
    if dset == 'train':
        data_iterator = train_iterator
    elif dset == 'dev':
        data_iterator = dev_iterator

    # actually get batches
    num = 0
    batches = {}
    data_iterator.init_epoch()
    for batch_idx, batch in enumerate(data_iterator):
        if batch_idx == batch_nums[num]:
            batches[batch_idx] = batch
            num += 1

        if num == max(batch_nums):
            break
        elif num == len(batch_nums):
            print('found them all')
            break
    return batches

def load_vocab():
    return pkl.load(open(os.path.join(path_to_file, 'sst_vocab.pkl'), 'rb'))

def load_model():
    model = LSTMSentiment()
    
def batch_from_str_list(s, vocab, device='cpu'):
    '''Put text into .text attribute of a batch
    '''
    batch = lambda: None # placeholder which holds .text attribute
    nums = np.expand_dims(np.array([vocab['stoi'][x] for x in s]).transpose(),
                          axis=1)
    batch.text = torch.LongTensor(nums).to(device) #cuda()
    return batch