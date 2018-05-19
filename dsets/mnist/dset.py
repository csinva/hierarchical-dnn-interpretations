from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from model import Net


# Training settings
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args("")


# load data
def load_data(train_batch_size, test_batch_size, use_cuda, data_dir='data', shuffle=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=shuffle, **kwargs)
    return train_loader, test_loader


def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    return model


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_im_and_label(num):
    torch.manual_seed(130)
    _, data_loader = load_data(train_batch_size=1, test_batch_size=1,
                               use_cuda=True, data_dir='mnist/data',
                               shuffle=False)
    for i, im in enumerate(data_loader):
        if i == num:
            return Variable(im[0].cuda()), im[0].numpy().squeeze(), im[1].numpy()[0]


def pred_ims(model, ims, layer='softmax'):
    if len(ims.shape) == 2:
        ims = np.expand_dims(ims, 0)
    ims_torch = Variable(torch.unsqueeze(torch.from_numpy(ims), 1)).float().cuda()
    preds = model(ims_torch)

    # todo - build in logit support
    # logits = model.logits(t)
    return preds.data.cpu().numpy()


if __name__ == '__main__':
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.batch_size, args.test_batch_size, args.cuda)

    # create model
    model = Net()
    if args.cuda:
        model.cuda()

    # train
    for epoch in range(1, args.epochs + 1):
        model = train(epoch, train_loader)
        test(model, test_loader)

    # save
    torch.save(model.state_dict(), 'mnist.model')
    # load and test
    # model_loaded = Net().cuda()
    # model_loaded.load_state_dict(torch.load('mnist.model'))
    # test(model_loaded, test_loader)
