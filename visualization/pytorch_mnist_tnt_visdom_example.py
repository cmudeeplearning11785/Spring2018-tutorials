import argparse
import sys

import torch
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
from torchnet.engine import Engine
from torchnet.logger.visdomlogger import VisdomPlotLogger, VisdomLogger
from torchvision import datasets, transforms
from tqdm import tqdm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)


def model_fn():
    return nn.Sequential(
        Flatten(),
        nn.Linear(in_features=784, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10)
    )


def mnist_data_loaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print(kwargs)
    return train_loader, test_loader


def train_model(args):
    model = model_fn()
    if args.cuda:
        model = model.cuda()
    train_loader, validate_loader = mnist_data_loaders(args)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter()
    confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)

    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Loss'})
    train_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Class Error'})
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Loss'})
    test_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Class Error'})
    confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(10)),
                                                                'rownames': list(range(10))})

    def train_fn(sample):
        inputs = sample[0]
        targets = sample[1]
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs.float() / 255.0)
        targets = Variable(targets)
        output = model(inputs)
        loss = criterion(output, targets)
        return loss, output

    def reset_meters():
        classerr.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])
        confusion_meter.add(state['output'].data,
                            torch.LongTensor(state['sample'][1]))

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_err_logger.log(state['epoch'], classerr.value()[0])

        # do validation at the end of each epoch
        reset_meters()
        engine.test(train_fn, validate_loader)
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_err_logger.log(state['epoch'], classerr.value()[0])
        confusion_logger.log(confusion_meter.value())
        print('Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(train_fn, train_loader, maxepoch=args.epochs, optimizer=optimizer)


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train_model(args)


if __name__ == '__main__':
    main(sys.argv[1:])
