import argparse
import math
import sys

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms


def mnist_data_loader(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


class DiscriminatorNetwork(nn.Module):
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.cnn1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.bnn1 = nn.BatchNorm2d(64)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bnn2 = nn.BatchNorm2d(128)
        self.linear1 = nn.Linear(128 * 7 * 7, 1024)
        self.bnn3 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 1)
        self.act = nn.LeakyReLU()

    def forward(self, input):
        h = input
        h = self.cnn1(h)
        h = self.bnn1(h)
        h = self.act(h)
        h = self.cnn2(h)
        h = self.bnn2(h)
        h = self.act(h)
        h = h.view(h.size(0), -1)
        h = self.linear1(h)
        h = self.bnn3(h)
        h = self.act(h)
        h = self.linear2(h)
        h = h.squeeze(1)
        return h


class Reshape(nn.Module):
    def __init__(self, shape):
        self.shape=shape
    def forward(self, input):
        return input.view(*self.shape)

class GeneratorNetwork(nn.Module):
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.linear1 = nn.Linear(args.latent_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 7 * 7 * 128)
        self.bnn
        self.cnn1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cnn2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        h = input
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = h.view(-1, 128, 7, 7)
        h = self.cnn1(h)
        h = self.act(h)
        h = self.cnn2(h)
        h = self.act(h)
        h = self.cnn3(h)
        h = self.sigmoid(h)
        return h


def format_images(images):
    # convert (n, c, h, w) to a single image grid
    c = images.size(1)
    h = images.size(2)
    w = images.size(3)
    gridsize = int(math.floor(math.sqrt(images.size(0))))
    images = images[:gridsize * gridsize]
    images = images.view(gridsize, gridsize, c, h, w)
    images = images.permute(2, 0, 3, 1, 4).contiguous()
    images = images.view(1, c, gridsize * h, gridsize * w)
    return images


class GANModel(nn.Module):
    def __init__(self, args):
        super(GANModel, self).__init__()
        self.discriminator = DiscriminatorNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.latent_dim = args.latent_dim
        self._state_hooks = {}

    def y_fake(self, latent):
        xfake = self.generator(latent)
        self._state_hooks['generated_images'] = format_images(xfake)
        yfake = self.discriminator(xfake)
        return yfake

    def y_real(self, xreal):
        self._state_hooks['real_images'] = format_images(xreal)
        yreal = self.discriminator(xreal)
        return yreal

    def forward(self, xreal):
        latent = xreal.data.new(xreal.size(0), self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        return self.y_real(xreal), self.y_fake(latent)


class DiscriminatorLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        yreal, yfake = input
        real_targets = Variable(yreal.data.new(yreal.size(0)).fill_(1))
        fake_targets = Variable(yreal.data.new(yreal.size(0)).zero_())
        real_loss = super(DiscriminatorLoss, self).forward(yreal, real_targets)
        fake_loss = super(DiscriminatorLoss, self).forward(yfake, fake_targets)
        return real_loss + fake_loss


class GeneratorLoss(nn.BCEWithLogitsLoss):
    def forward(self, yfake):
        fake_targets = Variable(yfake.data.new(yfake.size(0)).fill_(1))
        fake_loss = super(GeneratorLoss, self).forward(yfake, fake_targets)
        return fake_loss


class GeneratorTrainingCallback(Callback):
    def __init__(self, args, parameters):
        self.criterion = GeneratorLoss()
        self.opt = Adam(parameters, args.generator_lr)
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.count = 0
        self.frequency = args.generator_frequency

    def end_of_training_iteration(self, **_):
        self.count += 1
        if self.count > self.frequency:
            self.train_generator()
            self.count = 0

    def train_generator(self):
        if self.trainer.is_cuda:
            latent = torch.cuda.FloatTensor(self.batch_size, self.latent_dim)
        else:
            latent = torch.FloatTensor(self.batch_size, self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        yfake = self.trainer.model.y_fake(latent)
        self.opt.zero_grad()
        loss = self.criterion(yfake)
        loss.backward()
        self.opt.step()


def run(args):
    train_loader = mnist_data_loader(args)
    model = GANModel(args)

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(DiscriminatorLoss)
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)
    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(10)
    logger = TensorboardLogger(log_scalars_every=(1, 'iteration'), log_images_every=(100, 'iteration'))
    trainer.build_logger(logger, log_directory=args.save_directory)
    logger.observe_state('generated_images')
    logger.observe_state('real_images')
    logger._trainer_states_being_observed_while_training.remove('training_inputs')
    trainer.register_callback(GeneratorTrainingCallback(args, model.generator.parameters()))
    trainer.bind_loader('train', train_loader)

    if args.cuda:
        trainer.cuda()

    trainer.fit()


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--save-directory', type=str, default='output/mnist_gan/v2', help='output directory')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs')
    parser.add_argument('--generator-frequency', type=int, default=5, metavar='N', help='number of epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--latent-dim', type=int, default=100, metavar='N', help='latent dimension')
    parser.add_argument('--discriminator-lr', type=float, default=1e-4, metavar='N', help='latent dimension')
    parser.add_argument('--generator-lr', type=float, default=3e-4, metavar='N', help='latent dimension')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
