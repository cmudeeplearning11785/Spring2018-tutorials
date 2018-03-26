import argparse
import os
import sys

import numpy as np
import torch
from inferno.extensions.metrics.categorical import CategoricalError
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


def read_corpus():
    filename = 't8.shakespeare.txt'
    lines = []
    with open(filename, 'r') as f:
        for pos, line in enumerate(f):
            if 243 < pos < 124440:
                if len(line.strip()) > 0:
                    lines.append(line)
    corpus = " ".join(lines)
    return corpus


def get_charmap(corpus):
    chars = list(set(corpus))
    chars.sort()
    charmap = {c: i for i, c in enumerate(chars)}
    return chars, charmap


def map_corpus(corpus, charmap):
    return np.array([charmap[c] for c in corpus], dtype=np.int64)


def batchify(array, args):
    batch_len = args.batch_len
    batches = array.shape[0] // batch_len
    array = array[:batches * batch_len]
    return array.reshape((batches, batch_len))


def make_inputs(targets):
    # batches: (n, batch_len)
    return np.pad(targets[:, :-1] + 1, [(0, 0), (1, 0)], mode='constant')


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


class TextModel(nn.Module):
    def __init__(self, charcount, args):
        super(TextModel, self).__init__()
        self.charcount = charcount
        self.embedding = nn.Embedding(num_embeddings=charcount + 1, embedding_dim=args.embedding_dim)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.embedding_dim, batch_first=True)])
        self.projection = nn.Linear(in_features=args.embedding_dim, out_features=charcount)

    def forward(self, input, forward=0, stochastic=False):
        h = input  # (n, t)
        h = self.embedding(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.projection(h)
                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits


def generate(model, sequence_length, batch_size, args, stochastic=False, inp=None):
    if inp is None:
        inp = Variable(torch.zeros(batch_size, 1)).long()
        if args.cuda:
            inp = inp.cuda()
    model.eval()
    logits = model(inp, forward=sequence_length, stochastic=stochastic)
    classes = torch.max(logits, dim=2)[1]
    return classes


class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super(CrossEntropyLoss3D, self).forward(input.view(-1, input.size()[2]), target.view(-1))


class CategoricalError3D(CategoricalError):
    def forward(self, prediction, target):
        return super(CategoricalError3D, self).forward(prediction.view(-1, prediction.size()[2]), target.view(-1))


class CustomLogger(Callback):
    def end_of_training_iteration(self, **_):
        training_loss = self.trainer.get_state('training_loss', default=0)
        training_error = self.trainer.get_state('training_error', default=0)
        print("Training loss: {} error: {}".format(training_loss.numpy()[0], training_error))


def to_text(preds, charset):
    return ["".join(charset[c] for c in line) for line in preds]


def print_generated(lines):
    for i, line in enumerate(lines):
        print("Generated text {}: {}".format(i, line))


def train_model(model, dataset, args):
    kw = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, **kw)
    trainer = Trainer(model) \
        .build_criterion(CrossEntropyLoss3D) \
        .build_metric(CategoricalError3D) \
        .build_optimizer('Adam', weight_decay=1e-6) \
        .save_every((1, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        .register_callback(CustomLogger) \
        .bind_loader('train', loader)

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()


def main(argv):
    # Argparse
    parser = argparse.ArgumentParser(description='PyTorch Shakespeare Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--save-directory', type=str, default='output/shakespeare',
                        help='output directory')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-len', type=int, default=200, metavar='N',
                        help='Batch length')
    parser.add_argument('--hidden-dim', type=int, default=256, metavar='N',
                        help='Hidden dim')
    parser.add_argument('--embedding-dim', type=int, default=128, metavar='N',
                        help='Embedding dim')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Read and process data
    corpus = read_corpus()
    print("Corpus: {}...{}".format(corpus[:50], corpus[-50:]))
    print("Total character count: {}".format(len(corpus)))
    chars, charmap = get_charmap(corpus)
    charcount = len(chars)
    print("Unique character count: {}".format(len(chars)))
    array = map_corpus(corpus, charmap)
    targets = batchify(array, args=args)
    inputs = make_inputs(targets)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))

    # Train or load a model
    checkpoint_path = os.path.join(args.save_directory, 'checkpoint.pytorch')
    if not os.path.exists(checkpoint_path):
        model = TextModel(charcount=charcount, args=args)
        train_model(model=model, dataset=dataset, args=args)
    else:
        trainer = Trainer().load(from_directory=args.save_directory)
        model = TextModel(charcount=charcount, args=args)
        model.load_state_dict(trainer.model.state_dict())
        if args.cuda:
            model = model.cuda()

    # Generate deterministic text

    print("Deterministic")
    generated = generate(model, sequence_length=1000, batch_size=2, stochastic=False, args=args).data.cpu().numpy()
    print_generated(to_text(preds=generated, charset=chars))

    # Seed deterministic text
    seeds = ['KING RICHARD', 'KING RICHARD', 'Enter Falsta', 'SHAKESPEARE ']
    assert len(set(len(s) for s in seeds)) == 1
    inp = np.array([[charmap[c] for c in l] for l in seeds], dtype=np.int64)
    inp = np.pad(inp + 1, [(0, 0), (1, 0)], mode='constant')
    inp = Variable(torch.from_numpy(inp))
    if args.cuda:
        inp = inp.cuda()
    # Generate stochastic text
    generated = generate(model, sequence_length=2000, batch_size=5, stochastic=False, inp=inp,
                         args=args).data.cpu().numpy()
    text = to_text(preds=generated, charset=chars)
    for i, (s, t) in enumerate(zip(seeds, text)):
        print("Deterministic #{} (seed={}): {}".format(i, s, t))

    # Generate stochastic text
    print("Stochastic")
    generated = generate(model, sequence_length=1000, batch_size=5, stochastic=True, args=args).data.cpu().numpy()
    print_generated(to_text(preds=generated, charset=chars))


if __name__ == '__main__':
    main(sys.argv[1:])
