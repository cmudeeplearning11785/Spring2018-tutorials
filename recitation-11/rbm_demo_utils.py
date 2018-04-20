import os
import numpy as np
import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from inferno.trainers.basic import Trainer
import matplotlib.pyplot as plt
import rbm_models


class MNIST(torch.utils.data.Dataset):
    def __init__(self, max_len=-1, include_label=True):
        super().__init__()
        self.mnist = datasets.MNIST(
            root='./data', train=True,
            download=True, transform=transforms.ToTensor())
        self.max_len = max_len
        self.include_label = include_label

    def __len__(self):
        if self.max_len < 0:
            return len(self.mnist)
        else:
            return self.max_len

    def __getitem__(self, idx):
        (img, label) = self.mnist[idx]
        img = img.view(-1)
        img = rbm_models.discretize(rbm_models.rescale(img))
        if not self.include_label:
            return (img, label)
        label_onehot = img.new(10).fill_(0)
        label_onehot[label] = 1
        label_onehot = rbm_models.rescale(label_onehot)
        result = torch.cat([img, label_onehot], dim=0)
        return (result, label)


class IdentityLoss(torch.nn.Module):
    def forward(self, x, _):
        return x


class LossPrinter(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        loss = self.criterion(*args, **kwargs)
        print("Loss: %f" % loss)
        return loss


def train(net, dataset, criterion, num_epochs,
          batch_size, learn_rate, dir_name):
    dir_name = os.path.join('net/', dir_name)
    trainer = Trainer(net[0])

    if (os.path.exists(os.path.join(dir_name, 'model.pytorch'))):
        net_temp = trainer.load_model(dir_name).model
        net[0].load_state_dict(net_temp.state_dict())
        print("Loaded checkpoint directly")
    else:
        if (not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=batch_size)
        net[0].train()

        trainer \
            .build_criterion(LossPrinter(criterion)) \
            .bind_loader('train', data_loader) \
            .build_optimizer('Adam', lr=learn_rate) \
            .set_max_num_epochs(num_epochs)

        if torch.cuda.is_available():
            trainer.cuda()

        trainer.fit()
        trainer.save_model(dir_name)
    net[0].cpu()
    net[0].eval()


def display_image(arr):
    width = int(np.sqrt(arr.size()[0]))
    label_onehot = arr[-10:]
    arr = (arr[:-10] + 1) / 2
    arr = arr.cpu().view(width, -1).numpy()
    plt.figure()
    plt.imshow(1.0 - arr, cmap='gray')
    _, pos = torch.max(label_onehot, 0)
    print(pos[0])


def display_reconstruction(net, dataset):
    (image, _) = dataset[np.random.randint(len(dataset))]
    display_image(image)
    image = torch.autograd.Variable(image).unsqueeze(dim=0)
    reconst = net.decode(net.encode(image)).data[0]
    display_image(reconst)
