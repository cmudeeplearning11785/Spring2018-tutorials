import numpy as np
import torch
import torch.utils.data


class OnesCounter(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        count_bitwidth = int(np.ceil(np.log2(input_size + 1)))
        self.to_hidden1 = torch.nn.Linear(input_size, 2 * input_size)
        self.hidden_sigmoid1 = torch.nn.Sigmoid()
        self.to_hidden2 = torch.nn.Linear(2 * input_size, 2 * input_size)
        self.hidden_sigmoid2 = torch.nn.Sigmoid()
        self.to_binary = torch.nn.Linear(2 * input_size, count_bitwidth)
        self.binary_sigmoid = torch.nn.Sigmoid()

    def forward(self, input_val):
        hidden1 = self.hidden_sigmoid1(self.to_hidden1(input_val))
        hidden2 = self.hidden_sigmoid2(self.to_hidden2(hidden1))
        return self.binary_sigmoid(self.to_binary(hidden2))


def load_data():
    # We'll just make our data on the spot here, but
    # we usually load real data sets from a file

    # Create 10000 random 7-bit inputs
    data = np.random.binomial(1, 0.5, size=(10000, 7))

    # Count the number of 1's in each input
    labels = data.sum(axis=1)

    # Create the binary encoding of the ground truth labels
    # As a bit of practice using Numpy, we're going to do this
    # without using a Python loop.
    labels_binary = np.zeros((10000, 3))
    labels_binary[:, 2] = (labels % 2 >= 1)
    labels_binary[:, 1] = (labels % 4 >= 2)
    labels_binary[:, 0] = (labels % 8 >= 4)

    return (data, labels_binary)


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def training_routine(num_epochs, minibatch_size, learn_rate):
    (data, labels_binary) = load_data()

    my_net = OnesCounter(7)  # Create the network,
    loss_fn = torch.nn.MSELoss()  # and choose the loss function / optimizer
    optim = torch.optim.SGD(my_net.parameters(), lr=learn_rate)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        my_net = my_net.cuda()
        loss_fn = loss_fn.cuda()

    dataset = torch.utils.data.TensorDataset(
        to_tensor(data), to_tensor(labels_binary))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=minibatch_size, shuffle=True)

    for _ in range(num_epochs):
        for (input_val, label) in data_loader:
            optim.zero_grad()  # Reset the gradients

            prediction = my_net(to_variable(input_val))  # Feed forward
            loss = loss_fn(prediction, to_variable(label))  # Compute losses
            loss.backward()  # Backpropagate the gradients

            optim.step()  # Update the network

    return my_net
