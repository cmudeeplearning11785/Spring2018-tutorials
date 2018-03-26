from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot

from inferno.trainers.basic import Trainer


def make_numeral_arrs(path):
    arrs = [1 - np.array(
        Image.open(path + '/' + str(v) + '.gif').convert('L')) // 255
        for v in range(10)]
    np.save("data/numbers.npy", arrs)


def display_image(arr):
    arr = np.transpose(np.array(
        arr[:3, :, :].cpu().numpy() * 255, dtype='uint8'), [1, 2, 0])
    matplotlib.pyplot.imshow(arr)


def display_attention(var):
    arr = np.log(var.data.numpy() + 1e-9)
    matplotlib.pyplot.imshow(arr)


word_dict = {"zero": 0, "one": 1, "two": 2, "three": 3,
             "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9,
             "red": 10, "green": 11, "blue": 12,
             "left": 13, "middle": 14, "right": 15}
word_inv_dict = {v: k for (k, v) in word_dict.items()}


def to_words(labels):
    return ' '.join([word_inv_dict[l] for l in labels])


class NumberColorsDataset(torch.utils.data.Dataset):

    def __init__(self):
        torch.utils.data.Dataset.__init__(self)
        self.numeral_arrs = np.load("data/numbers.npy")
        self.color_ord_list = np.array([
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]])

    def __len__(self):
        return 6000

    def __getitem__(self, idx):
        number = idx % 1000
        color_ord_idx = idx // 1000

        digits = np.zeros(3, dtype='i')
        digits[2] = number % 10
        number //= 10
        digits[1] = number % 10
        number //= 10
        digits[0] = number % 10
        color_ord = self.color_ord_list[color_ord_idx]

        imgs = np.array([
            self.numeral_arrs[digits[0]],
            self.numeral_arrs[digits[1]],
            self.numeral_arrs[digits[2]]])

        (_, h, w) = imgs.shape
        colored = np.zeros((3, 4, h, w))
        colored[0, color_ord[0], :, :] = imgs[0]
        colored[1, color_ord[1], :, :] = imgs[1]
        colored[2, color_ord[2], :, :] = imgs[2]

        combined = np.concatenate(colored, axis=2)
        combined[3, :, :] = np.linspace(0.0, 1.0, 3 * w).reshape(1, 1, 3 * w)

        color_inv_ord = np.argsort(color_ord)

        labels = []
        labels.append(word_dict["red"])
        labels.append(digits[color_inv_ord[0]])
        labels.append(color_inv_ord[0] + word_dict["left"])
        labels.append(word_dict["green"])
        labels.append(digits[color_inv_ord[1]])
        labels.append(color_inv_ord[1] + word_dict["left"])
        labels.append(word_dict["blue"])
        labels.append(digits[color_inv_ord[2]])
        labels.append(color_inv_ord[2] + word_dict["left"])
        # labels = [digits[0]]

        label_vec = torch.from_numpy(np.array(labels)).long()

        return torch.from_numpy(combined).float(), len(labels), \
            label_vec, label_vec


class NumberColorsNet(torch.nn.Module):
    def __init__(self, num_queries=4, query_size=16, val_size=64):
        super().__init__()

        # The convolutional layers encode the input
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Conv2d(
            4, 64, kernel_size=3, padding=0, stride=2))
        self.convs.append(torch.nn.Conv2d(
            64, 128, kernel_size=3, padding=0, stride=2))
        self.convs.append(torch.nn.Conv2d(
            128, 256, kernel_size=3, padding=0))
        self.convs.append(torch.nn.Conv2d(
            256, 256, kernel_size=3, padding=0))
        self.key_conv = torch.nn.Conv2d(
            256, query_size, kernel_size=3, padding=1)
        self.val_conv = torch.nn.Conv2d(
            256, val_size, kernel_size=3, padding=1)

        self.test = torch.nn.Linear(query_size, len(word_dict))

        # Embedding to convert output labels to RNN input
        self.embedding = torch.nn.Embedding(len(word_dict), len(word_dict))

        # The hidden state of the RNN layer is used as the query
        self.rnns = torch.nn.ModuleList()
        self.rnn_inith = torch.nn.ParameterList()
        self.rnn_initc = torch.nn.ParameterList()

        queries_total = num_queries * query_size

        self.rnns.append(torch.nn.LSTMCell(
            val_size * num_queries + len(word_dict), 128))
        self.rnn_inith.append(torch.nn.Parameter(torch.rand(1, 128)))
        self.rnn_initc.append(torch.nn.Parameter(torch.rand(1, 128)))

        self.rnns.append(torch.nn.LSTMCell(128, 64))
        self.rnn_inith.append(torch.nn.Parameter(torch.rand(1, 64)))
        self.rnn_initc.append(torch.nn.Parameter(torch.rand(1, 64)))

        # Linear layers convert the hidden state to the query
        self.query_linears = torch.nn.ModuleList()
        self.query_linears.append(torch.nn.Linear(64, 64))
        self.query_linears.append(torch.nn.Linear(64, queries_total))

        # Linear layers convert the hidden state to the output
        self.output_linears = torch.nn.ModuleList()
        self.output_linears.append(torch.nn.Linear(64, 64))
        self.output_linears.append(torch.nn.Linear(64, len(word_dict)))

        # Leaky relu activation
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        for param in self.parameters():
            if param.ndimension() >= 2:
                torch.nn.init.xavier_uniform(param)
            else:
                param.data.zero_()

        self.attentions = []
        self.num_queries = num_queries
        self.query_size = query_size

    def forward(self, img, num_iters, label_batch):
        for conv in self.convs:
            img = self.leaky_relu(conv(img))
        key = self.key_conv(img)
        val = self.val_conv(img)
        (b, c, w, h) = val.size()

        # return [self.test(img.view(b, c, w * h).mean(dim=2))], num_iters

        outputs = []
        self.attentions = []

        output_embed = img.data.new(b).long()
        output_embed[:] = 2
        output_embed = self.embedding(
            torch.autograd.Variable(output_embed))
        hidden = [h.repeat(b, 1) for h in self.rnn_inith]
        cell = [c.repeat(b, 1) for c in self.rnn_initc]

        for i in range(num_iters.max().data[0]):
            queries = hidden[-1]
            for (i, linear) in enumerate(self.query_linears):
                if i == len(self.query_linears) - 1:
                    queries = linear(queries)
                else:
                    queries = self.leaky_relu(linear(queries))
            queries = queries.view((b, self.query_size, self.num_queries))

            attention = torch.matmul(torch.transpose(
                key.view((b, self.query_size, w * h)), 1, 2), queries)
            attention = F.elu(attention) + 1.0
            attention /= (1e-6 + attention.sum(dim=1, keepdim=True))
            self.attentions.append(torch.transpose(
                attention, 1, 2).contiguous().view(
                (b, self.num_queries, w, h)))

            fused = torch.matmul(val.view((b, c, w * h)), attention).view(
                (b, c * self.num_queries))

            rnn_input = torch.cat([fused, output_embed], dim=1)

            for (j, rnn) in enumerate(self.rnns):
                hidden[j], cell[j] = rnn(rnn_input, (hidden[j], cell[j]))
                rnn_input = hidden[j]

            output = hidden[-1]
            for (i, linear) in enumerate(self.output_linears):
                if i == len(self.output_linears) - 1:
                    output = linear(output)
                else:
                    output = self.leaky_relu(linear(output))
            outputs.append(output)

            if self.training:
                output_embed = self.embedding(label_batch[:, i])
            else:
                output_embed = self.embedding(output.max(dim=1)[1])

        return torch.stack(outputs, dim=1), num_iters

    def get_attentions(self):
        return self.attentions


class SeqCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = 0.0
        self.count = 0

    def forward(self, val, label_batch):
        (outputs, num_iters) = val
        # loss = F.cross_entropy(outputs[0], label_batch[:, 0])
        # print(loss)

        output_list = []
        label_list = []
        for (output, labels, num_iter) in zip(
                outputs, label_batch, num_iters):
            num_iter = num_iter.data[0]
            output_list.append(output[:num_iter])
            label_list.append(labels[:num_iter])

        outputs = torch.cat(output_list)
        labels = torch.cat(label_list)
        loss = torch.nn.functional.cross_entropy(
            outputs, labels)

        self.losses += loss.data[0]
        self.count += 1
        if self.count % 10 == 0:
            print(self.losses / self.count)
            self.losses = 0.0
            self.count = 0
        return loss


def train(net, num_epochs, dataset):
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=100)
    net.train(mode=True)

    for i in range(num_epochs):
        lr = 0.002 / (1 + 0.2 * i)
        trainer = Trainer(net) \
            .build_criterion(SeqCrossEntropyLoss) \
            .build_optimizer('Adam', lr=lr) \
            .set_max_num_epochs(1) \
            .save_every((10, 'iterations')) \
            .save_to_directory('net/')

        trainer.bind_loader('train', data_loader, num_inputs=3)

        if torch.cuda.is_available():
            trainer.cuda()

        trainer.fit()


def main():
    net = NumberColorsNet()
    dataset = NumberColorsDataset()
    train(net, 200, dataset)
