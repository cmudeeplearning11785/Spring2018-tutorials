import torch

FLIP_PROB = 0.5


def rescale(x):
    return 2 * x - 1


def gumbel_noise_logits(template):
    canvas = torch.zeros_like(template)
    canvas.uniform_()
    return -torch.log(canvas)


def noisy_fields(fields, temperature):
    add_noise = gumbel_noise_logits(fields)
    sub_noise = gumbel_noise_logits(fields)
    return fields + (add_noise - sub_noise) * temperature


def discretize(values):
    return rescale((values > 0).float())


def _update(values, fields):
    products = values * fields
    rands = rescale(torch.zeros_like(values).uniform_())
    multipliers = -rescale(((products < 0) * (rands < FLIP_PROB)).float())
    values *= multipliers


class AbstractLoopNet(torch.nn.Module):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units
        self.weights = None

    def init_weights(self, fix_weights=True):
        torch.nn.init.xavier_normal(self.weights)
        if fix_weights:
            self.fix_weights()

    def fix_weights(self):
        weights = self.weights.data.clone()
        mask = (1 - torch.ones_like(weights[0]).diag())
        weights = weights + torch.transpose(weights, 0, 1)
        weights = mask * (weights / 2)
        self.weights.data.copy_(weights)

    def compute_fields(self, vals):
        raise NotImplementedError("Abstract function")

    def compute_energy(self, vals):
        fields = self.compute_fields(vals)
        return -0.5 * torch.sum(vals.data * fields.data, dim=1)

    def update(self, vals, fields):
        raise NotImplementedError("Abstract function")

    def run_iters(self, in_vals, num_iters):
        vals = in_vals.clone()
        for i in range(num_iters):
            fields = self.compute_fields(vals)
            self.update(vals.data, fields.data)
        return vals

    def forward(self, in_vals, num_iters):
        return self.run_iters(in_vals, num_iters)

    def inference(self, in_vals, num_iters, num_inference_iters):
        vals = self.run_iters(in_vals, num_iters)

        vals_avg = torch.zeros_like(vals)
        for i in range(num_inference_iters):
            fields = self.compute_fields(vals)
            self.update(vals.data, fields.data)
            vals_avg.data += vals.data
        vals_avg = vals_avg / num_inference_iters
        return discretize(vals_avg)


class HopfieldNet(AbstractLoopNet):
    def __init__(self, num_units):
        super().__init__(num_units)
        self.weights = torch.nn.Parameter(
            torch.Tensor(num_units, num_units))
        self.init_weights()

    def compute_fields(self, vals):
        return torch.matmul(
            self.weights.unsqueeze(dim=0),
            vals.unsqueeze(dim=2))[:, :, 0]

    def update(self, vals, fields):
        _update(vals, fields)


class StochasticHopfieldNet(AbstractLoopNet):
    def __init__(self, num_units, temperature):
        super().__init__(num_units)
        self.weights = torch.nn.Parameter(
            torch.Tensor(num_units, num_units))
        self.temperature = temperature
        self.init_weights()

    def compute_fields(self, vals):
        return torch.matmul(
            self.weights.unsqueeze(dim=0),
            vals.unsqueeze(dim=2))[:, :, 0]

    def update(self, vals, fields):
        _update(vals, noisy_fields(fields, self.temperature))


class BoltzmannMachine(AbstractLoopNet):
    def __init__(self, num_units, num_hidden,
                 temperature, hidden_setup_iters):
        super().__init__(num_units)
        self.num_hidden = num_hidden
        self.weights = torch.nn.Parameter(
            torch.Tensor(num_units + num_hidden, num_units + num_hidden))
        self.temperature = temperature
        self.hidden_setup_iters = hidden_setup_iters
        self.init_weights()

    def compute_fields(self, vals):
        return torch.matmul(
            self.weights.unsqueeze(dim=0),
            vals.unsqueeze(dim=2))[:, :, 0]

    def update(self, vals, fields, include_visible=True):
        old_vals = vals[:, :self.num_units].clone()
        _update(vals, noisy_fields(fields, self.temperature))

        if not include_visible:
            vals[:, :self.num_units] = old_vals

    def expand_data(self, in_visible_vals, num_samples):
        batch_size = in_visible_vals.size()[0]
        visible_vals = discretize(in_visible_vals)
        hidden_vals = torch.autograd.Variable(
            visible_vals.data.new(batch_size, self.num_hidden))
        vals_list = []

        for i in range(num_samples):
            hidden_vals.data.uniform_()
            hidden_vals.data.copy_(discretize(rescale(hidden_vals.data)))
            vals = torch.cat([visible_vals, hidden_vals], dim=1)
            for i in range(self.hidden_setup_iters):
                fields = self.compute_fields(vals)
                self.update(vals.data, fields.data, False)
            vals_list.append(vals)
        return torch.cat(vals_list, dim=0)


class RestrictedBoltzmannMachine(AbstractLoopNet):
    def __init__(self, num_units, num_hidden, temperature):
        super().__init__(num_units)
        self.num_hidden = num_hidden
        self.weights = torch.nn.Parameter(
            torch.Tensor(num_hidden, num_units))
        self.temperature = temperature
        self.init_weights(fix_weights=False)

    def compute_fields(self, vals):
        hidden = torch.matmul(
            self.weights.unsqueeze(dim=0),
            vals[:, :self.num_units].unsqueeze(dim=2))[:, :, 0]
        visible = torch.matmul(
            torch.transpose(self.weights, 0, 1).unsqueeze(dim=0),
            vals[:, self.num_units:].unsqueeze(dim=2))[:, :, 0]
        return torch.cat([visible, hidden], dim=1)

    def update(self, vals, fields):
        _update(vals, noisy_fields(fields, self.temperature))

    def expand_data(self, in_visible_vals):
        batch_size = in_visible_vals.size()[0]
        visible_vals = discretize(in_visible_vals)
        hidden_vals = torch.autograd.Variable(
            visible_vals.data.new(batch_size, self.num_hidden).fill_(1))

        vals = torch.cat([visible_vals, hidden_vals], dim=1)
        fields = self.compute_fields(vals)
        self.update(vals.data, fields.data)
        vals[:, :self.num_units] = in_visible_vals
        return vals


class HopfieldTrainWrapper(torch.nn.Module):
    def __init__(self, net, num_evolve_iters):
        super().__init__()
        assert isinstance(net, HopfieldNet)
        self.net = net
        self.num_evolve_iters = num_evolve_iters

    def forward(self, in_vals):
        self.net.fix_weights()
        batch_size = in_vals.size()[0]
        evolved_vals = self.net(in_vals, self.num_evolve_iters)

        positive_term = 1 / batch_size * torch.matmul(
            torch.transpose(in_vals, 0, 1), in_vals)
        negative_term = 1 / batch_size * torch.matmul(
            torch.transpose(evolved_vals, 0, 1), evolved_vals)
        weight_neg_grad = -(positive_term - negative_term)

        result = (self.net.weights *
                  weight_neg_grad).sum() / self.net.num_units
        if (result.data[0] < 0):
            return result * 0
        else:
            return result


class StochasticHopfieldTrainWrapper(torch.nn.Module):
    def __init__(self, net, num_rand_samples, num_rand_sample_iters):
        super().__init__()
        assert isinstance(net, StochasticHopfieldNet)
        self.net = net
        self.num_rand_samples = num_rand_samples
        self.num_rand_sample_iters = num_rand_sample_iters

    def forward(self, in_vals):
        self.net.fix_weights()
        batch_size = in_vals.size()[0]
        rand_starts = torch.autograd.Variable(discretize(rescale(
            torch.rand(self.num_rand_samples, self.net.num_units))))
        sample_vals = self.net(rand_starts, self.num_rand_sample_iters)

        positive_term = 1 / batch_size * torch.matmul(
            torch.transpose(in_vals, 0, 1), in_vals)
        negative_term = 1 / self.num_rand_samples * torch.matmul(
            torch.transpose(sample_vals, 0, 1), sample_vals)
        weight_neg_grad = -(positive_term - negative_term)

        result = (self.net.weights *
                  weight_neg_grad).sum() / self.net.num_units
        if (result.data[0] < 0):
            return result * 0
        else:
            return result


class BoltzmannTrainWrapper(torch.nn.Module):
    def __init__(self, net, num_train_samples,
                 num_rand_samples, num_rand_sample_iters):
        super().__init__()
        assert isinstance(net, BoltzmannMachine)
        self.net = net
        self.num_train_samples = num_train_samples
        self.num_rand_samples = num_rand_samples
        self.num_rand_sample_iters = num_rand_sample_iters

    def forward(self, in_visible_vals):
        self.net.fix_weights()
        batch_size = in_visible_vals.size()[0]
        batch_size_expand = batch_size * self.num_train_samples
        num_total_units = self.net.num_units + self.net.num_hidden
        in_vals = self.net.expand_data(
            in_visible_vals, self.num_train_samples)

        rand_starts = torch.autograd.Variable(discretize(
            rescale(torch.rand(self.num_rand_samples, num_total_units))))
        sample_vals = self.net(rand_starts, self.num_rand_sample_iters)

        positive_term = 1 / batch_size_expand * torch.matmul(
            torch.transpose(in_vals, 0, 1), in_vals)
        negative_term = 1 / self.num_rand_samples * torch.matmul(
            torch.transpose(sample_vals, 0, 1), sample_vals)
        weight_neg_grad = -(positive_term - negative_term)

        result = (self.net.weights *
                  weight_neg_grad).sum() / num_total_units
        if (result.data[0] < 0):
            return result * 0
        else:
            return result


class RestrictedBoltzmannTrainWrapper(torch.nn.Module):
    def __init__(self, net, num_evolve_iters):
        super().__init__()
        assert isinstance(net, RestrictedBoltzmannMachine)
        self.net = net
        self.num_evolve_iters = num_evolve_iters

    def forward(self, in_visible_vals):
        batch_size = in_visible_vals.size()[0]
        in_vals = self.net.expand_data(in_visible_vals)
        evolved_vals = self.net(in_vals, self.num_evolve_iters)

        positive_term = 1 / batch_size * torch.matmul(
            torch.transpose(in_vals[:, self.net.num_units:], 0, 1),
            in_vals[:, :self.net.num_units])
        negative_term = 1 / batch_size * torch.matmul(
            torch.transpose(evolved_vals[:, self.net.num_units:], 0, 1),
            evolved_vals[:, :self.net.num_units])
        weight_neg_grad = -(positive_term - negative_term)

        avg_units = (self.net.num_units + self.net.num_hidden) / 2
        result = (self.net.weights *
                  weight_neg_grad).sum() / avg_units
        if (result.data[0] < 0):
            return result * 0
        else:
            return result
