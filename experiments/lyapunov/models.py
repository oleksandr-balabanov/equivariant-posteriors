import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass
class FeedConfig:
    input_dim: object
    hidden_dim: object
    output_dim: object
    hidden_layers: object
    sigma_W: object
    sigma_b: object
    activ_func: object

    def serialize_human(self):
        return self.__dict__


@dataclass
class FeedProjConfig:
    input_dim: object
    hidden_dim: object
    output_dim: object
    hidden_layers: object
    sigma_W: object
    sigma_b: object
    activ_func: object

    def serialize_human(self):
        return self.__dict__


# %% Create class for neural network
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, config: FeedConfig, data_spec: DataSpec):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Initialize architecture parameters
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.hidden_layers = config.hidden_layers

        # Set up architecture
        self.layers = []
        layer = nn.Linear(config.input_dim, config.hidden_dim)
        nn.init.normal_(layer.weight, std=np.sqrt(config.sigma_W / config.input_dim))
        nn.init.normal_(layer.bias, std=np.sqrt(config.sigma_b))
        self.layers.append(layer)
        for l in range(config.hidden_layers):
            if l < config.hidden_layers - 1:
                layer = nn.Linear(config.hidden_dim, config.hidden_dim)
                nn.init.normal_(
                    layer.weight, std=np.sqrt(config.sigma_W / config.hidden_dim)
                )
                nn.init.normal_(layer.bias, std=np.sqrt(config.sigma_b))
                self.layers.append(layer)
            else:
                layer = nn.Linear(config.hidden_dim, config.output_dim)
                nn.init.normal_(
                    layer.weight, std=np.sqrt(config.sigma_W / config.hidden_dim)
                )
                nn.init.normal_(layer.bias, std=np.sqrt(config.sigma_b))
                self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

        # Initialize activation function
        if config.activ_func == "ReLU":
            self.g = nn.ReLU()
        elif config.activ_func == "Hardtanh":
            self.g = nn.Hardtanh()
        elif config.activ_func == "Tanh":
            self.g = nn.Tanh()
        elif config.activ_func == "Linear":
            self.g = lambda x: x
        else:
            raise Exception("Specified activation function not supported.")

    def forward(self, x):
        out = x.reshape(-1, 28 * 28)
        # Obtain local field
        out = self.layers[0](out)

        # Initialize array storing the local fields of each layer
        self.outs = torch.empty(size=(out.shape[0], out.shape[1], self.hidden_layers))
        self.outs[:, :, 0] = out

        # Obtain output
        out = self.g(out)

        for l in range(1, self.hidden_layers + 1):
            # Obtain local field
            out = self.layers[l](out)
            if l < self.hidden_layers:
                # Save only local fields of hidden layers, not final layer
                self.outs[:, :, l] = out
                # Obtain output
                out = self.g(out)

        return out, self.output_to_value(out)

    # def forward_full(self, x):
    # return self.output_to_value(self.forward(x))

    def output_to_value(self, output):
        return torch.softmax(output, dim=-1)


class FeedforwardNeuralNetModel_proj(nn.Module):
    def __init__(self, config: FeedProjConfig, data_spec: DataSpec):
        super(FeedforwardNeuralNetModel_proj, self).__init__()

        (
            input_dim,
            hidden_dim,
            output_dim,
            hidden_layers,
            sigma_W,
            sigma_b,
            activ_func,
        ) = (
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.hidden_layers,
            config.sigma_W,
            config.sigma_b,
            config.activ_func,
        )

        # Initialize architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        # Set up architecture
        self.layers = []
        layer = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(layer.weight, std=np.sqrt(sigma_W / input_dim))
        nn.init.normal_(layer.bias, std=np.sqrt(sigma_b))
        self.layers.append(layer)
        for l in range(hidden_layers):
            if l < hidden_layers - 1:
                layer = nn.Linear(hidden_dim, hidden_dim)
                nn.init.normal_(layer.weight, std=np.sqrt(sigma_W / hidden_dim))
                nn.init.normal_(layer.bias, std=np.sqrt(sigma_b))
                self.layers.append(layer)
            else:
                layer = nn.Linear(hidden_dim, 2)
                nn.init.normal_(layer.weight, std=np.sqrt(sigma_W / hidden_dim))
                nn.init.normal_(layer.bias, std=np.sqrt(sigma_b))
                self.layers.append(layer)

        layer = nn.Linear(2, output_dim)
        nn.init.normal_(layer.weight, std=np.sqrt(sigma_W / 2))
        nn.init.normal_(layer.bias, std=np.sqrt(sigma_b))
        self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        # Initialize activation function
        if activ_func == "ReLU":
            self.g = nn.ReLU()
        elif activ_func == "Hardtanh":
            self.g = nn.Hardtanh()
        elif activ_func == "Tanh":
            self.g = nn.Tanh()
        elif activ_func == "Linear":
            self.g = lambda x: x
        else:
            raise Exception("Specified activation function not supported.")

    def forward(self, x):
        out = x.reshape(-1, 28 * 28)
        # Obtain local field
        out = self.layers[0](out)

        # Initialize array storing the local fields of each layer
        self.outs = torch.empty(size=(out.shape[0], out.shape[1], self.hidden_layers))
        self.outs[:, :, 0] = out

        # Obtain output
        out = self.g(out)

        for l in range(1, self.hidden_layers + 2):
            # Obtain local field
            out = self.layers[l](out)
            if l < self.hidden_layers + 1:
                # Save only local fields of hidden layers, not final layer
                if l < self.hidden_layers:
                    self.outs[:, :, l] = out
                    # Obtain output
                    out = self.g(out)
                if l == self.hidden_layers:
                    self.bottleneck = out

        return out, self.output_to_value(out)

    # def forward_full(self, x):
    # return self.output_to_value(self.forward(x))

    def output_to_value(self, output):
        return torch.softmax(output, dim=-1)
