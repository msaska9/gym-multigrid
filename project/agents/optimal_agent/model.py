import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, num_actions=5):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.num_actions)
        )

    def forward(self, x):
        return self.network(x.float())


class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        self.nn = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)
        return out


class Q_learning(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, unit=hidden_unit, activation=F.relu):
        super(Q_learning, self).__init__()
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList()
        self.in_channels = in_channels
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(unit(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)

    def forward(self, x):
        print(x)
        out = x
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out
