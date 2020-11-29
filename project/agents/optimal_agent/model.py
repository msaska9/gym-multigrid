import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, self.num_actions)
        )

    def forward(self, x):
        return self.network(x.float())
