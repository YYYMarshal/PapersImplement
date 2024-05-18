import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(), requires_grad=False)

    def forward(self, x):
        x = F.relu(self.fc1(torch.cat([x], 1)))
        x = F.relu(self.fc2(x))
        x = self.scale * torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.Q1_fc1 = nn.Linear(state_dim + action_dim, 300)
        self.Q1_fc2 = nn.Linear(300, 300)
        self.Q1_fc3 = nn.Linear(300, 1)
        # Q2 architecture
        self.Q2_fc1 = nn.Linear(state_dim + action_dim, 300)
        self.Q2_fc2 = nn.Linear(300, 300)
        self.Q2_fc3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x1 = F.relu(self.Q1_fc1(torch.cat([x, u], 1)))
        x1 = F.relu(self.Q1_fc2(x1))
        x1 = self.Q1_fc3(x1)

        x2 = F.relu(self.Q2_fc1(torch.cat([x, u], 1)))
        x2 = F.relu(self.Q2_fc2(x2))
        x2 = self.Q2_fc3(x2)
        return x1, x2

    def get_q1_value(self, x, u):
        x1 = F.relu(self.Q1_fc1(torch.cat([x, u], 1)))
        x1 = F.relu(self.Q1_fc2(x1))
        x1 = self.Q1_fc3(x1)
        return x1
