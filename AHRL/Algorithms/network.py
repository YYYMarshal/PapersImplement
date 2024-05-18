import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, scale):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 300)
		self.l2 = nn.Linear(300, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.scale = nn.Parameter(torch.tensor(scale).float(), requires_grad=False)


	def forward(self, x):
		x = F.relu(self.l1(torch.cat([x], 1)))
		x = F.relu(self.l2(x))
		x = self.scale * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 300)
		self.l2 = nn.Linear(300, 300)
		self.l3 = nn.Linear(300, 1)
		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 300)
		self.l5 = nn.Linear(300, 300)
		self.l6 = nn.Linear(300, 1)

	def forward(self, x, u):
		x1 = F.relu(self.l1(torch.cat([x, u], 1)))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(torch.cat([x, u], 1)))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2

	def Q1(self, x, u):
		x1 = F.relu(self.l1(torch.cat([x, u], 1)))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1
