import torch
import numpy as np
import torch.nn as nn
from Algorithms import getTensor
from Algorithms.network import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AHRL(object):
    def __init__(self, state_dim, action_dim, scale, args):
        self.actor = Actor(state_dim, action_dim, scale)
        self.actor_target = Actor(state_dim, action_dim, scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0.0001)


        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scale = scale
        self.args = args


    def select_action(self, state, to_numpy=True):
        state = getTensor.get_tensor(state)

        if to_numpy:
            return self.actor(state).cpu().data.numpy().squeeze()
        else:
            return self.actor(state).squeeze()


    def train(self,  weight, replay_buffer, iterations, args):
        for it in range(iterations):
            state, next_state, episode_num, u, intrinsic_reward, d= replay_buffer.sample(args.batch_size)

            "The weighted intrinsic_reward"
            weight_row = episode_num[:, 0]
            weight_col = episode_num[:, 1]
            weight_temp = weight[weight_row, weight_col]
            weight_temp = weight_temp.reshape(args.batch_size, 1)
            intrinsic_reward = intrinsic_reward.reshape(args.batch_size, 1)
            reward = np.multiply(weight_temp,intrinsic_reward)
            """"""""""""

            state = getTensor.get_tensor(state)
            next_state = getTensor.get_tensor(next_state)
            action = getTensor.get_tensor(u)
            done = getTensor.get_tensor(1 - d)
            reward = getTensor.get_tensor(reward)

            noise = torch.FloatTensor(u).data.normal_(0, self.args.policy_noise).to(device)
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * args.discount * target_Q)
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = self.criterion(current_Q1, target_Q.detach()) + self.criterion(current_Q2, target_Q.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.args.policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_Actor.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_Actor.pth' % (directory, filename)))
