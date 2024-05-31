import numpy as np
import torch
import random
import math


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, km_num, max_size=int(18e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.km_num = km_num
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample1(self, batch_size, temp_number):
        ind = np.zeros((0), dtype=int)
        for i in range(len(temp_number) - 1):
            ind1 = random.sample(range(temp_number[i], temp_number[i + 1]), int(math.ceil(batch_size / self.km_num)))
            ind = np.hstack((ind1, ind))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sampleall(self):
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device)
        )

    def Choose_sample(self, result):
        index = []
        for i in range(self.km_num):
            index0 = np.where(result == i)
            index0 = np.array(index0)
            index0 = index0.tolist()
            index0 = index0[0]
            index0 = np.array(index0)
            index.append(index0)

        return index

    def sample_ind(self, ind):
        sample = []
        for i in range(len(ind)):
            temp_sample = [self.state[ind[i]], self.action[ind[i]], self.next_state[ind[i]], self.reward[ind[i]],
                           self.not_done[ind[i]]]
            sample.append(temp_sample)
        return sample


class ReplayBuffer1(object):
    def __init__(self, state_dim, action_dim, km_num, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.km_num = km_num
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample1(self, batch_size, temp_number):
        ind = np.zeros((0), dtype=int)
        for i in range(len(temp_number) - 1):
            ind1 = random.sample(range(temp_number[i], temp_number[i + 1]), int(math.ceil(batch_size / self.km_num)))
            ind = np.hstack((ind1, ind))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sampleall(self):
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device)
        )

    def Choose_sample(self, result):
        index = []
        for i in range(self.km_num):
            index0 = np.where(result == i)
            index0 = np.array(index0)
            index0 = index0.tolist()
            index0 = index0[0]
            index0 = np.array(index0)
            index.append(index0)

        return index

    def sample_ind(self, ind):
        sample = []
        for i in range(len(ind)):
            temp_sample = [self.state[ind[i]], self.action[ind[i]], self.next_state[ind[i]], self.reward[ind[i]],
                           self.not_done[ind[i]]]
            sample.append(temp_sample)
        return sample


class ReplayBuffer2(ReplayBuffer1):
    pass


class ReplayBuffer3(ReplayBuffer1):
    pass


class ReplayBuffer4(ReplayBuffer1):
    pass


class ReplayBuffer5(ReplayBuffer1):
    pass


class ReplayBuffer6(ReplayBuffer1):
    pass


class ReplayBuffer7(ReplayBuffer1):
    pass


class ReplayBuffer8(ReplayBuffer1):
    pass


class ReplayBuffer9(ReplayBuffer1):
    pass


class ReplayBuffer10(ReplayBuffer1):
    pass
