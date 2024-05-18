import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, maxsize=1e6, batch_size=100):
        self.storage = [[] for _ in range(6)]
        self.maxsize = maxsize
        self.next_idx = 0
        self.batch_size = batch_size

    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, g, u, r, d = [], [], [], [], [], []

        for i in ind: 
            X, Y, G, U, R, D = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        
        return np.array(x), np.array(y), np.array(g), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)






