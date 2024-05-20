import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter


def fun1():
    env = gym.make("LunarLander-v2", render_mode="human")
    env.reset()

    for step in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()
        print(step, end=" ")
        if step % 100 == 0:
            print("\n---------------")

    env.close()


def fun2():
    x = torch.Tensor(2, 3)
    print(x)
    y = x.uniform_(5, 10)
    print(y)


def fun3():
    # print(torch.tensor(2.0)) #tensor(2.)

    print(torch.Tensor(2).shape)  # torch.Size([2])
    print(torch.tensor(2).shape)  # torch.Size([])
    print(torch.empty(2))  # tensor([0., 0.])
    print(torch.empty(0).shape)  # torch.Size([0])s
    # print(torch.tensor(2, requires_grad=True).shape)  # torch.Size([])
    # print(torch.tensor(2, requires_grad=True).shape) # 崩了，RuntimeError: Only Tensors of floating point and complex dtype can require gradients
    print(torch.tensor(2.0, requires_grad=True).shape)  # torch.Size([])
    print(torch.tensor([2.0], requires_grad=True).shape)  # torch.Size([])
    # print(torch.Tensor([2.0], requires_grad=True).shape) # 也会崩，Tensor是类，得用小写的tensor函数才行


def fun4():
    # m = torch.distributions.categorical.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    m = torch.distributions.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    x = m.sample()  # equal probability of 0, 1, 2, 3
    print(x)


def fun5():
    x = torch.Tensor([2, 7, 3])  # 20次，70次，30次
    m = torch.distributions.Categorical(x)
    re = [0, 0, 0]  # 三个数抽到的个数
    for i in range(100):
        re[m.sample()] += 1  # sample就是抽一次
    print(re)


def fun6():
    x = torch.tensor([7, 8, 2, 6])
    print(x.argmax())
    print(x.argmax().item())


def fun7():
    writer = SummaryWriter("test_logs")
    for i in range(101):
        writer.add_scalar("y = x * x", i * i, i)
    for i in range(101):
        writer.add_scalar("y = x + 10", i + 10, i)

    writer.close()


fun7()
