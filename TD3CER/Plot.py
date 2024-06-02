import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot(env_name: str):
    file_name = f"TD3CER_{env_name}_0.npy"
    y_list = np.load(f"results/{file_name}")
    # x_list = list(range(len(y_list)))
    count = int(1e6 / 5e3) + 1
    x_list = np.linspace(0, 1e6, count)
    # print(x_list)
    xlabel = "x"
    ylabel = "y"
    title = f"TD3CER on {env_name}"
    plt.plot(x_list, y_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.ticklabel_format(style='plain', axis='x')

    plt.show()
    plt.close()


def main():
    # √
    env_name = "HalfCheetah-v2"
    plot(env_name)
    # √×
    env_name = "Walker2d-v3"
    plot(env_name)
    # √×
    env_name = "Ant-v3"
    plot(env_name)
    # ×
    env_name = "Hopper-v3"
    plot(env_name)
    # ×
    env_name = "Humanoid-v3"
    plot(env_name)
    # √×
    env_name = "InvertedDoublePendulum-v2"
    plot(env_name)


if __name__ == '__main__':
    main()
