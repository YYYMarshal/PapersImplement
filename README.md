# 简介

2023-9-25 20:48:13

* 每篇文章的论文 arXiv 链接、GitHub 链接和 PapersWithCode 链接都放在了各自文件夹的 README.md 文件中。
* 各个文章对应的代码的环境配置，也在各自文件夹的 README.md 文件中。

2024-5-18 22:15:55

* 打开这一个总的、大的项目，会出现一些代码的引用问题，以及在某个子项目中查找某个变量或者函数的调用时，可能导向其他子项目，所以目前的解决方案的建议是：分别打开每个子项目，以子项目为单独的一个PyCharm项目来运行。

# 论文与文件夹的对应表

| 论文                                                         | 文件夹   | 发表时间            | 期刊/会议/出版社                           | 备注                                                         |
| ------------------------------------------------------------ | -------- | ------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Learning Multi-Level Hierarchies with Hindsight              | HAC      | 2017(v1) - 2019(v5) | arXiv                                      | PyTorch 版的代码是非官方的，代码中只给了自定义的 MountainCarContinuous 环境，issues 中给了自定义的 Pendulum 环境，原文中的几个环境如何修改并未提及，所以我没法测试原文中的几个环境。 |
| The Option-Critic Architecture                               | ==OC==   | 2017                | AAAI                                       |                                                              |
| Addressing Function Approximation Error in Actor-Critic Methods | ==TD3==  | 2018                | ICML                                       |                                                              |
| Data-Efficient Hierarchical Reinforcement Learning           | ==HIRO== | 2018                | NeurIPS                                    |                                                              |
| IQ-Learn: Inverse soft-Q Learning for Imitation              | IQLearn  | 2021                | NeurIPS(CCF A)                             |                                                              |
| Anchor: The achieved goal to replace the subgoal for hierarchical reinforcement learning | AHRL     | 2021.8              | Knowledge-Based Systems(中科院1区，JCR Q1) | 这里的环境都是他魔改过的，不知道如何扩展到其他的环境，第三个测试任务已经趋于完美了。 |
| Clustering experience replay for the effective exploitation in reinforcement learning | TD3CER   | 2022.11             | Pattern Recognition(中科院1区)             |                                                              |
| Robust Policy Optimization in Deep Reinforcement Learning    | RPO      | 2022.12             | arXiv                                      | 从这里的源代码 CleanRL 中知道了 `args = tyro.cli(Args)`      |
| Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback | BPQL     | 2023                | NeurIPS                                    |                                                              |

## 排除

| 论文                                                         | 文件夹 | 发表时间 | 期刊/会议/出版社 | 备注                                                         |
| ------------------------------------------------------------ | ------ | -------- | ---------------- | ------------------------------------------------------------ |
| Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies | BrHPO  | 2024     | RLC              | 有源代码都跑不出文章中的效果，AntMaze、AntPush、AntFall在3e6步数下的成功率直接全部为0，而且 main.py 只开放了五个环境的测试，文章中却有6个任务的测试数据，多的那一个是 AntBigMaze。 |