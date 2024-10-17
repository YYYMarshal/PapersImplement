# 简介

2023-9-25 20:48:13

* 每篇文章的论文 arXiv 链接、GitHub 链接和 PapersWithCode 链接都放在了各自文件夹的 README.md 文件中。
* 各个文章对应的代码的环境配置，也在各自文件夹的 README.md 文件中。

2024-5-18 22:15:55

* 打开这一个总的、大的项目，会出现一些代码的引用问题，以及在某个子项目中查找某个变量或者函数的调用时，可能导向其他子项目，所以目前的解决方案的建议是：分别打开每个子项目，以子项目为单独的一个PyCharm项目来运行。

# 论文与文件夹的对应表

## Classic

| 论文                                                         | 文件夹  | 发表时间 | 期刊/会议/出版社 | 备注 |
| :----------------------------------------------------------- | ------- | -------- | ---------------- | ---- |
| Addressing Function Approximation Error in Actor-Critic Methods | ==TD3== | 2018     | ICML             |      |

## HRL

| 论文                                                         | 文件夹   | 发表时间            | 期刊/会议/出版社                           | 备注                                                         |
| ------------------------------------------------------------ | -------- | ------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Learning Multi-Level Hierarchies with Hindsight              | HAC      | 2017(v1) - 2019(v5) | arXiv                                      | PyTorch 版的代码是非官方的，代码中只给了自定义的 MountainCarContinuous 环境，issues 中给了自定义的 Pendulum 环境，原文中的几个环境如何修改并未提及，所以我没法测试原文中的几个环境。 |
| The Option-Critic Architecture                               | ==OC==   | 2017                | AAAI                                       |                                                              |
| Data-Efficient Hierarchical Reinforcement Learning           | ==HIRO== | 2018                | NeurIPS                                    |                                                              |
| Anchor: The achieved goal to replace the subgoal for hierarchical reinforcement learning | AHRL     | 2021.8              | Knowledge-Based Systems(中科院1区，JCR Q1) | 这里的环境都是他魔改过的，不知道如何扩展到其他的环境，第三个测试任务已经趋于完美了。 |

## 待复现

| 论文                                                         | 文件夹  | 发表时间 | 期刊/会议/出版社               | 备注                                                    |
| ------------------------------------------------------------ | ------- | -------- | ------------------------------ | ------------------------------------------------------- |
| IQ-Learn: Inverse soft-Q Learning for Imitation              | IQLearn | 2021     | NeurIPS(CCF A)                 |                                                         |
| Clustering experience replay for the effective exploitation in reinforcement learning | TD3CER  | 2022.11  | Pattern Recognition(中科院1区) |                                                         |
| Robust Policy Optimization in Deep Reinforcement Learning    | RPO     | 2022.12  | arXiv                          | 从这里的源代码 CleanRL 中知道了 `args = tyro.cli(Args)` |
| Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback | BPQL    | 2023     | NeurIPS                        |                                                         |



## 排除

| 论文                                                         | 文件夹 | 发表时间 | 期刊/会议/出版社 | 备注                                                         |
| ------------------------------------------------------------ | ------ | -------- | ---------------- | ------------------------------------------------------------ |
| Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies | BrHPO  | 2024     | RLC              | 有源代码都跑不出文章中的效果，AntMaze、AntPush、AntFall在3e6步数下的成功率直接全部为0，而且 main.py 只开放了五个环境的测试，文章中却有6个任务的测试数据，多的那一个是 AntBigMaze。 |

# 环境搭建

```
PapersImplement
```

## 环境创建

```cmd
conda create -n PapersImplement python=3.8
conda activate PapersImplement
```

## 环境配置

首次环境配置才需要执行下面的命令，如果以后如果有 别的项目中的一些包 必须跟 下面中的包 有不一样的版本，那么直接用环境复制即可。

### MuJoCo

```cmd
pip install --upgrade pip setuptools==57.5.0
pip install --upgrade pip wheel==0.37.0

pip install pip==24.0
pip install gym==0.18.3

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 2.1.2.14
pip3 install -U 'mujoco-py<2.2,>=2.1'

pip install cython==0.29.37

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

pip install patchelf==0.17.2.1
```

### 其他

```cmd

```



## PyCharm配置

下面这个在Windows上运行PyCharm来调用WSL2 Ubuntu中的环境时，才需要在PyCharm中配置。

PyCharm：运行——编辑配置——编辑配置模板——Python——环境变量——用户环境变量—— +（添加）

```cmd
# 名称：
LD_LIBRARY_PATH
# 值：
# yyy是我的用户id，在配置下面的这个值的时候需要做相应的修改
$LD_LIBRARY_PATH:/home/yyy/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

## MuJoCo 测试

运行测试运行下面的测试代码：

```python
import mujoco_py
import os

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
sim.step()
print(sim.data.qpos)
```

输出：

```
[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
[-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
  1.80085466e-21  4.45933954e-05 -2.70143345e-20  1.30126513e-19
 -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
 -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
 -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
 -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
 -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
```



## 环境移除

```cmd
conda deactivate
conda remove -n PapersImplement --all
```

## 环境复制

```cmd
conda create -n XXX --clone PapersImplement
```

