# 链接

ScienceDirect：https://www.sciencedirect.com/science/article/pii/S0950705121003919

GitHub：https://github.com/grcai/Anchor-An-algorithm-for-RL

# 环境配置

```cmd
conda create -n AHRL python=3.8
conda activate AHRL
# 最好使用这个版本
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy
# 最好使用这个版本
pip install gym==0.19.0
pip install six==1.16.0
# 2024-5-16 02:28:12
pip install tyro
```

# 原文的README

## Introduction

This is a code implementation of the algorithm Anchor-based HRL (AHRL) presented in the following paper. 

Anchor: The achieved goal to replace the subgoal for hierarchical reinforcement learning (Ruijia Li, Zhiling Cai, Tianyi Huang, William Zhu). 

AHRL is tested by [MuJoCo](http://www.mujoco.org/) and [OpenAI gym](https://github.com/openai/gym). Networks are trained using [PyTorch](https://github.com/pytorch/pytorch). 

## Usage

To train a policy on the Point Maze task by running:

`python main.py --env PointMaze`

To see the performance of a policy trained on the Point Maze task by running:

`python test.py --env PointMaze`

## Credits

We would like to thank:
* [TD3](https://github.com/sfujim/TD3) 
  Our codebase is based on theirs.
* [DSC](https://github.com/deep-skill-chaining/deep-skill-chaining) 
  Our environment Point Maze is based on their code.
* [HIRO](https://github.com/tensorflow/models/tree/master/research/efficient-hrl) 
  Our environment Ant Push is based on their code.
* [RLLAB](https://github.com/rllab/rllab)
  Our environment Double Inverted Pendulumis is based on their code.




