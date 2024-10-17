# Introduction

This is a code implementation of the algorithm Anchor-based HRL (AHRL) presented in the following paper. 

Anchor: The achieved goal to replace the subgoal for hierarchical reinforcement learning (Ruijia Li, Zhiling Cai, Tianyi Huang, William Zhu). 

AHRL is tested by [MuJoCo](http://www.mujoco.org/) and [OpenAI gym](https://github.com/openai/gym). Networks are trained using [PyTorch](https://github.com/pytorch/pytorch). 

# Usage

To train a policy on the Point Maze task by running:

`python main.py --env PointMaze`

To see the performance of a policy trained on the Point Maze task by running:

`python test.py --env PointMaze`

# Credits

We would like to thank:

* [TD3](https://github.com/sfujim/TD3) 
  Our codebase is based on theirs.
* [DSC](https://github.com/deep-skill-chaining/deep-skill-chaining) 
  Our environment Point Maze is based on their code.
* [HIRO](https://github.com/tensorflow/models/tree/master/research/efficient-hrl) 
  Our environment Ant Push is based on their code.
* [RLLAB](https://github.com/rllab/rllab)
  Our environment Double Inverted Pendulumis is based on their code.



