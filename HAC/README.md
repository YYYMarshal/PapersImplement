```
Learning Multi-Level Hierarchies with Hindsight
```

# 链接

arXiv：https://arxiv.org/abs/1712.00948

GitHub（official - TensorFlow）：https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-

GitHub（PyTorch）：https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch

PapersWithCode：https://paperswithcode.com/paper/learning-multi-level-hierarchies-with

# 环境配置

```python
conda create -n HAC python=3.6
conda activate HAC

pip install -r S:\YYYXUEBING\Project\PyCharm\Papers\Hierarchical-Actor-Critic-HAC-PyTorch-master\requirements.txt
```

其中，requirements.txt 的内容如下：

```python
# 代码中有 env.render() 相关，所以用了 gym 0.18.3
# gym
gym==0.18.3
# 因为 python 3.6，只能最高用 torch 1.10.2
# torch
torch==1.10.2
pyglet
six
```

