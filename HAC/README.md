```
Learning Multi-Level Hierarchies with Hindsight
```

# 链接

arXiv：https://arxiv.org/abs/1712.00948

GitHub（official - TensorFlow）：https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-

GitHub（PyTorch）：https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch

PapersWithCode：https://paperswithcode.com/paper/learning-multi-level-hierarchies-with

# 环境配置

## Linux

> GPU: 3060Ti G6X
>
> Ubuntu 18

```python
conda create -n HAC python=3.8
conda activate HAC
```

```cmd
# 安装 gym 0.18.3 前的准备：
# http://t.csdnimg.cn/CRHK7
# pip install --upgrade pip setuptools==57.5.0
# pip install --upgrade pip wheel==0.37.0
pip install pip==0.24.0
pip install gym==0.18.3
```

```cmd
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

```cmd
# 路径需替换成自己文件所在的路径
pip install -r xxx\requirements.txt
```

其中，requirements.txt 的内容如下：

```python
pyglet
six
```



## Windows

```python
conda create -n HAC python=3.6
conda activate HAC

# 路径需替换成自己文件所在的路径
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

