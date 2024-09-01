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

==PyTorch 1.12.1, CUDA 11.3==

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

==MuJoCo 相关==

```cmd
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

运行下面的测试代码：

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

```cmd
# ERROR！！！
# Cython.Compiler.Errors.CompileError: xxx\mujoco210\lib\site-packages\mujoco_py\cymj.pyx
pip install cython==0.29.21
```

```cmd
# ERROR！！！
# distutils.errors.CompileError: command 'gcc' failed with exit status 1

# https://zhuanlan.zhihu.com/p/547442285
sudo apt install libosmesa6-dev
pip install patchelf

# http://t.csdnimg.cn/CCfTX
sudo apt install libgl1-mesa-glx --reinstall

# Solution！！！
# http://t.csdnimg.cn/9NCTu
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
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

