```
Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies
RLC 2024
```

# 链接

PapersWithCode: https://paperswithcode.com/paper/bidirectional-reachable-hierarchical

GitHub: https://github.com/Roythuly/BrHPO/tree/main

# 环境配置

3060Ti G6X

WSL2 - Ubuntu 18

```cmd
conda create -n BrHPO python=3.8
conda activate BrHPO

pip install pip==24.0
pip install gym==0.18.3

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install -U 'mujoco-py<2.2,>=2.1'
```

PyCharm：运行——编辑配置——编辑配置模板——Python——环境变量——用户环境变量—— +（添加）

```cmd
# 名称：
LD_LIBRARY_PATH
# 值：
# yyy是我的用户id，在配置下面的这个值的时候需要做相应的修改
$LD_LIBRARY_PATH:/home/yyy/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

```cmd
pip install "cython<3"
pip install patchelf
```

```
pip install tensorboard
pip install six
pip install ipdb
```

