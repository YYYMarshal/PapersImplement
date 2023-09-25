```
IQ-Learn: Inverse soft-Q Learning for Imitation
```

# 链接

arXiV：https://arxiv.org/abs/2106.12142

GitHub：https://github.com/Div99/IQ-Learn

PapersWithCode：https://paperswithcode.com/paper/iq-learn-inverse-soft-q-learning-for

# 环境配置

## 简明版环境配置

```python
conda create -n IQLearn python=3.8
conda activate IQLearn

pip install -r S:\YYYXUEBING\Project\PyCharm\Papers\IQ-Learn-main\iq_learn\requirements.txt
# 先把上面的执行成功后，再安装下面的 torchvision==0.8.2，因为直接 pip install torchvision==0.8.2，会显示无法下载，所以我就先下载下来了 .whl 文件，然后再本地安装了。
# torch 相关的 .whl 文件下载地址：https://download.pytorch.org/whl/torch_stable.html
pip install S:\PyTorch_whl\torchvision-0.8.2+cpu-cp38-cp38-win_amd64.whl
```

其中，requirements.txt 的内容如下：

```python
protobuf==3.20.1
numpy==1.23.4

gym[box2d]==0.17.1
hydra-core==1.0.6
stable_baselines3==1.0
tensorboard==2.4.0
tensorboardX==2.1
torch==1.7.1
# torchvision==0.8.2
tornado==5.1.1
tqdm==4.42.1
# wandb
opencv-python==4.5.1.48
atari-py==0.2.6
gym_minigrid==1.0.2
# mujoco_py==2.0.2
mujoco_py==2.0.2.8

termcolor==2.3.0
```

## 第一次（未成功）

这个方案并未成功，下面的方案成功了。

```cmd
conda create -n IQLearn python=3.8
conda activate IQLearn

# 无效
pip install numpy
conda install nb_conda

# 下面这个不用安装，后面安装 SB3 的时候，会下载 torch 2.0.1
pip install S:\PyTorch_whl\torch-1.12.1+cu113-cp38-cp38-win_amd64.whl
# https://zhuanlan.zhihu.com/p/650565648
pip3 install hydra
# ERROR: Could not build wheels for hydra, which is required to install pyproject.toml-based projects
# https://zhuanlan.zhihu.com/p/588011598
pip install wandb
pip install tensorboardX
pip install scipy
pip install matplotlib
pip install seaborn

pip install gym
pip install stable_baselines3
# dmc2gym
# baselines_zoo

# https://zhuanlan.zhihu.com/p/613684088
pip install mujoco_py==2.0.2.8
```

## 环境的移除

```python
conda deactivate
conda remove -n IQLearn --all
```

## 第二次（成功）

下面这个方案成功了。

```python
conda create -n IQLearn python=3.8
conda activate IQLearn

pip install -r S:\YYYXUEBING\Project\PyCharm\Papers\IQ-Learn-main\iq_learn\requirements.txt
    
"""
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
"""
# https://blog.csdn.net/suiyingy/article/details/125218783
# 之前：4.24.3
pip install protobuf==3.20.1
pip install S:\PyTorch_whl\torchvision-0.8.2+cpu-cp38-cp38-win_amd64.whl

"""
AttributeError: module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe. 
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
"""
# https://www.jianshu.com/p/66ce2eeb50d3
# 1.24.4
pip install numpy==1.23.4

pip install termcolor

"""
  File "S:\Users\YYYXB\anaconda3\envs\IQLearn\lib\site-packages\wandb\sdk\wandb_login.py", line 228, in prompt_api_key
    raise UsageError("api_key not configured (no-tty). call " + directive)
wandb.errors.UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key])

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
"""
# https://blog.csdn.net/qq_43732303/article/details/131046305
pip install wandb
wandb login 66c1e6f9a168d3929b825cdcde46876b17af3307
# https://www.iotword.com/4714.html
os.environ["WANDB_API_KEY"] = '66c1e6f9a168d3929b825cdcde46876b17af3307'  # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"  # 离线  （此行代码不用修改）

"""
注册 wandb 账号后，使用上面的命令进行登录后，发现又出现了新的问题：
wandb: ERROR Abnormal program exit

从下面的帖子看到一个评论，了解到可能是因为网络的原因：
https://zhuanlan.zhihu.com/p/493093033
亓逸
[流泪]依赖于外网web还是很痛苦的（区别于tensorboard）

又翻出来了之前打开的博客，直接看向评论，评论是让卸载 wandb：
https://blog.csdn.net/qq_44824148/article/details/126573509
由此受到启发，我直接在 train_rl.py 中注释掉了 wandb 相关的代码。
"""

# python S:\YYYXUEBING\Project\PyCharm\Papers\IQ-Learn-main\iq_learn\train_iq.py agent=softq method=iq env=cartpole expert.demos=1 expert.subsample_freq=20 agent.init_temp=0.001 method.chi=True method.loss=value_expert
```

