```cmd
conda create -n RPO python=3.8
conda activate RPO
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "Gymnasium[all]"

# 然后用 mujoco150 中的 mujoco_py 替换 RPO 虚拟环境中的 mujoco_py。
pip install cython==0.29.21

# 2024-5-15 02:17:51
# 本来还提示要让用 mujoco210，然后我就想着用 150 替换一下试试呗，然后竟然就成功运行了，就这么成功了？最快速的一次。。。
```