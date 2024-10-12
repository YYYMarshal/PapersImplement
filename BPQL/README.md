# Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback
This repository contains the PyTorch implementation of **Belief Projection-Based Q-Learning (BPQL)** introduced in the paper:

**Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback** by Jangwon Kim et al., presented at Advances in Neural Information Processing Systems (NeurIPS), 2023.


## Paper Link
>See the paper here: https://proceedings.neurips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html

## Test Environment
```
python == 3.8.10
gym == 0.26.2
mujoco_py == 2.1.2.14
pytorch == 2.1.0
numpy == 1.24.3
```

## How to Run?
### Run the script file 
```
>chmod +x run.sh
>./run.sh
```

### or run main.py with arguments
```
# 2024.10.11 晚上8:12才配置好环境，一开始是在终端上运行了几次，后来才到PyCharm上运行的，应该8:20吧，算正式开始。
# 2024.10.12 早上10:30就已经跑完了。所以大概跑一次十三四个小时吧。
python main.py --env-name HalfCheetah-v3 --random-seed 2023 --obs-delayed-steps 5 --act-delayed-steps 4 --max-step 1000000
```
---

## Citation Example
```
@inproceedings{kim2023cocel,
   author = {Kim, Jangwon and Kim, Hangyeol and Kang, Jiwook and Baek, Jongchan and Han, Soohee},
   booktitle = {Advances in Neural Information Processing Systems},
   pages = {678--696},
   title = {Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback},
   volume = {36},
   year = {2023}
}
```
