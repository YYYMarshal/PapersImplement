<h1>BrHPO</span></h1>

Official implementation of

`Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies` by

Yu Luo, Fuchun Sun, Tianying Ji and Xianyuan Zhan

### Getting started

We provide examples on how to train and evaluate **BrHPO** agent.

### Training & Visualization

To visualize the performance of BrHPO, 

```python
python render_result.py  logs/AntMaze/visualization 6000  # for AntMaze

python render_result.py  logs/AntPush/visualization 6000  # for AntPush

python render_result.py  logs/Reacher3D/visualization 4800  # for Reacher3D
```

To train BrHPO,

```python
python main.py --env_name AntMaze
```

### Citation

If you find our work useful, please consider citing our paper as follows:

````
@inproceedings{Luo2024BrHPO,
  title={Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies}, 
  author={Yu Luo and Fuchun Sun and Tianjing Ji and Xianyuan Zhan},
  booktitle={Reinforcement Learning Conference},
  year={2024}
}
````

