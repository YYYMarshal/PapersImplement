PyTorch implementation of TD3_CER. If you use our code or data please cite "Clustering Experience Replay for the Effective Exploitation in Reinforcement Learning".

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.0](https://github.com/pytorch/pytorch) and Python 3.7. 

# Results

Code is no longer exactly representative of the code used in the paper. Minor adjustments to hyperparameters, etc, to improve performance. Learning curves are still the original results found in the paper.

Each learning curve is formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). Evaluations are performed every 5000-time steps, over a total of 1 million time steps. 

Numerical results can be found in the paper.