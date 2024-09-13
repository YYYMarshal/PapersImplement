from .config import Config

default_config_maze = Config({
    # global program config
    "seed": 0,
    "tag": "default",
    "start_steps": 5e3,
    "cuda": True,
    "device": 0,
    "num_steps": 3e6,
    "eval": True,
    "eval_times": 10,
    "eval_freq": 50,
    "log_path": None,
    "model_path": None,
    "debug_mode": False,
    "is_load": False,
    "model_i_epoch": None,

    # task config
    "env_name": "AntMaze",
    "temporal_horizon": 20,
    "subgoal_dim": 2,
    "l_action_dim": 8,
    "h_action_dim": 2,
    "action_max": 30,
    "max_steps": 500,
    "goal_scale": 2,
    "low_reward_bonus": 10,

    # policy config
    "high_replay_size": 100000,
    "low_replay_size": 1000000,
    "low_batch_size": 128,
    "high_batch_size": 128,
    "low_agent_train_freq": 1,
    "high_agent_train_freq": 20,
    "high_reward_scale": 1,
    "low_reward_scale": 1,
    "target_update_interval": 2,
    "automatic_entropy_tuning": False,
    "lr": 0.0003,
    "critic_lr": 0.001,
    "policy_lr": 0.0001,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "hidden_size": 256,

})

default_config_manipulation = Config({
    # global program config
    "seed": 0,
    "tag": "default",
    "start_steps": 1e3,
    "cuda": True,
    "device": 0,
    "num_steps": 5e5,
    "eval": True,
    "eval_times": 10,
    "eval_freq": 40,
    "log_path": None,
    "model_path": None,
    "debug_mode": False,
    "is_load": False,
    "model_i_epoch": None,

    # task config
    "env_name": "Reacher3D",
    "temporal_horizon": 10,
    "subgoal_dim": 3,
    "l_action_dim": 8,
    "h_action_dim": 3,
    "action_max": 30,
    "max_steps": 100,
    "goal_scale": 2,
    "low_reward_bonus": 0.1,

    # policy config
    "high_replay_size": 100000,
    "low_replay_size": 1000000,
    "low_batch_size": 128,
    "high_batch_size": 128,
    "low_agent_train_freq": 1,
    "high_agent_train_freq": 20,
    "high_reward_scale": 1,
    "low_reward_scale": 1,
    "target_update_interval": 2,
    "automatic_entropy_tuning": False,
    "lr": 0.0003,
    "critic_lr": 0.001,
    "policy_lr": 0.0001,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "hidden_size": 256,

})
