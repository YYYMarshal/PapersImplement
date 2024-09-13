import gym
import random
import numpy as np
import torch
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from envs.antenv import EnvWithGoal, GatherEnv
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.create_gather_env import create_gather_env
import shutil
from .model.sac import SAC
from .utils.replaybuffer import ReplayMemory
from .algo import Algo

import ipdb


def get_env_params(env, args):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'sub_goal': args.subgoal_dim,
              'l_action_dim': args.l_action_dim,
              'h_action_dim': args.h_action_dim,
              'action_max': args.action_max,
              'max_steps': args.max_steps}
    return params


def launch(args):
    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)
        if args.env_name in ["Reacher3D-v0", "AntMazeBottleneck-v0"]:
            test_env.set_evaluate()
    seed = args.seed

    env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    assert np.all(env.action_space.high == -env.action_space.low)
    env_params = get_env_params(env, args)
    low_reward_func = env.low_reward_func
    high_reward_func = env.high_reward_func

    # suppose subgoal_space = goal_space
    if args.env_name in ['Pusher-v0', 'Reacher3D-v0', 'AntMazeBottleneck-v0']:
        high_agent = SAC(env.observation_space['observation'].shape[0], env.goal_space.shape[0], env.goal_space, args,
                         'high')
        low_agent = SAC(env.observation_space['observation'].shape[0], env.goal_space.shape[0], env.action_space, args,
                        'low')
    else:
        high_agent = SAC(env.observation_space.shape[0], env.goal_space.shape[0], env.goal_space, args, 'high')
        low_agent = SAC(env.observation_space.shape[0], env.goal_space.shape[0], env.action_space, args, 'low')
    if args.is_load:
        high_agent.load_checkpoint(args.model_path, args.model_i_epoch, 'high')
        low_agent.load_checkpoint(args.model_path, args.model_i_epoch, 'low')
    else:
        log_path = './logs/{}/{}/{}_{}'.format(args.env_name, args.tag,
                                               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                               args.seed)
        os.makedirs(log_path)
        with open(os.path.join(log_path, "config.log"), 'w') as f:
            f.write(str(args))

        shutil.copytree('.', log_path + '/code', ignore=shutil.ignore_patterns('logs', 'SAC_logs'))
        args.log_path = log_path
        args.model_path = log_path + '/model'
        os.makedirs(args.model_path)
    high_replay = ReplayMemory(args.high_replay_size, args.seed)
    low_replay = ReplayMemory(args.low_replay_size, args.seed)

    algo = Algo(
        env=env, env_params=env_params, args=args,
        test_env=test_env,
        low_agent=low_agent, high_agent=high_agent, low_replay=low_replay, high_replay=high_replay,
        low_reward_func=low_reward_func, high_reward_func=high_reward_func
    )
    return algo
