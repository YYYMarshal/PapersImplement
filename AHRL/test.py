# Written by Ruijia Li (ruijia2017@163.com), UESTC, 2020-12-1.
import argparse
from Algorithms import ahrl
from Environments.AntPush.maze_env import AntPushEnv
from Environments.PointMaze.maze_env import PointMazeEnv
from Environments.DoubleInvertedPendulum.DoubleInvertedPendulum import DoubleInvertedPendulumEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PointMaze")             # Environment name: DoubleInvertedPendulum, PointMaze or AntPush
    parser.add_argument("--max_test_steps", default=5e2, type=int)     # Max test steps to run environment
    args = parser.parse_args()

    if args.env_name == "AntPush":
        env = AntPushEnv()
    elif args.env_name == "PointMaze":
        env = PointMazeEnv()
    else: 
        env = DoubleInvertedPendulumEnv()
        
    state = env.reset()
    obs = env.reset()
    state = obs['state']
    
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    maxEpisode_step = env.max_step()

    file_name = "%s%s" % (args.env_name, str(50))
    policy = ahrl.AHRL(state_dim=state_dim, action_dim=action_dim, scale=max_action, args=args)
    policy.load(file_name, "./Results/Example")

    Reward = 0
    total_step = 0
    env_done = False
    episode_step = 0


    "Test"
    while total_step < args.max_test_steps:
        #env.render()

        if env_done or episode_step == maxEpisode_step:
            print("   Reward={}".format(round(Reward)))
            obs = env.reset()
            state = obs['state']
            anchor = obs['achieved_goal']
            Reward = 0
            achieved = 0
            episode_step = 0

        episode_step += 1
        total_step += 1

        action = policy.select_action(state)
        
        next_obs, reward, env_done, _ = env.step(action)
        next_state = next_obs['state']
        achieved_goal = next_obs['achieved_goal']

        Reward = Reward + reward
        state = next_state
