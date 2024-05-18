import os

# Only for Windows System
os.add_dll_directory(os.environ["USERPROFILE"] + "/.mujoco/mjpro150/bin")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time, torch, argparse
import numpy as np
from Algorithms import ahrl, buffer, evaluate, normalization
from Environments.AntPush.maze_env import AntPushEnv
from Environments.PointMaze.maze_env import PointMazeEnv
from Environments.DoubleInvertedPendulum.DoubleInvertedPendulum import DoubleInvertedPendulumEnv
import matplotlib.pyplot as plt


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",
                        default="PointMaze")  # Environment name : DoubleInvertedPendulum, PointMaze or AntPush
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_steps", default=5e5, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=1.0)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--max_buffer_size", default=2e5, type=int)  # Max size for replay buffer
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.05)  # Target network update rate
    parser.add_argument("--eta", default=0.001)  # Regularizer for intrinsic reward
    parser.add_argument("--alpha", default=0.001)  # Lower bound of the normalized function
    parser.add_argument("--policy_noise", default=0.1)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--anchor_freq", default=10, type=int)  # Frequency of changing the anchor
    args = parser.parse_args()

    file_name = "%s%s" % (args.env_name, str(args.seed))

    if args.env_name == "AntPush":
        env = AntPushEnv()
    elif args.env_name == "PointMaze":
        env = PointMazeEnv()
    else:
        env = DoubleInvertedPendulumEnv()

    "Environment"
    maxEpisode_step = env.max_step()
    obs = env.reset()
    state = obs['state']
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    "Seed"
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    "Initialize"
    replay_buffer = buffer.ReplayBuffer(maxsize=args.max_buffer_size)
    policy = ahrl.AHRL(state_dim=state_dim, action_dim=action_dim, scale=max_action, args=args)

    col = int(np.ceil(maxEpisode_step / args.anchor_freq))
    anchor_weight = np.zeros((50000, col))
    step = 0
    episode_step = 0
    anchor_step = 0
    anchor_reward = 0
    episode_num = 0
    anchor_num = 0
    # episode_reward = 0
    done = True
    # env_done = False
    anchor = obs["achieved_goal"]
    evaluation_reward = []
    evaluation_achieve = []

    "AHRL"
    while step < args.max_steps:
        if done:
            if step != 0:
                "Normalize"
                tempFall = -anchor_weight[0:episode_num, 0:col]
                tempAbs = np.abs(tempFall)

                MaxAbs = np.max(tempAbs)
                noZeroMinAbs = np.min(np.where(tempAbs == 0, tempAbs.max(), tempAbs))
                noZeroMax = np.max(np.where(tempFall == 0, tempAbs.min(), tempFall))
                noZeroMin = np.min(np.where(tempFall == 0, tempAbs.max(), tempFall))

                if noZeroMax * noZeroMin > 0 and noZeroMinAbs / (MaxAbs + 1e-8) > args.alpha:
                    minNormal = noZeroMinAbs / (MaxAbs + 1e-8)
                else:
                    minNormal = args.alpha

                weight = normalization.Normalization(tempFall, noZeroMin, noZeroMax, minNormal, 1)

                "Train"
                policy.train(weight, replay_buffer, episode_step, args)

            "Reset"
            obs = env.reset()
            state = obs['state']
            anchor = obs['achieved_goal']
            done = 0
            episode_step = 0
            anchor_step = 0
            anchor_num = 1
            episode_num = episode_num + 1

        "Action"
        action = policy.select_action(state)
        action = (action + np.random.normal(0, args.expl_noise, size=action_dim)).clip(-max_action, max_action)
        next_obs, reward, env_done, _ = env.step(action)
        next_state = next_obs['state']
        achieved_goal = next_obs['achieved_goal']
        episode_step += 1

        "Intrinsic Reward"
        anchor_reward = anchor_reward + reward
        intrinsic_reward = -1 / (np.linalg.norm(achieved_goal - anchor) + args.eta)

        "Buffer"
        if env_done or episode_step == maxEpisode_step: done = 1
        transition = [state, next_state, [episode_num - 1, anchor_num - 1], action, intrinsic_reward, float(done)]
        replay_buffer.add(transition)

        "Anchor"
        anchor_step += 1
        if step != 0 and anchor_step % args.anchor_freq == 0:
            anchor_weight[episode_num - 1, anchor_num - 1] = anchor_reward

            anchor = next_obs['achieved_goal']
            anchor_num = anchor_num + 1
            anchor_step = 0
            anchor_reward = 0

        if env_done and anchor_step % args.anchor_freq != 0:
            anchor_weight[episode_num - 1, anchor_num - 1] = anchor_reward
            anchor_num = anchor_num + 1
            anchor_reward = 0

        "Update"
        state = next_state
        step += 1

        """ Evaluate """
        achieved = 0
        evaluate_reward = 0
        if step % args.eval_freq == 0:
            achieved, evaluate_reward = evaluate.evaluate_policy(env, policy, maxEpisode_step)
            evaluation_achieve.append(achieved)
            evaluation_reward.append(evaluate_reward)
        if step % 10000 == 0:
            print("Achieved: %d    Reward: %d" % (achieved, evaluate_reward))

        "Save"
        if step % 100000 == 0:
            np.save("./Results/%s" % file_name + "Reward", evaluation_reward)
            np.save("./Results/%s" % file_name + "Achieved", evaluation_achieve)
            policy.save("%s" % file_name, directory="./Results")

        "Time"
        if step % 100000 == 0:
            T = (time.time() - start_time)
            Time = time.strftime("%H:%M", time.localtime())
            print('%d: %d:%d:%d   %s ' % (step, int(T / 3600), int(T / 60) % 60, int(T % 60), Time))
    show_result(args.env_name, evaluation_achieve, evaluation_reward)


def show_result(env_name: str, evaluation_achieve, evaluation_reward):
    # PointMaze
    # 平均值：9.02, -190.93
    # 平均值：9.16, -172.34

    # DoubleInvertedPendulum
    # 平均值：9.9, -13.99
    print(f"平均值：{np.mean(evaluation_achieve)}, {np.mean(evaluation_reward)}")
    algorithm = "AHRL"
    xlabel = "Training Steps(×5000)"

    # plt.subplot(121)
    x_list = list(range(len(evaluation_achieve)))
    title = f"{algorithm} on {env_name}"
    plt.plot(x_list, evaluation_achieve)
    plt.xlabel(xlabel)
    plt.ylabel("Success rate")
    plt.title(title)

    plt.show()

    # plt.subplot(122)
    x_list = list(range(len(evaluation_reward)))
    title = f"{algorithm} on {env_name}"
    plt.plot(x_list, evaluation_reward)
    plt.xlabel(xlabel)
    plt.ylabel("Average reward")
    plt.title(title)

    plt.show()


if __name__ == "__main__":
    main()
