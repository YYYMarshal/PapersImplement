import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3CER
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing


# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def no_noise(policy, env_name, seed, replay_buffer):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    episode_timesteps = 0

    avg_reward = 0.
    state, done = eval_env.reset(), False
    while not done:
        episode_timesteps += 1
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        avg_reward += reward

    print(avg_reward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3CER")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=384, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--cluster_num", default=1e5, type=int)
    parser.add_argument("--km_num", default=3, type=int)

    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3CER.TD3(**kwargs)

    buffer = utils.ReplayBuffer(state_dim, action_dim, args.km_num)
    buffer1 = utils.ReplayBuffer1(state_dim, action_dim, args.km_num)
    buffer2 = utils.ReplayBuffer2(state_dim, action_dim, args.km_num)
    buffer3 = utils.ReplayBuffer3(state_dim, action_dim, args.km_num)
    buffer4 = utils.ReplayBuffer4(state_dim, action_dim, args.km_num)
    buffer5 = utils.ReplayBuffer5(state_dim, action_dim, args.km_num)
    buffer6 = utils.ReplayBuffer6(state_dim, action_dim, args.km_num)
    buffer7 = utils.ReplayBuffer7(state_dim, action_dim, args.km_num)
    buffer8 = utils.ReplayBuffer8(state_dim, action_dim, args.km_num)
    buffer9 = utils.ReplayBuffer9(state_dim, action_dim, args.km_num)
    buffer10 = utils.ReplayBuffer10(state_dim, action_dim, args.km_num)

    BufferList = [buffer, buffer1, buffer2, buffer3, buffer4, buffer5, buffer6, buffer7, buffer7, buffer8, buffer9,
                  buffer10]

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env_name, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    kinds_number = []

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        k = int(t // args.cluster_num)
        k1 = int((t + 1e3) // args.cluster_num)
        if k == k1:
            BufferList[k].add(state, action, next_state, reward, done_bool)
        else:
            BufferList[k].add(state, action, next_state, reward, done_bool)
            BufferList[k1].add(state, action, next_state, reward, done_bool)

        args.batch_size = int(256 * (k + 4) / 4)

        args.batch_size = min(args.batch_size, 768)

        state = next_state
        episode_reward += reward

        if t > 0 and (t + 1) % args.cluster_num == 0:

            temp_buffer = BufferList[k].sampleall()
            temp_buffer1 = temp_buffer[0]
            temp_buffer2 = temp_buffer[2]
            temp_buffer = torch.cat([temp_buffer1, temp_buffer2], 1)
            temp_buffer = temp_buffer.cpu().numpy()

            T = preprocessing.MinMaxScaler().fit(temp_buffer)
            temp_buffer = T.transform(temp_buffer)

            kmeans = KMeans(args.km_num, n_init=20, random_state=0)
            kmeans.fit(temp_buffer)
            labels = kmeans.predict(temp_buffer)
            labels = np.array(labels)

            index = BufferList[k].Choose_sample(labels)
            kinds_i = [0]
            ind = np.zeros((0), dtype=int)

            for i in range(len(index)):
                kinds_i.append(kinds_i[i] + len(index[i]) - 1)
                ind = np.hstack((ind, index[i]))

            kinds_number.append(kinds_i)
            print(kinds_i)
            print(kinds_number)
            temp_sample = BufferList[k].sample_ind(ind)

            for i in range(len(temp_sample)):
                BufferList[k].add(temp_sample[i][0], temp_sample[i][1], temp_sample[i][2], temp_sample[i][3],
                                  1 - temp_sample[i][4])

        # Train agent after collecting sufficient data
        if t >= args.batch_size and t < args.cluster_num:
            policy.train(BufferList[0], args.batch_size, kinds_number, 0)

        if t >= args.cluster_num:
            i = int(t // args.cluster_num)
            j = int(t % (2 * i + 8))
            if j < i:
                policy.train(BufferList[j], args.batch_size, kinds_number[j], 1)
            else:
                policy.train(BufferList[i], args.batch_size, kinds_number, 0)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %.3f") % (
                t + 1, episode_num + 1, episode_timesteps, episode_reward))
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            ii = int(t // args.cluster_num)
            ii1 = int((t + 1e3) // args.cluster_num)
            if t > 2 * args.start_timesteps:
                if ii == ii1:
                    no_noise(policy, args.env_name, args.seed, BufferList[ii])
                else:
                    no_noise(policy, args.env_name, args.seed, BufferList[ii])
                    no_noise(policy, args.env_name, args.seed, BufferList[ii1])

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env_name, args.seed))
            np.save("./results/%s" % (file_name), evaluations)


if __name__ == "__main__":
    main()
