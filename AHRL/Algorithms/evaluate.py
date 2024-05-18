import torch


def evaluate_policy(env, controller_policy, max_episode_step):
    # 评估的回合数
    num_episode_evaluate = 10
    env.evaluate = True
    achieved = 0

    with torch.no_grad():
        total_reward = 0.

        for episode in range(num_episode_evaluate):
            obs = env.reset()
            state = obs["state"]
            done = False
            episode_step = 0
            while not done and episode_step < max_episode_step:
                # env.render()

                episode_step += 1
                action = controller_policy.select_action(state)
                next_obs, reward, done, _ = env.step(action)
                next_state = next_obs["state"]
                total_reward += reward
                state = next_state
                if done:
                    achieved += 1

    return achieved, round(total_reward / num_episode_evaluate)
