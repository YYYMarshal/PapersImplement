import torch


def evaluate_policy(env, controller_policy, maxEpisodeT):
    Itr = 10
    env.evaluate = True
    achieved = 0

    with torch.no_grad():
        Reward = 0.

        for eval_ep in range(Itr):
            obs = env.reset()
            state = obs['state']
            done = False
            episodeT = 0
            while not done and episodeT < maxEpisodeT:
                # env.render()

                episodeT += 1
                action = controller_policy.select_action(state)
                next_obs, reward, done, _ = env.step(action)
                next_state = next_obs['state']
                Reward += reward
                state = next_state
                if done: achieved = achieved+1

    return achieved, round(Reward / Itr)