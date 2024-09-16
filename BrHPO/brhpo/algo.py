from copy import copy, deepcopy
import sys
# import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .utils.replaybuffer import ReplayMemory


# import ipdb


class Algo:
    def __init__(self,
                 env, env_params, args,
                 test_env, low_agent, high_agent, low_replay, high_replay,
                 low_reward_func, high_reward_func
                 ):
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.args = args
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.low_replay = low_replay
        self.high_replay = high_replay

        self.low_reward_func = low_reward_func
        self.high_reward_func = high_reward_func
        self.total_numsteps = 0
        self.total_episode = 0
        self.low_agent_update = 0
        self.high_agent_update = 0
        self.train_flag = True
        if not self.args.is_load:
            log_path = self.args.log_path
            self.writer = SummaryWriter(log_path)

    def run_eval_render(self):
        avg_success = 0
        avg_reward = 0
        avg_final_distance = 0
        for n_test in range(self.args.eval_times):
            info = None
            observation = self.test_env.reset()
            ob = observation['observation']
            dg = observation['desired_goal']
            ag = observation['achieved_goal']
            episode_step = 0
            while episode_step <= self.args.max_steps:
                ob_high = deepcopy(ob)
                sg = self.high_agent.select_action(ob_high, dg, evaluate=True)
                sg_low = sg + ag
                for _ in range(self.args.temporal_horizon):
                    action = self.low_agent.select_action(ob, sg_low, evaluate=True)
                    new_observation, reward, _, info = self.test_env.step(action)
                    self.test_env.render()
                    ob = new_observation['observation']
                    ag = new_observation['achieved_goal']
                    avg_reward += reward
                    episode_step += 1
                    if info['is_success']:
                        break
                if info['is_success']:
                    avg_success += 1
                    break
            avg_final_distance += self.test_env.goal_distance(ag, dg)
            print("Achieved goal: (%.2f, %.2f), Goal: (%.2f, %.2f)" % (ag[0], ag[1], dg[0], dg[1]))
        avg_reward = avg_reward / self.args.eval_times
        avg_success = avg_success / self.args.eval_times
        avg_final_distance = avg_final_distance / self.args.eval_times
        if self.args.debug_mode:
            print("Achieved goal: (%.2f, %.2f), Goal: (%.2f, %.2f)" % (ag[0], ag[1], dg[0], dg[1]))
        print("Test Episodes: {}, Avg. Reward: {}, Success: {}".format(self.total_episode, round(avg_reward, 2),
                                                                       round(avg_success, 2)))
        print(' ')
        return avg_reward, avg_success

    def run_eval(self):
        avg_success = 0
        avg_reward = 0
        avg_final_distance = 0
        for n_test in range(self.args.eval_times):
            info = None
            observation = self.test_env.reset()
            ob = observation['observation']
            dg = observation['desired_goal']
            ag = observation['achieved_goal']
            episode_step = 0
            while episode_step <= self.args.max_steps:
                ob_high = deepcopy(ob)
                sg = self.high_agent.select_action(ob_high, dg, evaluate=True)
                sg_low = sg + ag
                for _ in range(self.args.temporal_horizon):
                    action = self.low_agent.select_action(ob, sg_low, evaluate=True)
                    new_observation, reward, _, info = self.test_env.step(action)
                    ob = new_observation['observation']
                    ag = new_observation['achieved_goal']
                    avg_reward += reward
                    episode_step += 1
                    if info['is_success']:
                        break
                if info['is_success']:
                    avg_success += 1
                    break
            avg_final_distance += self.test_env.goal_distance(ag, dg)
        avg_reward = avg_reward / self.args.eval_times
        avg_success = avg_success / self.args.eval_times
        avg_final_distance = avg_final_distance / self.args.eval_times
        self.writer.add_scalar('test/avg_reward', avg_reward, self.total_numsteps)
        self.writer.add_scalar('test/avg_success', avg_success, self.total_numsteps)
        self.writer.add_scalar('test/avg_final_distance', avg_final_distance, self.total_numsteps)
        if self.args.debug_mode:
            print("Achieved goal: (%.2f, %.2f), Goal: (%.2f, %.2f)" % (ag[0], ag[1], dg[0], dg[1]))
        print("Test Episodes: {}, Avg. Reward: {}, Success: {}".format(self.total_episode, round(avg_reward, 2),
                                                                       round(avg_success, 2)))
        print(' ')
        return avg_reward, avg_success

    def low_agent_train(self):
        if self.low_replay.__len__() > self.args.low_batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                (self.low_agent.update_parameters(
                    self.low_replay, self.args.low_batch_size, self.low_agent_update))
            self.writer.add_scalar('loss/low_critic_1', critic_1_loss, self.low_agent_update)
            self.writer.add_scalar('loss/low_critic_2', critic_2_loss, self.low_agent_update)
            self.writer.add_scalar('loss/low_policy', policy_loss, self.low_agent_update)
            self.writer.add_scalar('loss/low_entropy_temprature', alpha, self.low_agent_update)
            self.low_agent_update += 1
        else:
            pass

    def high_agent_train(self):
        if self.high_replay.__len__() > self.args.high_batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                (self.high_agent.update_parameters(
                    self.high_replay, self.args.high_batch_size, self.high_agent_update))
            self.writer.add_scalar('loss/high_critic_1', critic_1_loss, self.high_agent_update)
            self.writer.add_scalar('loss/high_critic_2', critic_2_loss, self.high_agent_update)
            self.writer.add_scalar('loss/high_policy', policy_loss, self.high_agent_update)
            self.writer.add_scalar('loss/high_entropy_temprature', alpha, self.high_agent_update)
            self.high_agent_update += 1
        else:
            pass

    def run(self):
        Training_epoch = 0
        while self.train_flag:
            if not self.args.debug_mode:
                print('Training Epoch %d: Iter (out of %d)=' % (Training_epoch, self.args.eval_freq), end=' ')
                sys.stdout.flush()

            for n_iter in range(self.args.eval_freq):
                if not self.args.debug_mode:
                    if n_iter % 5 == 0:
                        print("%d" % n_iter, end=' ' if n_iter < self.args.eval_freq - 1 else '\n')
                        sys.stdout.flush()
                observation, done = self.env.reset(), False
                ob = observation['observation']
                dg = observation['desired_goal']
                ag = observation['achieved_goal']
                episode_step = 0
                episode_reward = 0
                worker_reward = 0
                manager_reward = 0
                while not done:
                    ob_high = deepcopy(ob)
                    if self.total_numsteps <= self.args.start_steps:
                        sg = self.env.goal_space.sample()
                    else:
                        sg = self.high_agent.select_action(ob_high, dg)
                    sg_low = sg + ag
                    reward_high = 0
                    temp_buffer = ReplayMemory(self.args.temporal_horizon, self.args.seed)
                    for _ in range(self.args.temporal_horizon):
                        if self.total_numsteps <= self.args.start_steps:
                            action = self.env.action_space.sample()
                        else:
                            action = self.low_agent.select_action(ob, sg_low)
                        new_observation, reward, _, _ = self.env.step(action)
                        new_ob = new_observation['observation']
                        new_ag = new_observation['achieved_goal']
                        reward_high += self.args.high_reward_scale * self.high_reward_func(new_ag, dg)
                        reward_low = self.args.low_reward_scale * self.low_reward_func(sg_low, new_ag)
                        worker_reward += reward_low
                        episode_reward += reward
                        mask = 1 if episode_step == self.args.max_steps else float(not done)
                        temp_buffer.push(ob, sg_low, action, reward_low, new_ob, mask)

                        if self.total_numsteps % self.args.low_agent_train_freq == 0:
                            self.low_agent_train()
                        if self.total_numsteps % self.args.high_agent_train_freq == 0:
                            self.high_agent_train()
                        ob = new_ob
                        ag = new_ag

                        self.total_numsteps += 1
                        episode_step += 1

                        if episode_step >= self.args.max_steps:
                            done = True
                            break
                    new_ob_high = new_ob
                    # subgoal reachability computation
                    subgoal_reachability = temp_buffer.buffer[-1][3] / temp_buffer.buffer[0][3]

                    # low-level reward bonus
                    for idx in range(self.args.temporal_horizon):
                        ob, sg_low, action, reward_low, new_ob, mask = temp_buffer.buffer[idx]
                        reward_low += reward_low + self.args.low_reward_bonus * subgoal_reachability
                        self.low_replay.push(ob, sg_low, action, reward_low, new_ob, mask)
                    self.high_replay.push(ob_high, dg, sg, reward_high, new_ob_high, mask)

                    manager_reward += reward_high

                final_distance = -self.low_reward_func(new_ag, dg)
                if self.args.debug_mode:
                    print("total steps: %d achieved goal: (%.2f, %.2f), goal: (%.2f, %.2f)" % (
                        self.total_numsteps, new_ag[0], new_ag[1], dg[0], dg[1]))
                    print(' ')
                self.writer.add_scalar('reward/episode_reward', episode_reward, self.total_numsteps)
                self.writer.add_scalar('reward/worker_reward', worker_reward, self.total_numsteps)
                self.writer.add_scalar('reward/manager_reward', manager_reward, self.total_numsteps)
                self.writer.add_scalar('reward/final_distance', final_distance, self.total_numsteps)
                self.total_episode += 1

            print("Total Training Steps: %d" % self.total_numsteps)
            self.run_eval()
            Training_epoch += self.args.eval_freq

            # if self.total_episode % 400 == 0:
            #     self.high_agent.save_checkpoint(self.args.model_path, self.total_episode, 'high')
            #     self.low_agent.save_checkpoint(self.args.model_path, self.total_episode, 'low')

            if self.total_numsteps >= self.args.num_steps:
                self.train_flag = False
