"""Adapted from gym/envs/mujoco/inverted_double_pendulum.py"""
import numpy as np
from gym import utils
from Environments.DoubleInvertedPendulum import mujoco_env

current_y = 0

class DoubleInvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum.xml', 5)
        utils.EzPickle.__init__(self)

    def max_step(self):
        return 100

    def step(self, action):
        global current_y
        self.do_simulation(action, self.frame_skip)
        new_obs = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        r = - dist_penalty - vel_penalty
        if y > 1.0:
           current_y = current_y + 1
        else:
           current_y = 0
        if current_y == 10:
           done = 1
        else:
           done = 0
        next_obs = {
        'state': new_obs.copy(),
        'achieved_goal': current_y
        }
        if current_y == 10:
           current_y = 0
        return next_obs, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        obs = self._get_obs()
        global current_y
        current_y = 0
        new_obs = {
        'state': obs.copy(),
        'achieved_goal': current_y}
        return new_obs

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
