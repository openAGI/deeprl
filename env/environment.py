# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import gym
import random
import numpy as np

try:
    import scipy.misc
    imresize = scipy.misc.imresize
except:
    import cv2
    imresize = cv2.resize


class Environment(object):
    def __init__(self, cfg):
        self.env = gym.make(cfg.env_name)

        self.action_repeat, self.random_start = cfg.action_repeat, cfg.random_start

        self.display = cfg.display
        self.dims = (cfg.screen_width, cfg.screen_height)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        try:
            if self.lives == 0:
                self._screen = self.env.reset()
                self._step(0)
                self.render()
        except AttributeError:
            self._screen = self.env.reset()
            self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in xrange(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @ property
    def screen(self):
        y = 0.2126 * self._screen[:, :, 0] + 0.7152 * self._screen[:, :, 1] + 0.0722 * self._screen[:, :, 2]
        y = y.astype(np.uint8)
        return imresize(y, self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def action_bound(self):
        return self.env.action_space.high

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()


class GymEnvironment(Environment):
    def __init__(self, cfg):
        super(GymEnvironment, self).__init__(cfg)

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.lives

        for _ in xrange(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulated_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Environment):
    def __init__(self, cfg):
        super(SimpleGymEnvironment, self).__init__(cfg)

    def act(self, action, is_training=True):
        self._step(action)

        self.after_act(action)
        return self.state
