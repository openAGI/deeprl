# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import cv2
import gym
import random


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
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
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
        return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY) / 255., self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

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
