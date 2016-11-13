import numpy as np
# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import random


class ExperienceBuffer():

    def __init__(self, cfg, model_dir):
        self.model_dir = model_dir
        self.buffer = []
        self.buffer_size = cfg.TRAIN.buffer_size

        self.cnn_format = cfg.cnn_format
        self.actions = np.empty(self.buffer_size, dtype=np.uint8)
        self.rewards = np.empty(self.buffer_size, dtype=np.integer)
        self.screens = np.empty((self.buffer_size, cfg.screen_height, cfg.screen_width), dtype=np.float16)
        self.terminals = np.empty(self.buffer_size, dtype=np.bool)
        self.history_length = cfg.history_length
        self.dims = (cfg.screen_height, cfg.screen_width)
        self.batch_size = cfg.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size

    def getState(self, index):
        assert self.count > 0, "replay buffer is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]

    def sample(self):
        # buffer must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

            actions = self.actions[indexes]
            rewards = self.rewards[indexes]
            terminals = self.terminals[indexes]

            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals

    def add_experience(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample_experience(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
