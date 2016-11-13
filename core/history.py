# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import numpy as np


class History:
    def __init__(self, cfg):
        self.cnn_format = cfg.cnn_format
        self.history = np.zeros([cfg.history_length, cfg.screen_height, cfg.screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        return np.transpose(self.history, (1, 2, 0))
