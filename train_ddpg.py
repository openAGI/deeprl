# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import random
import os
import tensorflow as tf

from core.solver_ddpg import SolverDDPG
from env.environment import GymEnvironment, SimpleGymEnvironment
from config.config import cfg

# Set random seed
tf.set_random_seed(123)
random.seed(12345)


def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        cfg.env_name = 'Pendulum-v0'
        cfg.env_type = 'simple'
        if cfg.env_type == 'simple':
            env = SimpleGymEnvironment(cfg)
        else:
            env = GymEnvironment(cfg)

        if not os.path.exists('/tmp/model_dir'):
            os.mkdir('/tmp/model_dir')

        solver = SolverDDPG(cfg, env, sess, '/tmp/model_dir')

        solver.train()

if __name__ == '__main__':
    tf.app.run()
