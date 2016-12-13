# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAIN = edict()
__C.VAL = edict()
__C.num_gpus = 1
__C.scale = 10000
__C.display = False
__C.max_step = 5000 * __C.scale
__C.memory_size = 100 * __C.scale

__C.random_start = 30
__C.cnn_format = 'NCHW'
__C.discount = 0.99
__C.target_q_update_step = 1 * __C.scale
__C.TRAIN.learning_rate = 0.00025
__C.TRAIN.learning_rate_minimum = 0.00025
__C.TRAIN.learning_rate_decay = 0.96
__C.TRAIN.learning_rate_decay_step = 5 * __C.scale
__C.TRAIN.buffer_size = 50000

__C.TRAIN.optname='rmsprop'
__C.ep_test = 0.01
__C.ep_end = 0.1
__C.ep_start = 1.
__C.ep_end_t = __C.memory_size
__C.tau = 0.001
__C.history_length = 4
__C.train_frequency = 4
__C.batch_size = 32
__C.learn_start = 5. * __C.scale

__C.min_delta = -1
__C.max_delta = 1

__C.double_q = False
__C.dueling = False

__C.test_step = 5 * __C.scale
__C.save_step = __C.test_step * 10

__C.env_name = 'Breakout-v0'

__C.screen_width = 84
__C.screen_height = 84
__C.max_reward = 1.
__C.min_reward = -1.

__C.model = 'MODEL1'

__C.backend = 'tf'
__C.env_type = 'detail'
__C.action_repeat = 1

__C.memory_allocation = 1
__C.TRAIN.batch_size = 16
__C.TRAIN.validation_split = 10
__C.TRAIN.learning_rate = 0.001
__C.TRAIN.num_epochs = 200
__C.TRAIN.training_size = 40000
__C.TRAIN.num_class = 5
__C.TRAIN.checkpoint_dir = '/media/Data/output/ckpt'
__C.TRAIN.im_depth = 3
__C.num_dataprocess_threads = 8
__C.num_readers = 4
__C.queue_memory_factor = 8
__C.input_queue_memory_factor = 8
__C.TRAIN.max_angle = 11
__C.TRAIN.sigma_max = 0.01
__C.num_threads = 8
__C.num_shards = 8
__C.output_dir = '/media/Data/output'
__C.TRAIN.fine_tune = False
__C.TRAIN.moving_average_decay = 0.9999
__C.TRAIN.rmsprop_decay = 0.9
__C.TRAIN.rmsprop_momentum = 0.9
__C.TRAIN.rmsprop_epsilon = 1.0
__C.TRAIN.sgd_momentum = 0.9
__C.TRAIN.adam_beta1 = 0.9
__C.TRAIN.adam_beat2 = 0.999
__C.TRAIN.adam_epsilon = 1e-08
__C.TRAIN.num_epochs_per_decay = 30
__C.TRAIN.learning_rate_decay_factor = 0.16
__C.TRAIN.max_steps = 9100000
__C.VAL.batch_size = 64
__C.VAL.subset = 'val'
