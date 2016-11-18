import tensorflow as tf
from tqdm import tqdm

from core.history import History
from dataset.replay import ExperienceBuffer
from models.custom_model import Model
from core.base import Base
from utils import utils


class Player(Base):
    def __init__(self, cfg, environment, sess, model_dir):
        super(Player, self).__init__(cfg)
        self.sess = sess
        self.inputs = tf.placeholder('float32', [None, self.cfg.screen_height, self.cfg.screen_width, self.cfg.history_length], name='inputs')
        self.target_inputs = tf.placeholder('float32', [None, self.cfg.screen_height, self.cfg.screen_width, self.cfg.history_length], name='target_inputs')
        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int64', [None], name='action')
        self.env = environment
        self.history = History(self.cfg)
        self.model_dir = model_dir
        self.memory = ExperienceBuffer(cfg, self.model_dir)
        self.learning_rate_minimum = 0.0001
        self.double_q = True

    def play(self, load_model=True, test_ep=None, num_step=100000, num_episodes=200, display=True):
        model_q = Model()
        model_target_q = Model(is_target_q=True)
        end_points_q = model_q.model_def(self.inputs, self.env, name='main_q')
        _ = model_target_q.model_def(self.target_inputs, self.env, name='target_q')

        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep=None)

        if load_model:
            utils.load_model(self.saver, self.sess, self.model_dir)
        else:
            self.sess.run(init)

        if test_ep is None:
            test_ep = self.cfg.ep_test

        if not display:
            gym_dir = '/tmp/%s-%s' % (self.cfg.env_name, utils.get_time())
            self.env.env.monitor.start(gym_dir)

        best_reward, best_episode = 0, 0
        for episode in xrange(num_episodes):
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0

            for _ in xrange(self.cfg.history_length):
                self.history.add(screen)

            for t in tqdm(xrange(num_step), ncols=70):
                # 1. predict
                action = self.predict(end_points_q['pred_action'], self.history.get(), ep=test_ep)
                # 2. act
                screen, reward, terminal = self.env.act(action, is_training=False)
                # 3. observe
                self.history.add(screen)

                current_reward += reward
                if terminal:
                    break

            if current_reward > best_reward:
                best_reward = current_reward
                best_episode = episode

            print " [%d] Best reward : %d" % (best_episode, best_reward)

        if not display:
            self.env.env.monitor.close()
