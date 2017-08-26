# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from models.custom_model import Model
from core.base import Base
from utils import utils
from core import logger


class Solver(Base):

    def __init__(self, cfg, environment, sess, model_dir, double_q=True, **kwargs):
        self.inputs = tf.placeholder('float32', [
                                     None, cfg.screen_height, cfg.screen_width, cfg.history_length], name='inputs')
        self.target_inputs = tf.placeholder('float32', [
                                            None, cfg.screen_height, cfg.screen_width, cfg.history_length], name='target_inputs')
        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int64', [None], name='action')
        self.double_q = double_q
        super(Solver, self).__init__(
            cfg, environment, sess, model_dir, **kwargs)

    def train(self):
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_random_game()
        self.optim, self.loss, self.end_points_q, self.end_points_target_q = self.tower_loss(
            self.inputs, self.target_inputs)
        self.targetops = self.update_target_graph(
            tf.trainable_variables(), self.cfg.tau)
        self.saver = tf.train.Saver(max_to_keep=None)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        start_step = self.step_op.eval()

        for _ in range(self.cfg.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.cfg.max_step), ncols=70, initial=start_step):
            if self.step == self.cfg.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            self.updated_lr = self.lr_policy.initial_lr
            ep = (self.cfg.ep_end + max(0., (self.cfg.ep_start - self.cfg.ep_end) *
                                        (self.cfg.ep_end_t - max(0., self.step - self.cfg.learn_start)) / self.cfg.ep_end_t))
            # 1. predict
            action = self.predict(
                self.end_points_q['pred_action'], self.history.get(), ep=ep)
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward
            logger.debug('Step: %d, Reward: %f' % (self.step, reward))
            logger.debug('Step: %d, Expected Reward: %f' %
                         (self.step, ep_reward))

            actions.append(action)
            total_reward += reward
            logger.debug('Step: %d, Total Reward: %f' %
                         (self.step, total_reward))

            if self.step >= self.cfg.learn_start:
                if self.step % self.cfg.test_step == self.cfg.test_step - 1:
                    avg_reward = total_reward / self.cfg.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)
                    logger.debug('Step: %d, Avg_R: %.4f, Avg_L: %.6f, Avg_Q: %3.6f, Avg_EP_R: %.4f, Max_EP_R: %.4f, Min_EP_R: %.4f, # Game: %d' % (
                        self.step, avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval(
                            {self.step_input: self.step + 1})
                        utils.save_model(self.saver, self.sess,
                                         self.model_dir, self.step + 1)

                        max_avg_ep_reward = max(
                            max_avg_ep_reward, avg_ep_reward)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
        end_time = time.time()
        print('Total training time %6.1fs' % start_time - end_time)

    def observe(self, screen, reward, action, terminal):
        reward = max(self.cfg.min_reward, min(self.cfg.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.cfg.learn_start:
            if self.step % self.cfg.train_frequency == 0:
                self.train_mini_batch()

            if self.step % self.cfg.target_q_update_step == self.cfg.target_q_update_step - 1:
                self.update_target(self.targetops, self.sess)

    def train_mini_batch(self):
        if self.memory.count < self.cfg.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        if self.double_q:
            # Double Q-learning
            pred_action = self.end_points_q[
                'pred_action'].eval({self.inputs: s_t_plus_1})

            q_t_plus_1_with_pred_action = self.end_points_target_q['target_q_with_idx'].eval({
                self.target_inputs: s_t_plus_1,
                self.end_points_target_q['target_q_idx']: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
            target_q_t = (1. - terminal) * self.cfg.discount * \
                q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.end_points_target_q[
                'q'].eval({self.target_inputs: s_t_plus_1})

            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.cfg.discount * \
                max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optim, self.end_points_q['q'], self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.inputs: s_t,
            self.learning_rate: self.updated_lr})

        # self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def tower_loss(self, inputs, target_inputs):
        model_q = Model()
        model_target_q = Model(is_target_q=True)
        end_points_q = model_q.model_def(inputs, self.env, name='main_q')
        end_points_target_q = model_target_q.model_def(
            target_inputs, self.env, name='target_q')

        action_one_hot = tf.one_hot(
            self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
        q_acted = tf.reduce_sum(
            end_points_q['q'] * action_one_hot, reduction_indices=1, name='q_acted')

        delta = self.target_q_t - q_acted
        clipped_delta = tf.clip_by_value(
            delta, self.cfg.min_delta, self.cfg.max_delta, name='clipped_delta')

        loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
        opt = self.optimizer(self.learning_rate, optname='rmsprop', decay=self.cfg.TRAIN.rmsprop_decay,
                             momentum=self.cfg.TRAIN.rmsprop_momentum, epsilon=self.cfg.TRAIN.rmsprop_epsilon, beta1=self.cfg.TRAIN.rmsprop_beta1, beta2=self.cfg.TRAIN.rmsprop_beta2)
        self.grads_and_vars = opt.compute_gradients(loss)
        optim = opt.apply_gradients(self.grads_and_vars)
        return optim, loss, end_points_q, end_points_target_q
