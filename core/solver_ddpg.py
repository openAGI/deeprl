# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from models.ddpg_model import ActorModel, CriticModel
from core.base import Base
from utils import utils


class SolverDDPG(Base):
    def __init__(self, cfg, environment, sess, model_dir, **kwargs):
        self.s_dim = environment.state_dim
        self.a_dim = environment.action_dim
        self.inputs_actor = tf.placeholder('float32', [None, self.s_dim], name='inputs_actor')
        self.target_inputs_actor = tf.placeholder('float32', [None, self.s_dim], name='target_inputs_actor')
        self.inputs_critic = tf.placeholder('float32', [None, self.s_dim], name='inputs_critic')
        self.target_inputs_critic = tf.placeholder('float32', [None, self.s_dim], name='target_inputs_critic')
        self.actions = tf.placeholder('float32', [None, self.a_dim], name='actions')
        self.target_actions = tf.placeholder('float32', [None, self.a_dim], name='target_actions')
        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
        super(SolverDDPG, self).__init__(cfg, environment, sess, model_dir, **kwargs)

    def train(self):
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_random_game()
        i = np.random.randint(11)
        j = np.random.randint(19)
        first_input = np.reshape(screen, (1, 3)) + (1. / (1. + i + j))
        action = self.predict(self.end_points_actor['scaled_out'], first_input, agent_type='actor')

        self.optim_actor, self.loss_actor, self.end_points_actor, self.end_points_target_actor = self.tower_loss_actor(self.actor_inputs, self.actor_target_inputs, actor_name='actor/main_')
        self.optim_critic, self.loss_critic, self.end_points_critic, self.end_points_target_critic = self.tower_loss_critic(self.critic_inputs, self.critic_target_inputs, self.actions, self.target_actions, critic_name='critic/main_')
        tvariables_actor = [var for var in tf.trainable_variables() if var.name.startswith('actor/')]
        tvariables_critic = [var for var in tf.trainable_variables() if var.name.startswith('critic/')]
        self.targetops_actor = self.update_target_graph(tvariables_actor, self.cfg.tau, main_name='actor/main_', target_name='critic/target')
        self.targetops_critic = self.update_target_graph(tvariables_critic, self.cfg.tau, main_name='critic/main_', target_name='critic/target_')
        self.saver = tf.train.Saver(max_to_keep=None)

        init = tf.initialize_all_variables()
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
            ep = (self.cfg.ep_end + max(0., (self.cfg.ep_start - self.cfg.ep_end) * (self.cfg.ep_end_t - max(0., self.step - self.cfg.learn_start)) / self.cfg.ep_end_t))
            # 1. predict
            action = self.predict(self.end_points_actor['scaled_out'], self.history.get(), ep=ep, agent_type='actor')
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(np.reshape(screen, self.s_dim), reward, np.reshape(action, self.a_dim), terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

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

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        utils.save_model(self.saver, self.sess, self.model_dir, self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

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
                self.update_target(self.targetops_actor, self.sess)
                self.update_target(self.targetops_critic, self.sess)

    def train_mini_batch(self):
        if self.memory.count < self.cfg.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        ep = (self.cfg.ep_end + max(0., (self.cfg.ep_start - self.cfg.ep_end) * (self.cfg.ep_end_t - max(0., self.step - self.cfg.learn_start)) / self.cfg.ep_end_t))
        action_s_t_plus_1 = self.predict_target(self.end_points_target_actor['scaled_out'], s_t_plus_1, ep=ep, agent_type='actor')
        target_q = self.end_points_target_critic['out'].eval({self.target_inputs: s_t_plus_1, self.target_actions: action_s_t_plus_1})

        terminal = np.array(terminal) + 0.
        target_q_t = (1. - terminal) * self.cfg.discount * target_q + reward

        _, q_t, loss = self.sess.run([self.optim_critic, self.end_points_critic['out'], self.loss], {
            self.predicted_q_value: target_q_t,
            self.actions: action,
            self.critic_inputs: s_t,
            self.learning_rate: self.updated_lr})
        action_out = self.predict(self.end_points_actor['scaled_out'], s_t, ep=ep, agent_type='actor')
        a_grads = self.sess.run(self.action_gradients, {self.critic_inputs: s_t, self.actions: action_out})
        _, = self.sess.run(self.optim_actor, {self.actor_inputs: s_t, self.action_gradients: a_grads[0]})

        # self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def tower_loss_actor(self, inputs, target_inputs, actor_name='actor/main_'):
        model_actor = ActorModel()
        model_target_actor = ActorModel(is_target_q=True)
        end_points_actor = model_actor.model_def(inputs, self.env, name='main')
        # Target Network
        end_points_target_actor = model_target_actor.model_def(target_inputs, self.env, name='target')
        # This gradient will be provided by the critic network
        self.action_gradients = tf.placeholder(tf.float32, [None, self.a_dim])
        # Combine the gradients here
        self.actor_model_params = [var for var in tf.trainable_variables() if var.name.startswith(actor_name)]
        self.actor_gradients = tf.gradients(end_points_actor['scaled_out'], self.actor_model_params, -self.action_gradients)
        # Optimization Op
        opt = self.optimizer(self.learning_rate, optname=self.cfg.optname, decay=self.cfg.decay, momentum=self.cfg.momentum, epsilon=self.cfg.epsilon, beta1=self.cfg.beta1, beta2=self.cfg.beta2)

        optim = opt.apply_gradients(zip(self.actor_gradients, self.actor_model_params))
        return optim, end_points_actor, end_points_target_actor

    def tower_loss_critic(self, inputs, target_inputs, actions, target_actions, critic_name='critic/main_'):
        model_critic = CriticModel()
        model_target_critic = CriticModel(is_target_q=True)
        end_points_critic = model_critic.model_def(inputs, actions, name='main')
        # Target Network
        end_points_target_critic = model_target_critic.model_def(target_inputs, target_actions, self.env, name='target')
        # This gradient will be provided by the critic network
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        loss = tf.reduce_mean(tf.square(self.predicted_q_value, end_points_critic['out']))
        # Optimization Op
        opt = self.optimizer(self.learning_rate, optname=self.cfg.optname, decay=self.cfg.decay, momentum=self.cfg.momentum, epsilon=self.cfg.epsilon, beta1=self.cfg.beta1, beta2=self.cfg.beta2)
        self.critic_model_params = [var for var in tf.trainable_variables() if var.name.startswith(critic_name)]
        self.critic_gradients_vars = opt.compute_gradients(loss, self.critic_model_params)
        optim = opt.apply_gradients(self.critic_gradients_vars)
        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(end_points_critic['out'], actions)
        return optim, loss, end_points_critic, end_points_target_critic
