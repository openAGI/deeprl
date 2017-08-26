# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import random
import numpy as np
from core.history import History
from core.lr_policy import PolyDecayPolicy
from dataset.replay import ExperienceBuffer
import core.logger as log
from utils import utils


class Base(object):

    def __init__(self, cfg, environment, sess, model_dir, state='image', state_dim=3, log_file_pathname='/tmp/deeprl.log', verbosity=1, lr_policy=PolyDecayPolicy(0.001), start_epoch=1, resume_lr=0.001, n_iters_per_epoch=100, gpu_memory_fraction=0.9):
        self.cfg = cfg
        self.sess = sess
        self.weight_dir = 'weights'
        self.env = environment
        self.history = History(self.cfg, state=state, state_dim=state_dim)
        self.model_dir = model_dir
        self.memory = ExperienceBuffer(
            self.cfg, self.model_dir, state=state, state_dim=state_dim)
        self.learning_rate_minimum = 0.0001
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        self.lr_policy = lr_policy
        self.lr_policy.start_epoch = start_epoch
        self.lr_policy.base_lr = resume_lr
        self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
        self.gpu_memory_fraction = gpu_memory_fraction
        log.setFileHandler(log_file_pathname)
        log.setVerbosity(str(verbosity))

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

    def predict(self, pred_action_op, s_t, ep=0.1, agent_type=None, batch_size=32, is_train_loop=False):
        if agent_type is None:
            if random.random() < ep:
                action = random.randrange(self.env.action_size)
            else:
                action = pred_action_op.eval({self.inputs: [s_t]})[0]
        elif agent_type == 'actor':
            if random.random() < ep and not is_train_loop:
                action = random.randrange(self.env.action_dim)
                # action = np.reshape(np.array([random.randrange(self.env.action_dim) for i in xrange(0, batch_size)]), (batch_size, self.env.action_dim))
            else:
                action = pred_action_op.eval({self.inputs_actor: s_t})
        elif agent_type == 'critic':
            if random.random() < ep:
                action = random.randrange(self.env.action_dim)
            else:
                action = pred_action_op.eval({self.inputs_critic: s_t})[0]

        action = np.clip(action, self.env.action_low, self.env.action_high)
        return action

    def predict_target(self, pred_action_op, s_t, ep=0.1, agent_type=None):
        if agent_type is None:
            if random.random() < ep:
                action = random.randrange(self.env.action_size)
            else:
                action = pred_action_op.eval({self.target_inputs: [s_t]})[0]
        elif agent_type == 'actor':
            action = pred_action_op.eval({self.target_inputs_actor: s_t})
        elif agent_type == 'critic':
            if random.random() < ep:
                action = random.randrange(self.env.action_dim)
            else:
                action = pred_action_op.eval(
                    {self.target_inputs_critic: s_t})[0]

        action = np.clip(action, self.env.action_low, self.env.action_high)
        return action

    def update_target_graph(self, tfVars, tau, main_name='main_q', target_name='target_q'):
        main_target = utils.get_vars_main_target(
            tfVars, main_name=main_name, target_name=target_name)
        op_holder = []
        for idx, var in enumerate(main_target['main_vars']):
            op_holder.append(main_target['target_vars'][idx].assign(
                (var.value() * tau) + ((1 - tau) * main_target['target_vars'][idx].value())))
        return op_holder

    def update_target(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)
        total_vars = len(tf.trainable_variables())
        a = tf.trainable_variables()[0].eval(session=sess)
        b = tf.trainable_variables()[total_vars / 2].eval(session=sess)
        if a.all() == b.all():
            log.info('Target Set Success')
        else:
            log.error('Target Set Failed')

    def optimizer(self, lr, optname='momentum', decay=0.9, momentum=0.9, epsilon=0.000000008, beta1=0.9, beta2=0.999):
        """ definew the optimizer to use.

        Args:
            lr: learning rate, a scalar or a policy
            optname: optimizer name
            decay: variable decay value, scalar
            momentum: momentum value, scalar

        Returns:
            optimizer to use
         """
        if optname == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=lr, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
            log.info('Using adadelta optimizer for learning')
        if optname == 'adagrad':
            opt = tf.train.AdagradOptimizer(
                lr, initial_accumulator_value=0.1, use_locking=False, name='Adadelta')
            log.info('Using adagrad optimizer for learning')
        if optname == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(
                lr, decay=0.9, momentum=0.0, epsilon=epsilon)
            log.info('Using rmsprop optimizer for learning')
        if optname == 'momentum':
            opt = tf.train.MomentumOptimizer(
                lr, momentum, use_locking=False, name='momentum', use_nesterov=True)
            log.info('Using momentum optimizer for learning')
        if optname == 'adam':
            opt = tf.train.AdamOptimizer(
                learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=False, name='Adam')
            log.info('Using adam optimizer for learning')
        return opt
