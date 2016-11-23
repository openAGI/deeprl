import tensorflow as tf
import random
from utils import utils


class Base(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def predict(self, pred_action_op, s_t, ep=0.1):
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = pred_action_op.eval({self.inputs: [s_t]})[0]

        return action

    def predict_target(self, pred_action_op, s_t, ep=0.1):
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = pred_action_op.eval({self.target_inputs: [s_t]})[0]

        return action

    def update_target_graph(self, tfVars, tau, main_name='main_q', target_name='target_q'):
        main_target = utils.get_vars_main_target(tfVars, main_name=main_name, target_name=target_name)
        op_holder = []
        for idx, var in enumerate(main_target['main_vars']):
            op_holder.append(main_target['target_vars'][idx].assign((var.value() * tau) + ((1 - tau) * main_target['target_vars'][idx].value())))
        return op_holder

    def update_target(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)
        total_vars = len(tf.trainable_variables())
        a = tf.trainable_variables()[0].eval(session=sess)
        b = tf.trainable_variables()[total_vars / 2].eval(session=sess)
        if a.all() == b.all():
            print "Target Set Success"
        else:
            print "Target Set Failed"

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
            opt = tf.train.AdadeltaOptimizer(learning_rate=lr, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
        if optname == 'adagrad':
            opt = tf.train.AdagradOptimizer(lr, initial_accumulator_value=0.1, use_locking=False, name='Adadelta')
        if optname == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.0, epsilon=epsilon)
        if optname == 'momentum':
            opt = tf.train.MomentumOptimizer(lr, momentum, use_locking=False, name='momentum', use_nesterov=True)
        if optname == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=False, name='Adam')
        return opt
