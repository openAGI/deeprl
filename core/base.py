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

    def update_target_graph(self, tfVars, tau):
        main_target = utils.get_vars_main_target(tfVars)
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
