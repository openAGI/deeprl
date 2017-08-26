import numpy as np
import tensorflow as tf
import logger as log

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class VanillaPolicyGradient(object):

    def __init__(
            self,
            n_actions,
            n_features,
            sess,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.sess = sess
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        if output_graph:
            tf.summary.FileWriter("/tmp/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        layer = tf.layers.dense(
            inputs=layer,
            units=100,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        # fc3
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='logits'
        )

        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads_vars = optimizer.compute_gradients(
                loss, var_list=tf.trainable_variables())

            clipped_grads_vars = []
            for grad, var in grads_vars:
                if grad is not None:
                    if isinstance(grad, tf.IndexedSlices):
                        tmp = tf.clip_by_norm(grad.values, max_norm)
                        grad = tf.IndexedSlices(
                            tmp, grad.indices, grad.dense_shape)
                    else:
                        grad = tf.clip_by_norm(grad, 100)
                clipped_grads_vars.append((grad, var))

            self.train_op = optimizer.apply_gradients(grads_vars)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
                                     self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
