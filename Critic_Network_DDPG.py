from Neural_Network import NeuralNetwork
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np

ac_dim = 1


class Critic_Net(NeuralNetwork):
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.name = name
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        self.inp = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)

        self.inp_act = tf.concat([self.inp, self.action], -1)
        #
        # self.inpW = tf.Variable(tf.random_uniform([self.state_dim + self.action_dim, 64], -0.5, 0.5))
        # self.inpB = tf.Variable(tf.constant(0.1, shape=[64]))
        # self.h1 = tf.nn.relu(tf.matmul(self.inp_act, self.inpW) + self.inpB)
        #
        # self.h2W = tf.Variable(tf.random_uniform([64, 128], -0.5, 0.5))
        # self.h2B = tf.Variable(tf.constant(0.1, shape=[128]))
        # self.h2 = tf.nn.relu(tf.matmul(self.h1, self.h2W) + self.h2B)
        #
        # self.h3W = tf.Variable(tf.random_uniform([128, 64], -0.5, 0.5))
        # self.h3B = tf.Variable(tf.constant(0.1, shape=[64]))
        # self.h3 = tf.nn.relu(tf.matmul(self.h2, self.h3W) + self.h3B)
        #
        # self.h4W = tf.Variable(tf.random_uniform([64, self.action_dim], -0.5, 0.5))
        #
        # self.outB = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))
        # self.outputs = tf.nn.relu(tf.matmul(self.h3, self.h4W) + self.outB)
        self.dense1 = tf.layers.dense(self.inp_act, 32, tf.nn.relu,
                                      kernel_initializer=tf.random_uniform_initializer, trainable=True,
                                      name=self.name + "/dense1", reuse=tf.AUTO_REUSE)

        self.dense2 = tf.layers.dense(self.dense1, 64, tf.nn.relu,
                                      kernel_initializer=tf.initializers.random_uniform, trainable=True,
                                      name=self.name + "/dense2", reuse=tf.AUTO_REUSE)

        self.dense3 = tf.layers.dense(self.dense2, 32, tf.nn.relu,
                                      kernel_initializer=tf.random_uniform_initializer, trainable=True,
                                      name=self.name + "/dense3", reuse=tf.AUTO_REUSE)

        self.outputs = tf.layers.dense(self.dense3, self.action_dim, None,
                                       kernel_initializer=tf.random_uniform_initializer, trainable=True,
                                       name=self.name + "/outputs", reuse=tf.AUTO_REUSE)

        self.net_params = tf.trainable_variables(self.name)

        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.loss = tf.reduce_mean(tf.squared_difference(self.outputs, self.y_))

        self.step = self.trainer.minimize(self.loss)

        self.action_grads = tf.gradients(self.outputs, self.action)

        self.saver = tf.train.Saver()

    def predict(self, sess, states, actions):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), self.state_dim])

        feed = {self.inp: states, self.action: actions}
        prediction = sess.run(self.outputs, feed)

        return prediction

    def update(self, sess, states, actions, targets, summary):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """

        # pred = self.predict(sess, states, actions)
        states = np.atleast_2d(states)
        # states = np.reshape(states, [len(states), 3])
        return sess.run((self.loss, self.outputs, self.step, self.net_params),
                        feed_dict={self.inp: states, self.action: actions, self.y_: targets})

    def action_gradients(self, sess, states, actions):

        grads = sess.run(self.action_grads, feed_dict={
            self.inp: states,
            self.action: actions})
        return grads


class Critic_Target_Network(Critic_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, critic, learning_rate=0.001, batch_size=128,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate, batch_size)
        self.tau = tau
        self.critic = critic  # added line
        self._register_associate()  # modified line

    # modified method
    def _register_associate(self):
        self.init_target = [self.net_params[i].assign(self.critic.net_params[i]) for i in range(len(self.net_params))]

        self.update_target = [self.net_params[i].assign(
            tf.scalar_mul(self.tau, self.critic.net_params[i]) + tf.scalar_mul(1. - self.tau, self.net_params[i])) for i
            in range(len(self.net_params))]

    # added method. Target network starts identical to original network
    def init(self, sess):
        sess.run(self.init_target)

    # modified method
    def update(self, sess):
        sess.run(self.update_target)
