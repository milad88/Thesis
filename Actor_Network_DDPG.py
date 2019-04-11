import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected

action_bound = 2.0


class Actor_Net():
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01, batch_size=128):
        # super().__init__(num_actions, name)
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.name = name
        self._build_model()

    def _build_model(self):
        # self.action = tf.placeholder(dtype=tf.float128, shape=[None, self.action_dim])
        self.inp = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)
        #
        # self.inpW = tf.Variable(tf.random_uniform([self.state_dim, 64], -0.5, 0.5))
        # self.inpB = tf.Variable(tf.constant(0.1, shape=[64]))
        # self.h1 = tf.nn.relu(tf.matmul(self.inp, self.inpW) + self.inpB)
        # self.h2W = tf.Variable(tf.random_uniform([64, 128], -0.5, 0.5))
        # self.h2B = tf.Variable(tf.constant(0.1, shape=[128]))
        # self.h2 = tf.nn.relu(tf.matmul(self.h1, self.h2W) + self.h2B)
        #
        # self.h3W = tf.Variable(tf.random_uniform([128, 64], -0.5, 0.5))
        # self.h3B = tf.Variable(tf.constant(0.1, shape=[64]))
        # self.h3 = tf.nn.relu(tf.matmul(self.h2, self.h3W) + self.h3B)
        # self.h4W = tf.Variable(tf.random_uniform([64, self.action_dim], -0.5, 0.5))
        # self.outB = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))
        # self.outputs = tf.nn.tanh(tf.matmul(self.h3, self.h4W) + self.outB)
        self.dense1 = tf.layers.dense(self.inp, 16, tf.nn.relu,
                                      kernel_initializer=tf.random_uniform_initializer, trainable=True, name=self.name+"/dense1", reuse=tf.AUTO_REUSE)

        self.dense2 = tf.layers.dense(self.dense1, 32, tf.nn.relu,
                                      kernel_initializer=tf.initializers.random_uniform, trainable=True, name=self.name+"/dense2", reuse=tf.AUTO_REUSE)

        self.dense3 = tf.layers.dense(self.dense2, 8, tf.nn.tanh,
                                      kernel_initializer=tf.random_uniform_initializer, trainable=True, name=self.name+"/dense3", reuse=tf.AUTO_REUSE)

        self.outputs = tf.layers.dense(self.dense3, self.action_dim, tf.nn.tanh,
                                       kernel_initializer=tf.random_uniform_initializer, trainable=True, name=self.name+"/outputs", reuse=tf.AUTO_REUSE)
        self.net_params = tf.trainable_variables(self.name)

        # self.scaled_outputs = self.outputs *2
        # self.scaled_outputs = tf.scalar_mul(action_bound, self.outputs)

        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        # self.target_critic_output = tf.placeholder(tf.float32, [None, self.action_dim])

        # self.net_params = tf.get_default_graph().get_collection()#(["dense1", "dense2","dense3","outputs"])
        # self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.actor_gradients = tf.gradients(ys=self.outputs, xs=self.net_params, grad_ys=-self.action_gradients)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.actor_gradients, self.net_params))

        # self.step = self.trainer.minimize(self.loss)

        self.saver = tf.train.Saver()

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        # states = np.atleast_2d(states)
        # np.reshape(states, [len(states), self.state_dim])

        feed = {self.inp: states}
        prediction = sess.run(self.outputs, feed)
        return prediction

    # action gradient to be fed

    def update(self, sess, states, grads, summary):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """

        # pred = self.predict(sess, states)

        #  for i in range(len(actions)):
        #   pred[i][actions[i]] = targets[i]

        # states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), self.state_dim])
        # hists = []
        # for ix, grad in enumerate(self.actor_gradients):
        # self.actor_gradients[ix] = grad / float(self.batch_size)
        # hists.append(tf.summary.histogram(str(ix)+'/gradient', grad))
        sess.run((self.optimize, self.net_params), feed_dict={self.inp: states, self.action_gradients: grads[0]})


class Actor_Target_Network(Actor_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, actor, learning_rate=0.001, batch_size=128,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate, batch_size)
        # self._build_model( num_actions, action_dim, name, action_bound, state_dim)
        self.tau = tau
        self.actor = actor  # added line
        self._register_associate()  # modified line

        # modified method

    def _register_associate(self):
        # print(self.net_params)
        self.init_target = [self.net_params[i].assign(self.actor.net_params[i]) for i in range(len(self.net_params))]

        self.update_target = [self.net_params[i].assign(
            tf.scalar_mul(self.tau, self.actor.net_params[i]) + tf.scalar_mul(1. - self.tau, self.net_params[i])) for i
            in range(len(self.net_params))]

        # added method. Target network starts identical to original network

    def init(self, sess):
        sess.run(self.init_target)

        # modified method

    def update(self, sess):
        sess.run(self.update_target)
