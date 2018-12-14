import tensorflow as tf
from Policies import *
from utility import EpisodeStats, ReplayBuffer
import sys
import numpy as np



class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """

    def __init__(self, num_actions, name):
        self._build_model(num_actions)
        self.name = name

    def _build_model(self, num_actions):
        """
        Creates a neural network, e.g. with two
        hidden fully connected layers and 20 neurons each). The output layer
        has #A neurons, where #A is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
        a learning rate of 0.0005). For initialization, you can simply use a uniform
        distribution (-0.5, 0.5), or something different.
        """

        self.inp = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        self.inpW = tf.Variable(tf.random_uniform([3, 64], -0.5, 0.5))
        self.inpB = tf.Variable(tf.constant(0.1, shape=[64]))

        self.h1 = tf.nn.relu(tf.matmul(self.inp, self.inpW) + self.inpB)
        self.h1W = tf.Variable(tf.random_uniform([64, 128], -0.5, 0.5))
        self.h1B = tf.Variable(tf.constant(0.1, shape=[128]))

        self.h3 = tf.nn.relu(tf.matmul(self.h1, self.h1W) + self.h1B)
        self.h3W = tf.Variable(tf.random_uniform([128, num_actions], -0.5, 0.5))
        self.h3b = tf.Variable(tf.constant(0.1, shape=[num_actions]))

        self.out = tf.matmul(self.h3, self.h3W) + self.h3b

        self.y_ = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.out))

        # train_writer = tf.summary.FileWriter('./log/loss', sess.graph)
        self.step = self.trainer.minimize(self.loss)

        tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 3])
        # print(states)
        feed = {self.inp: states}
        prediction = sess.run(self.out, feed)
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

        pred = self.predict(sess, states)


        for i in range(len(actions)):
            pred[i][actions[0]] = targets[i]

        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 3])
        loss, summ, _ = sess.run((self.loss, summary, self.step), feed_dict={self.inp: states, self.y_: pred})
        return self.loss, summ

    def get_loss(self):
        return self.loss

    # Why is relative path not working?
    def save_model(self, sess):  #
        self.saver.save(sess, "./models/" + self.name + "/model.ckpt")

    def load_model(self, sess):
        self.saver.restore(sess, "./models/" + self.name + "/model.ckpt")
