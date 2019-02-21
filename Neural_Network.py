import tensorflow as tf
from Policies import *
from utility import EpisodeStats, ReplayBuffer
import sys
import numpy as np

from itertools import chain
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
        has #A neurons, where #A is the num ber of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
        a learning rate of 0.0005). For initialization, you can simply use a uniform
        distribution (-0.5, 0.5), or something different.
        """

        self.inp = tf.placeholder(shape=[None, 3], dtype=tf.float64)
        self.inpW = tf.Variable(tf.random_uniform([3, 64], -0.5, 0.5,dtype=tf.float64))
        self.inpB = tf.Variable(tf.constant(0.1, shape=[64],dtype=tf.float64))

        self.h1 = tf.nn.relu(tf.matmul(self.inp, self.inpW) + self.inpB)
        self.h1W = tf.Variable(tf.random_uniform([64, 128], -0.5, 0.5,dtype=tf.float64))
        self.h1B = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float64))

        self.h3 = tf.nn.relu(tf.matmul(self.h1, self.h1W) + self.h1B)
        self.h3W = tf.Variable(tf.random_uniform([128, num_actions], -0.5, 0.5,dtype=tf.float64))
        self.h3b = tf.Variable(tf.constant(0.1, shape=[num_actions],dtype=tf.float64))

        self.out = tf.nn.softmax(tf.matmul(self.h3, self.h3W) + self.h3b)

        self.y_ = tf.placeholder(shape=[None, num_actions], dtype=tf.float64)
       #self.q_v = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)


        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)#,  momentum=0.95, epsilon=0.01)
        self.loss = tf.reduce_sum(tf.square(self.y_ - self.out))
        #self.loss = tf.reduce_sum(tf.square(self.y_ - self.q_v))
        #self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ , logits= self.out)
        #self.loss = tf.constant(0.0, dtype=tf.float32)

        self.step = self.trainer.minimize(self.loss)
        #train_writer = tf.summary.FileWriter('./log/loss', sess.graph)
        tf.summary.merge_all()
        # self.saver = tf.train.Saver()

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        states = np.atleast_2d(states)

#        states = np.reshape(states, [len(states), 3])
        # print(states)
        feed = {self.inp: states}
        prediction = sess.run(self.out, feed)
        return prediction

    def update(self, sess, states, actions, targets, pred, env, action_space, summary):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """

        #states = np.atleast_2d(states)
        #states = np.reshape(states, [len(states), 3])



        loss, summ = sess.run((self.loss, self.step), feed_dict={self.inp: states, self.y_: pred})
        #self.y_: targets, self.q_v: rwds
        #return loss, summ
        return loss

    def get_loss(self):
        return self.loss

    # Why is relative path not working?
    def save_model(self, sess):  #
        self.saver.save(sess, "./models/" + self.name + "/model_final.ckpt")

    def load_model(self, sess):
        self.saver.restore(sess, "./models/" + self.name + "/model.ckpt")
