from Neural_Network import NeuralNetwork
import tensorflow as tf
import numpy as np

ac_dim = 1


class Critic_Net(NeuralNetwork):
    def __init__(self, num_actions, action_dim, name, action_bound, state_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = name
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._build_model(num_actions)

    def _build_model(self, num_actions):

        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        self.inp = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)

        self.inp_act = tf.concat([self.inp, self.action], 1)

        self.inpW = tf.Variable(tf.random_uniform([self.state_dim +  self.action_dim, 16], -0.5, 0.5))
        self.inpB = tf.Variable(tf.constant(0.1, shape=[16]))
        self.h1 = tf.nn.relu(tf.matmul(self.inp_act, self.inpW) + self.inpB)

        self.h2W = tf.Variable(tf.random_uniform([16, 32], -0.5, 0.5))
        self.h2B = tf.Variable(tf.constant(0.1, shape=[32]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.h2W) + self.h2B)

        self.h3W = tf.Variable(tf.random_uniform([32, 16], -0.5, 0.5))
        self.h3B = tf.Variable(tf.constant(0.1, shape=[16]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.h3W) + self.h3B)

        self.h4W = tf.Variable(tf.random_uniform([16, self.action_dim], -0.5, 0.5))

        self.outB = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))
        self.outputs = tf.nn.relu(tf.matmul(self.h3, self.h4W) + self.outB)

        # self.out = np.rint(self.out) # round to 0 or 1 as out
        self.y_ = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32)

        #        Q_grad = K.gradients(Q_pred, actions)

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
        states = np.reshape(states, [len(states), 3])

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

        #pred = self.predict(sess, states, actions)


        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 3])
        return sess.run((self.loss, self.outputs, self.step), feed_dict={self.inp: states,self.action: actions, self.y_: targets})


    def action_gradients(self, sess, states, actions):
        return sess.run(self.action_grads, feed_dict={
            self.inp: states,
            self.action: actions})


class Critic_Target_Network(Critic_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, num_actions, action_dim, name, action_bound, state_dim, learning_rate=0.001, tau=0.001):
        super().__init__(num_actions, action_dim, name, action_bound, state_dim, learning_rate)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):

        critic_vars =tf.trainable_variables()#"critic"
        target_vars =tf.trainable_variables()#"critic_target"

        #print(tf_vars)

        #total_vars = len(tf_vars)

        op_holder = []
        for idx, var in enumerate(target_vars):  # // is to retun un integer
            op_holder.append(var.assign(
                (critic_vars[idx].value() * self.tau) + ((1 - self.tau) * var.value())))
        #return target_vars.assign((critic_vars * self.tau )+((1 - self.tau) * target_vars))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)
