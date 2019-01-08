import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected

action_bound = 2.0


class Actor_Net():
    def __init__(self, num_actions, action_dim, name, action_bound, state_dim, learning_rate=0.01, batch_size=32):
        # super().__init__(num_actions, name)
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.name = name
        self._build_model(num_actions)

    def _build_model(self, num_actions):

        # self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        self.inp = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)

        self.inpW = tf.Variable(tf.random_uniform([self.state_dim, 16], -0.5, 0.5))
        self.inpB = tf.Variable(tf.constant(0.1, shape=[16]))
        self.h1 = tf.nn.relu(tf.matmul(self.inp, self.inpW) + self.inpB)

        self.h2W = tf.Variable(tf.random_uniform([16, 32], -0.5, 0.5))
        self.h2B = tf.Variable(tf.constant(0.1, shape=[32]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.h2W) + self.h2B)

        self.h3W = tf.Variable(tf.random_uniform([32, 16], -0.5, 0.5))
        self.h3B = tf.Variable(tf.constant(0.1, shape=[16]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.h3W) + self.h3B)

        self.h4W = tf.Variable(tf.random_uniform([16, self.action_dim], -0.5, 0.5))

        self.outB = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))

        self.outputs = tf.nn.tanh(tf.matmul(self.h3, self.h4W) + self.outB)
        self.scaled_outputs = tf.scalar_mul(action_bound, self.outputs)

        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.actor_gradients = tf.gradients(ys=self.outputs, xs=self.net_params, grad_ys=-self.action_gradients)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.net_params))

        # Use the same stdev for all inputs.
        # sy_logstd_a = tf.get_variable("log_Std1", [ac_dim], initializer=tf.zeros_initializer())  # Variance

        # batch of actions taken by the policy, used for policy gradient computation



        # self.sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        # Now, need to compute the logprob for each action taken.
        # self.action_dist = tf.contrib.distributions.Normal(loc=self.sy_mean_na, scale=tf.exp(sy_logstd_a), validate_args=True)
        # sy_logprob_n is in [batch_size, ac_dim] shape.
        # self.sy_logprob_n = self.action_dist.log_prob(self.sy_ac_na)

        # Now, need to sample an action based on input. This should be a 1-D vector
        # with ac_dim float in it.
        # sy_sampled_ac = action_dist.sample()[0]


        # self.adv = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)  # advantage function estimate

        # We do tf.reduce_mean on sy_logprob_n here, as it's shape is [batch_size,
        # ac_dim]. Not sure what's the best way to deal with ac_dim -- but pendulum's
        # ac_dim is 1, so using reduce_mean here is fine.

        # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")
        # self.loss = - tf.reduce_mean(tf.reduce_mean(self.sy_logprob_n, 1))

        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # self.loss = tf.reduce_mean(tf.square(self.y_ - self.out))

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
        if states[-1].shape == (1,):
            if len(states) == 3:
                states = np.array(np.ravel(states))
            else:
                states = states[:-3]

        states = np.atleast_2d(states)
        np.reshape(states, [len(states), 3])
        # print(states.shape)
        feed = {self.inp: states}
        prediction = sess.run(self.scaled_outputs, feed)
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

        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 3])
        #hists = []
        #for ix, grad in enumerate(self.actor_gradients):
            #self.actor_gradients[ix] = grad / float(self.batch_size)
            #hists.append(tf.summary.histogram(str(ix)+'/gradient', grad))
        sess.run((self.optimize), feed_dict={self.inp: states, self.action_gradients: grads[0]})



class Actor_Target_Network(Actor_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, num_actions, action_dim, name, action_bound, state_dim, learning_rate=0.001, tau=0.001):
        super().__init__(num_actions, action_dim, name, action_bound, state_dim, learning_rate)
        # self._build_model( num_actions, action_dim, name, action_bound, state_dim)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):

        actor_vars = tf.trainable_variables()#"actor"
        target_vars = tf.trainable_variables()#"actor_target"

        # print(tf_vars)

        # total_vars = len(tf_vars)

        op_holder = []
        for idx, var in enumerate(target_vars):  # // is to retun un integer
            op_holder.append(var.assign(
                (actor_vars[idx].value() * self.tau) + ((1 - self.tau) * var.value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)
