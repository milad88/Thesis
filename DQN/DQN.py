"""
Code based on code from exercise 5
"""

from Neural_Network import NeuralNetwork
import tensorflow as tf
import numpy as np
from Policies import *
import sys

from utility import ReplayBuffer, EpisodeStats


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, num_actions, name, tau=0.001):
        NeuralNetwork.__init__(self, num_actions, name)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars // 2].assign((var.value
                                                                    () * self.tau) + ((1 - self.tau) * tf_vars[
                idx + total_vars // 2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)


def q_learning(sess, env, approx, num_episodes, max_time_per_episode, action_space, num_actions, discount_factor=0.99,
               epsilon=0.1,
               batch_size=128, target=None, policy_fn=make_greedy_policy, stat=False, g_stat=[],writer=None, summary=None):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Implements the options of online learning or using experience replay and also
    target calculation by target networks, depending on the flags. You can reuse
    your Q-learning implementation of the last exercise.

    Args:
      env: OpenAI environment.
      approx: Action-Value function estimator
      num_episodes: Number of episodes to run for.
      max_time_per_episode: maximum number of time steps before episode is terminated
      discount_factor: gamma, discount factor of future rewards.
      epsilon: Chance to sample a random action. Float betwen 0 and 1.
      batch_size: Number of samples per batch.
      target: Slowly updated target network to calculate the targets. Ignored if None.

    Returns:
      An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    loss_episodes = []
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    buffer = ReplayBuffer()
    nTimes_actions = np.ones(num_actions)

    #train_writer = tf.summary.FileWriter('./log/loss', sess.graph)

    #merged = tf.summary.merge_all()

    for i_episode in range(num_episodes):
        loss = []
        # to decay epsilon in case we use epsilon greedy decay policy
        decay = np.exp(-1 / (num_episodes / 15) * i_episode)
        policy = policy_fn(
            approx, epsilon, num_actions, i_episode, nTimes_actions, decay)
        #greedy = make_greedy_policy(approx, epsilon, num_actions, i_episode, nTimes_actions, decay)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()

        observation = env.reset()
        done = False
        i = 0
        p_r = 0
        while i < max_time_per_episode:
            i += 1
            old_observation = observation

            action = policy(sess, observation)
            nTimes_actions[action] = nTimes_actions[action] + 1
            move = action_space[action]
            observation, reward, done, info = env.step([move])
            if reward == 1 or i < batch_size:

                buffer.add_transition(old_observation, move, observation, reward, done)
            s, a, ns, r, d = buffer.next_batch(batch_size)

            pred = target.predict(sess, ns)

            td = r + discount_factor * np.amax(pred, axis=1)

            # merge = tf.summary.merge_all()
            estimator_prop = approx.predict(sess, s)
            #_, rewards, _, _ = env.step(np.take(action_space, np.argmax(estimator_prop, axis=1)))
            l = approx.update(sess, s, a, td, estimator_prop, env, action_space, summary)
            #loss.append(l[0])

            #writer.add_summary(l[1], i_episode)
            #train_writer.flush()

            stats.episode_rewards[i_episode] += reward

            if stat > 0:
                p_r += reward

            if i > batch_size and i % batch_size == 0:
                target.update(sess)

        if stat > 0:
            g_r = 0
            observation = env.reset()
            done = False
            i = 0
            while not done and i < max_time_per_episode:
                i += 1
                action = policy(sess, observation)
                move = action_space[action]
                observation, reward, done, info = env.step([move])
                g_r += reward
            g_stat.append(int(np.round(g_r)))

        loss_episodes.append(sum(loss))
        stats.episode_lengths[i_episode] = i

#    approx.save_model(sess)
    return stats, loss_episodes
