import numpy as np
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])
batch_size = 32

def plot_stats(stats):
    fig11 = plt.figure(figsize=(10, 5))

    plt.plot(np.ravel(stats))
    plt.xlabel("Episode")
    plt.ylabel("loss per episode")
    plt.show(fig11)


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.step = -1

    def add_transition(self, state, action, next_state, reward, done):

        if np.array(state).shape == (3,1):
            state = list(itertools.chain.from_iterable(state))

        if np.array(next_state).shape == (3, 1):
            next_state = list(itertools.chain.from_iterable(next_state))

        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)


    def next_batch(self, batch_size):

        #self._data.states[-1] =np.ravel(self._data.states[-1])
        #print(self._data.states)
        self.size = batch_size
        if batch_size >= len(self._data.states):
            return np.array(self._data.states), np.array(self._data.actions), np.array(self._data.next_states), np.array(self._data.rewards), np.array(self._data.dones)
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

#        if self.transition_size() > 5*batch_size:
 #           self._data.states = self._data.states[-5*batch_size:]
  #          self._data.actions = self._data.actions[-5*batch_size:]
   #         self._data.next_states = self._data.next_states[-5*batch_size:]
     #       self._data.dones = self._data.dones[-5*batch_size:]
      #      self._data.rewards = self._data.rewards[-5*batch_size:]


    def transition_size(self):
        return len(self._data.states)
