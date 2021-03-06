import sys
from Policies import *
from utility import *
from Critic_Network_DDPG import *
from Actor_Network_DDPG import *
import tensorflow as tf
from pendulum import PendulumEnv
from continuous_moutain_car import Continuous_MountainCarEnv
if __name__ == "__main__":
    print("start")
    env = Continuous_MountainCarEnv()
    action_space = env.action_space
    # num_actions = len(action_space)
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    state_dim = env.observation_space.shape[0]
    batch_size = 32
    learning_rate = 0.001
    discount_factor = 0.99
    num_episodes = 5000
    len_episode = 200
    epsilon = 0.1
    load = False
    if not load:

        policies = [make_epsilon_greedy_decay_policy, make_epsilon_greedy_policy, make_ucb_policy]

        g_stat = []
        policy = policies[2]
        fname = './DDPG/log/g_.csv'

        with tf.Session() as sess:

            name = policy.__name__  # No reason to save more than one from each policy atm
            with tf.name_scope("actor"):
                actor = Actor_Net(action_dim, "actor", action_bound, state_dim,
                                  learning_rate=learning_rate)

            with tf.name_scope("critic"):
                critic = Critic_Net(action_dim, "critic", action_bound, state_dim,
                                    learning_rate=learning_rate)

            with tf.name_scope("actor_target"):
                target_actor = Actor_Target_Network(action_dim, "actor_target", action_bound, state_dim, actor,
                                                    learning_rate=learning_rate)
            with tf.name_scope("critic_target"):
                target_critic = Critic_Target_Network(action_dim, "critic_target", action_bound, state_dim, critic,
                                                      learning_rate=learning_rate)

            writer = tf.summary.FileWriter('./DDPG/log/DDPG_loss', sess.graph)
            summ_critic_loss = tf.summary.scalar('loss_critic', critic.get_loss())

            sess.run(tf.global_variables_initializer())
            """
            DDPG
            """
            loss_episodes = []
            stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
            buffer = ReplayBuffer()
            # nTimes_actions = np.ones(num_actions)
            action = 0
            a_grads = 0
            loss_critic = [0]

            for i_episode in range(num_episodes):

                loss = []
                # to decay epsilon in case we use epsilon greedy decay policy
                decay = np.exp(-1 / (num_episodes / 15) * i_episode)

                #greedy = make_greedy_policy(actor, epsilon, num_actions, i_episode, nTimes_actions, decay)

                # Print out which episode we're on, useful for debugging.
                # Also print reward for last episode
                last_reward = stats.episode_rewards[i_episode - 1]
                grad = a_grads
                if type(a_grads) == list:
                    grad = a_grads[0][0]

                print("\rEpisode {}/{} ({}) action {} gradient {} critic loss {}".format(i_episode + 1, num_episodes, last_reward, action, grad, loss_critic[0]))
                sys.stdout.flush()

                done = False
                i = 0
                g_r = 0

                observation = env.reset()

                while not done and i < len_episode:
                    loss = []
                    i += 1
                    old_observation = observation

                    action = actor.predict(sess, [observation])
                    env.render()
                    observation, reward, done, info = env.step(action)
                    #if i < 5 or reward > 0:
                    buffer.add_transition(old_observation, action[0], observation, reward, done)
                    s, a, ns, r, d = buffer.next_batch(batch_size)
                    print("observation**************")
                    print(s)
                    print()
                    print()
                    pred_actions = target_actor.predict(sess, ns)

                    q_values = target_critic.predict(sess, ns, pred_actions)
                    # y = r + (discount_factor * q_values)
                    y = r + np.multiply(discount_factor, np.ravel(q_values))
                    y = np.reshape(y, [len(y), 1])

                    g_r += reward
                    g_stat.append(int(np.round(g_r)))
                    actor_outs = actor.predict(sess, s)
                    # actor_outs = np.reshape(actor_outs, [len(actor_outs), 1])
                    loss_critic = critic.update(sess, s, a, y, summ_critic_loss)

                    loss.append(loss_critic[0])
                    #target_critic_out = target_critic.predict(sess,s,a)
                    a_grads = critic.action_gradients(sess, s, actor_outs)

                    sys.stdout.flush()
                    #actor.update(sess, s, target_critic_out, None)
                    actor.update(sess, s, a_grads, None)

                    stats.episode_rewards[i_episode] += reward

                    target_critic.update(sess)
                    target_actor.update(sess)

                    g_stat.append(int(np.round(g_r)))
                    print("net params critic")
                    print(sess.run(critic.net_params[0][0]))
                l = sum(loss)
                summ_critic_loss = tf.Summary(value=[tf.Summary.Value(tag="loss_critic",
                                                                      simple_value=l)])
                writer.add_summary(summ_critic_loss, i_episode)

                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_rewards",
                                                                     simple_value=stats.episode_rewards[i_episode])]), i_episode)


                writer.flush()
                loss_episodes.append(l)

                stats.episode_lengths[i_episode] = i

            plot_episode_stats(stats)
            plot_stats(loss_episodes)
            # return stats, loss_episodes
