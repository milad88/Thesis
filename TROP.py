import sys
from pendulum import PendulumEnv
from Actor_Network_TRPO import *
from Critic_Network_TRPO import *
from utility import *


if __name__ == "__main__":
    print("start")
    env = PendulumEnv()
    action_space = np.arange(-2, 2.01, 0.01)
    num_actions = len(action_space)
    action_dim = 1
    action_bound = env.action_space.high
    state_dim = 3
    batch_size = 32
    learning_rate = 0.001
    delta = 0.01
    discount_factor = 0.99
    num_episodes = 10
    len_episode = 100
    epsilon = 0.1
    load = False
    if not load:

        g_stat = []

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 8
        config.gpu_options.per_process_gpu_memory_fraction = 0.33

        with tf.Session(config=config)as sess:

            with tf.name_scope("actor"):
                actor = Actor_Net(num_actions, action_dim, "actor", action_bound, state_dim,
                                  learning_rate=learning_rate)

            with tf.name_scope("critic"):
                critic = Critic_Net(num_actions, action_dim, "critic", action_bound, state_dim,
                                    learning_rate=learning_rate)


            writer = tf.summary.FileWriter('./TRPO/TRPO_loss', sess.graph)
            summ_critic_loss = tf.summary.scalar('loss_critic', critic.get_loss())

            sess.run(tf.global_variables_initializer())
            g = sess.graph
            """
            Trpo
            """
            loss_episodes = []
            stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
            buffer = ReplayBuffer()
            nTimes_actions = np.ones(num_actions)

            for i_episode in range(num_episodes):
                loss = []
                # Also print reward for last episode
                last_reward = stats.episode_rewards[i_episode - 1]
                print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
                sys.stdout.flush()

                done = False
                i = 0
                g_r = 0

                observation = env.reset()

                while not done and i < len_episode:

                    first = True
                    if i != 0:
                        first = False
                        sess.graph.clear_collection("theta_sff")
                    loss = []
                    i += 1
                    old_observation = observation
                    action = np.take(actor.predict(sess, observation), [0])

                    env.render()
                    observation, reward, done, info = env.step([action])

                    buffer.add_transition(old_observation, action, observation, reward, done)
                    s, a, ns, r, d = buffer.next_batch(batch_size)

                    pred_actions = actor.predict(sess, ns)

                    q_values = critic.predict(sess, ns, pred_actions)

                    r = np.reshape(r,[-1,1])
                    y = q_values - r

                    g_r += reward
                    g_stat.append(int(np.round(g_r)))

                    loss_critic = critic.update(sess, s, a, y, summ_critic_loss)

                    loss.append(loss_critic)

                    sys.stdout.flush()

                    actor.update(sess, s, a, y, None, first)

                    stats.episode_rewards[i_episode] += reward

                    g_stat.append(int(np.round(g_r)))

                    #sess.graph.as_default()

                l = sum(loss)
                summ_critic_loss = tf.Summary(value=[tf.Summary.Value(tag="loss_critic",
                                                                      simple_value=l)])
                writer.add_summary(summ_critic_loss, i_episode)

                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_rewards",
                                                                     simple_value=stats.episode_rewards[i_episode])]), i_episode)


                writer.flush()
                loss_episodes.append(l)

                stats.episode_lengths[i_episode] = i

                gc.collect()
                tf.keras.backend.clear_session()
                #tf.reset_default_graph()

                #tf.get_default_graph().finalize()

            plot_episode_stats(stats)
            plot_stats(loss_episodes)
            # return stats, loss_episodes
