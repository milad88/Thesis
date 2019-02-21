import os
from DQN import *
from Policies import *
from utility import *

if "../../" not in sys.path:
    sys.path.append("../../")

from lib.envs.pendulum import PendulumEnv

if __name__ == "__main__":
    print("start")

    action_space = np.arange(-2, 2.01, 0.01)
    num_actions = len(action_space)

    approx = None
    target = None
    load = False

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    statrun = True

    # # Choose one.
    # stats = q_learning(sess, env, approx, 3000, 10.000)  # env, approx, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1, use_experience_replay=False, batch_size=128, target=None
    if not load:
        if statrun:
            n_ep = 1000
            l_ep = 100
            batch_size = 16
            policies = [make_epsilon_greedy_decay_policy, make_epsilon_greedy_policy, make_ucb_policy]

            g_stat = []
            policy = policies[2]
            fname = './log/g_' + str(num_actions) + '.csv'
            with tf.Session(config=config) as sess:

                train_writer = tf.summary.FileWriter('./log/loss', sess.graph)

                name = policy.__name__  # No reason to save more than one from each policy atm

                env = PendulumEnv()
                env.reset()

                approx = NeuralNetwork(num_actions, "approx")
                target = TargetNetwork(num_actions, "target")

#                summ_loss = tf.summary.scalar('loss', approx.loss)
 #               tf.summary.histogram('loss_hist', approx.loss)

                sess.run(tf.global_variables_initializer())
                stats, loss = q_learning(sess, env, approx, n_ep, l_ep, action_space, num_actions,
                                         batch_size=batch_size,
                                         target=target,
                                         policy_fn=policy, stat=True,
                                         g_stat=g_stat,
                                         writer=train_writer,
                                         summary=None)
                train_writer.flush()

            g_stat = np.array([g_stat])
            if not os.path.exists(fname):
                with open(fname, 'wb') as abc:
                    np.savetxt(abc, g_stat, fmt='%i', delimiter=",")
            else:
                with open(fname, 'ab') as abc:
                    np.savetxt(abc, g_stat, fmt='%i', delimiter=",")

        else:
            with tf.Session(config=config) as sess:
                approx = NeuralNetwork(num_actions, "custom")
                target = TargetNetwork(num_actions, "custom")
                sess.run(tf.global_variables_initializer())
                stats, loss = q_learning(sess, env, approx, 1000, 100, action_space, num_actions, batch_size=128,
                                         target=target,
                                         policy_fn=make_ucb_policy)

        plot_episode_stats(stats)
        plot_stats(loss)
    else:
        with tf.Session(config=config) as sess:
            approx = NeuralNetwork(num_actions, "make_ucb_policy")
            sess.run(tf.global_variables_initializer())
            approx.load_model(sess)
            for _ in range(10):
                state = env.reset()
                for _ in range(100):
                    env.render()
                    action = approx.predict(sess, [state])
                    action = action_space[np.argmax(action)]
                    action = [action]
                    print(action)
                    state, _, done, _ = env.step(action)
                    if done:
                        break