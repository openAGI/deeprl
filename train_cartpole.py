import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from core.solver_pg import VanillaPolicyGradient, tf
import core.logger as log


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    DISPLAY = True
    DISPLAY_REWARD_THRESHOLD = 400
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = gym.make('CartPole-v0')
        env.seed(1)
        env = env.unwrapped

        log.setFileHandler('vanilla_policy.log')
        log.setVerbosity('1')
        log.info('Starting Policy Gradient optimization')
        log.info('Cartpole vanilla policy gradient training')
        log.info(str(env.action_space))
        log.info(str(env.observation_space))
        log.info(str(env.observation_space.high))
        log.info(str(env.observation_space.low))

        pg = VanillaPolicyGradient(
            n_actions=env.action_space.n,
            n_features=env.observation_space.shape[0],
            sess=sess,
            learning_rate=0.02,
            reward_decay=0.99,
        )

        for i_episode in tqdm(range(0, 3000), ncols=70, initial=0):

            observation = env.reset()

            while True:

                if DISPLAY:
                    env.render()

                action = pg.choose_action(observation)

                observation_, reward, done, info = env.step(action)

                pg.store_transition(observation, action, reward)

                if done:
                    ep_rs_sum = sum(pg.ep_rs)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                    if running_reward > DISPLAY_REWARD_THRESHOLD:
                        DISPLAY = True     # rendering
                    log.info('episode:%d, reward: %d' %
                             (i_episode, int(running_reward)))

                    vt = pg.learn()

                    if i_episode == 0:
                        plt.plot(vt)    # plot the episode vt
                        plt.xlabel('episode steps')
                        plt.ylabel('normalized state-action value')
                        plt.show()
                    break

                observation = observation_


if __name__ == '__main__':
    main()
