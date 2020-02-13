import gym
import numpy as np
import matplotlib.pyplot as plt
import model_free.policy_gradient.model as model
import tensorflow as tf

env = gym.make('CartPole-v0')

obs_space = 4
num_actions = 2
episodes = 1000
agent = model.DiscretePolicyGradient(obs_space, num_actions)


def play():
    score_list = []
    for i in range(episodes):
        score = 0
        obs = env.reset()
        file = open('CartPole.txt', 'a')
        while True:
            action = agent.choose_action(obs)
            nxt_obs, reward, done, _ = env.step(action)
            reward = -1.0 if done else 0.0
            agent.memory_store(obs, action, reward, done)
            score += 1
            if done:

                score_list.append(score)
                mean_score = np.mean(score_list[-10 if i >= 10 else 0:])
                file.write('%f \n' % score)
                print('episode:', i, 'score:', score, 'mean:', mean_score)
                break
            obs = nxt_obs
        if mean_score > 195:
            agent.model.save('CartPole-PG.h5')
            break


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    play()