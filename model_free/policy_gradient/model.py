import tensorflow as tf
import numpy as np


class DiscretePolicyGradient:
    def __init__(self, obs_space, num_actions,
                 learning_rate=1e-3,
                 init_epsilon=0.1,
                 fin_epsilon=0,
                 exploration_frame=10000,
                 discount_factor=0.95,
                 mode='visual'
                 ):
        """
        :param obs_space: observation space. It should have three dimensions [width, height, channels] in visual model
        and one dimension in feature mode.
        :param num_actions: the number of actions. Scalar
        """
        self.obs_space = obs_space
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.fin_epsilon = fin_epsilon
        self.cur_epsilon = init_epsilon
        self.exploration_frame = exploration_frame
        self.discount_factor = discount_factor
        self.mode = mode
        self.replay_buffer = []
        self.model = tf.keras.models.Sequential(self.network())
        self.model.compile(loss=self.loss_function, optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        self.time_step = 0

    def loss_function(self, action_reward, policy):
        action, reward = tf.split(action_reward, [self.num_actions, 1], axis=1)
        batch_loss = tf.expand_dims(tf.reduce_sum(-tf.math.log(policy) * action, axis=1), axis=1) * reward
        loss = tf.reduce_sum(batch_loss)
        return loss

    def network(self):
        networks = [tf.keras.layers.Dense(100, input_dim=self.obs_space, activation='relu'),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(self.num_actions, activation='softmax')]
        return networks

    def get_action(self, obs):
        if np.random.uniform() > self.cur_epsilon:
            prob = self.model.predict(np.array([obs]))[0]
            action = np.random.choice(len(prob), p=prob)
        else:
            action = np.random.randint(self.num_actions)
        if self.time_step < self.exploration_frame:
            self.cur_epsilon -= (self.init_epsilon - self.fin_epsilon) / self.exploration_frame
        self.time_step += 1
        return action

    def memory_store(self, obs, action, reward, done):
        self.replay_buffer.append([obs, action, reward])
        if done:
            discount_reward = 0
            for i in reversed(range(len(self.replay_buffer))):
                self.replay_buffer[i][2] += self.discount_factor * discount_reward
                discount_reward = self.replay_buffer[i][2]
            self.train()
            self.replay_buffer = []

    def train(self):
        obs_batch = np.array([record[0] for record in self.replay_buffer])
        action_batch = [[1 if record[1] == i else 0 for i in range(self.num_actions)] for record in self.replay_buffer]
        reward_batch = [record[2] for record in self.replay_buffer]
        reward_batch = np.expand_dims(reward_batch, axis=1)
        action_reward = np.append(action_batch, reward_batch, axis=1)
        self.model.fit(obs_batch, action_reward)

