import random
import os
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from gym.envs.registration import register

register(
    id='EVCharging-v0',
    entry_point='ev_charging_env:EVChargingEnv',
)

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Agent should explore
            action = random.randrange(self.action_size)
        else:
            # Agent should exploit
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])  # returns action
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        if os.path.exists(name):
            self.model.load_weights(name)
            print(f"file {name} loaded successfully")
            self.log_weights()  # Log weights after loading
        else:
            print(f"File {name} does not exist.")

    def log_weights(self):
        weights = self.model.get_weights()
        for i, weight in enumerate(weights):
            print(f"Layer {i//2} {'Weights' if i % 2 == 0 else 'Biases'}: {weight.shape}\n{weight}")
    
    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('EVCharging-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/ev-dqn.weights.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        cumulative_reward = 0
        action_counts = np.zeros(action_size)

        for time in range(1440):  # 1440 minutes in a day
            env.render()
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            print(f"Each reward: {reward}")
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}, Cumulative Reward: {cumulative_reward:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        print(f"Episode: {e} Action Distribution: {action_counts}")
        if e % 10 == 0:
            agent.save("./save/ev-dqn.weights.h5")
            print("Saved")
