import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import gym
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from gym.envs.registration import register

register(
    id='EVCharging-v0',
    entry_point='ev_charging_env:EVChargingEnv',
)

def create_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Define state size and action size according to your environment
state_size = 4  # Example state size; replace with actual
action_size = 3  # Example action size; replace with actual

# Load the trained model weights
model = create_model(state_size, action_size)
model.load_weights('./save/ev-dqn.weights.h5')

# Set up the simulation environment
env = gym.make('EVCharging-v0')

C_t = [[0, 4.6], [1, 4.5], [2, 4.5], [3, 4.4], [4, 4.5], [5, 4.6], [6, 4.6], [7, 4.7],
       [8, 4.9], [9, 4.9], [10, 5.0], [11, 4.9], [12, 5.0], [13, 6.9], [14, 6.6], [15, 7.2],
       [16, 7.2], [17, 8.1], [18, 10.6], [19, 8.4], [20, 6.9], [21, 6.7], [22, 5.7], [23, 5.0]] 

# Define the action selection policy
def select_action(state):
    state = state.reshape((1, -1))
    q_values = model.predict(state)
    action = np.argmax(q_values[0])
    print(q_values)
    return action

# Run a single episode to collect and plot data
episode = 1
max_steps = 100

battery_levels = []
electricity_prices = []
power = []
time_steps = []

state = env.reset()
total_reward = 0
t = 0
state = np.array([0.30, 0.4334323, 0.898237, 0.0])

for step in range(max_steps):
    # Assuming state contains battery level and electricity price at specific indices
    env.render()
    battery_level = state[0]  # Adjust index as needed
    index = int(state[1]*24)
    electricity_price = C_t[index][1]  # Adjust index as needed
    time_step = t * 15
    
    battery_levels.append(battery_level)
    electricity_prices.append(electricity_price)
    time_steps.append(time_step)

    action = select_action(state)
    if action == 0:
        power.append(0)
        power.append(7)
        time_steps.append(time_step + 15)
        electricity_prices.append(electricity_price)
        battery_levels.append(battery_level)
        t += 1
    elif action == 1:
        power.append(7)
    else:
        power.append(22)

    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

    t += 1

    if done:
        break

print(f'Episode {episode}, Total Reward: {total_reward}')
print(f'Power: {power}')
print(f'Time steps: {time_steps}')
print(f'Electricity prices: {electricity_prices}')
print(f'Battery levels: {battery_levels}')

# Ensure the lengths of time_steps and electricity_price match
if len(time_steps) != len(electricity_prices):
    raise ValueError("Length of time_steps and electricity_prices must be the same")

# Ensure the lengths of time_steps and battery_levels match
if len(time_steps) != len(battery_levels):
    raise ValueError("Length of time_steps and battery_levels must be the same")

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the bar chart for charging power
bars = ax1.bar(time_steps, power, width=12, color='blue', label='Charging Power (kW)')  # Adjust width to make bars span 15 mins

# Adding labels and title
ax1.set_xlabel('Time Period (minutes)')
ax1.set_ylabel('Charging Power (kW)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(np.arange(0, max(time_steps) + 1, 30))  # Setting x-ticks to show every 30 minutes
ax1.set_xticklabels(np.arange(0, (max(time_steps) // 30) + 1) * 0.5, rotation=45)

# Adding the battery level on top of each bar
for bar, battery_level in zip(bars, battery_levels):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(battery_level, 2), ha='center', va='bottom', color='blue')

# Create a second y-axis for electricity price
ax2 = ax1.twinx()
ax2.plot(time_steps, electricity_prices, color='red', marker='o', linestyle='-', linewidth=2, label='Electricity Price ($/kWh)')
ax2.set_ylabel('Electricity Price ($/kWh)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Adding title and grid
plt.title('Charging Power and Electricity Price Over Time')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Adding legend
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Display the plot
plt.show()
