import gym
from gym import spaces
import numpy as np
import math

class EVChargingEnv(gym.Env):
    def __init__(self):
        super(EVChargingEnv, self).__init__()
        self.state_size = 4
        self.action_size = 3
        self.total_waiting_time = 0.0
        self.discount_per_hour = 0.005
        self.max_discount = 0.5
        
        # Define action and observation space
        # Actions: idle = 0, conventional charging = 1, fast charging = 2
        self.action_space = spaces.Discrete(self.action_size)
        
        # States: [SoC(t), hour, minute, charging mode]
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.availability_index = np.random.rand(24)
        
        # Initialize state
        self.state = None
        self.prev_action = None
        self.previously_idled = False
        self.reset()
        
    def reset(self):
        # Reset the state to an initial condition
        SoC = np.random.rand()
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        charging_mode = 0
        self.availability_index = np.random.rand(24)
        self.state = np.array([SoC, hour/24.0, minute/60.0, charging_mode])  # Example initial state
        self.prev_action = None
        self.previously_idled = False
        return self.state
    
    def calculate_discount_factor(self):
        if 1 - self.discount_per_hour * self.total_waiting_time > 1:
            return 1.0
        elif 1 - self.discount_per_hour * self.total_waiting_time < self.max_discount:
            return self.max_discount
        else:
            return 1 - self.discount_per_hour * self.total_waiting_time
    
    def step(self, action):
        SoC, hour, minute, charging_mode = self.state
        tp = 0.25   #time period
        P_c = 7     # conventional charging power
        P_f = 22    #fast charging power
        Q_b = 70      # battery capacity in KWh
        
        # Apply action and update state
        if action == 0 and not self.previously_idled:  # Idle
            charging_mode = 0
            self.total_waiting_time += 2 * tp
            self.previously_idled = True
        elif action == 1 or (self.previously_idled and action == 0):  # Conventional charging
            charging_mode = 1
            SoC += (tp * P_c)/Q_b
        elif action == 2:  # Fast charging
            charging_mode = 2
            SoC += (tp * P_f)/Q_b
        
        # Time progression
        minute *= 60
        hour *= 24
        if action == 0:
            minute += 30
        else:
            minute += 15

        if minute >= 60:
            minute = 0
            hour += 1
        if hour >= 24:
            hour = 0
        
        normalized_charging_mode = charging_mode / 2.0
        # Update state
        self.state = np.array([SoC, hour / 24.0, minute / 60.0, normalized_charging_mode])

        # Calculate reward
        C_t = [[0, 4.6], [1, 4.5], [2, 4.5], [3, 4.4], [4, 4.5], [5, 4.6], [6, 4.6], [7, 4.7],
                         [8, 4.9], [9, 4.9], [10, 5.0],[11, 4.9], [12, 5.0], [13, 6.9], [14, 6.6], [15, 7.2],
                         [16, 7.2], [17, 8.1], [18, 10.6], [19, 8.4], [20, 6.9], [21, 6.7], [22, 5.7], [23, 5.0]]   # time of use tariff
        
        Wi = np.random.normal(0.05, 0.0075)     #idle waiting cost
        Wc = 0.5 * Wi   #charging waiting cost
        k = self.calculate_discount_factor()

        Ux = np.interp(self.availability_index[int(hour)], [0, 0.25, 0.5, 0.75, 1], [-2, -1, 0, 1, 2])

        if charging_mode == 0 and not self.previously_idled:  # Idle
            r1 = -k * (tp * P_c * C_t[int(hour)][1] + tp * Wc + tp * Wi)
            r2 = Ux
            reward = r1 + r2 # Idle waiting cost

        elif charging_mode == 1:  # Conventional charging
            r1 = -(tp * P_c * C_t[int(hour)][1] + tp * Wc)
            r2 = Ux
            reward = r1 + r2  # Charging cost
        elif charging_mode == 2:  # Fast charging
            r1 = -(tp * P_f * C_t[int(hour)][1] + tp * Wc)
            r2 = Ux
            reward = r1 + r2  # Higher charging cost
        else:
            reward = 0
        
        # Penalty for overcharging
        if SoC > 1.0:
            SoC = 1.0
            reward -= 1.0
        
        done = SoC >= 1.0  # Done if fully charged

        return self.state, reward, done, {}
    
    def render(self, mode='human', close=False):
        SoC, hour, minute, charging_mode = self.state
        availability = self.availability_index[int(hour * 24)]
        print(f"State of Charge: {SoC:.2f}, Time: {int(hour * 24)}:{int(minute * 60)}, Charging Mode: {charging_mode * 2}, index: {availability}")

# Register the environment
gym.envs.registration.register( 
    id='EVCharging-v0',
    entry_point=EVChargingEnv,
)
