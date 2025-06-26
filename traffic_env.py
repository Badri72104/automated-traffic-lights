import gym
from gym import spaces
import numpy as np
import cv2
class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # 2 actions: 0 = keep lights, 1 = switch
        self.action_space = spaces.Discrete(2)

        # Observation: [N, S, E, W, light_phase]
        self.observation_space = spaces.Box(low=0, high=50, shape=(5,), dtype=np.int32)

        self.reset()

    def reset(self):
        self.traffic = np.random.randint(0, 10, size=4)  # Random cars at NSWE
        self.phase = 0  # 0 = NS green, 1 = EW green
        self.time = 0
        return self._get_obs()

    def _get_obs(self):
        return np.append(self.traffic, self.phase)

    def step(self, action):
        if action == 1:
            self.phase = 1 - self.phase  # Toggle lights

        passed = np.zeros(4, dtype=int)

        if self.phase == 0:
            passed[0] = min(3, self.traffic[0])  # North
            passed[1] = min(3, self.traffic[1])  # South
        else:
            passed[2] = min(3, self.traffic[2])  # East
            passed[3] = min(3, self.traffic[3])  # West

        self.traffic -= passed
        self.traffic += np.random.randint(0, 3, size=4,dtype=int)  # New cars arrive

        self.time += 1
        reward = -sum(self.traffic)  # Penalize queues
        done = self.time >= 100  # End after 100 steps

        return self._get_obs(), reward, done, {}


# Inside the TrafficEnv class
    def render(self, mode='human'):
        img = np.zeros((400, 400, 3), dtype=np.uint8)

        # Define positions for 4 directions (N, E, S, W)
        directions = ['North', 'East', 'South', 'West']
        positions = {
            'North': (150, 50),
            'East': (250, 150),
            'South': (150, 250),
            'West': (50, 150)
        }

        # Draw rectangles for lanes
        for i, dir in enumerate(directions):
            pos = positions[dir]
            count = int(self.traffic[i])

            # Color: More traffic = more red
            intensity = min(255, count * 25)
            color = (0, 0, 255 - intensity)  # Blue to red

            # Draw traffic block
            cv2.rectangle(img, pos, (pos[0] + 100, pos[1] + 100), color, -1)
            
            # Add text (vehicle count)
            cv2.putText(img, f"{dir}: {count}", (pos[0], pos[1] + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Draw intersection box
        cv2.rectangle(img, (150, 150), (250, 250), (255, 255, 255), 2)

        # Show image
        cv2.imshow("Traffic Simulation", img)
        cv2.waitKey(100)  # Refresh every frame
