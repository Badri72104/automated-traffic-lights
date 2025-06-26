from traffic_env import TrafficEnv
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import time
import cv2
env = TrafficEnv()
model = DQN.load("traffic_rl_model")

obs = env.reset()
total_reward = 0
rewards = []

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    total_reward += reward
    if done:
        break

print("Total reward:", total_reward)

Plot
plt.plot(rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("Traffic Controller Performance")
plt.show()



for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done,info = env.step(action)
    
    env.render()  # 👈 Visual display
    time.sleep(0.2)  # Optional: slow it down

cv2.destroyAllWindows()
