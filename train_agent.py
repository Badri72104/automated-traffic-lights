from traffic_env import TrafficEnv
from stable_baselines3 import DQN

# Register custom environment
env = TrafficEnv()

# Create RL model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save model
model.save("traffic_rl_model")
