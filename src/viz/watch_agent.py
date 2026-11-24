import gymnasium as gym
from stable_baselines3 import PPO
import sys 

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load(sys.argv[1] if len(sys.argv) > 1 else "ppo_lunar_baseline")

for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: total reward = {round(total_reward, 1)}")

env.close()
