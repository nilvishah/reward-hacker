import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

model_path = sys.argv[1]
name = Path(model_path).stem

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=f"results/videos/{name}",
    name_prefix=name,
    episode_trigger=lambda episode: True,
)
model = PPO.load(model_path)

for episode in range(2):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()
print(f"Saved videos to results/videos/{name}/")
