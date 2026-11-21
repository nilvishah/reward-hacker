import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from reward_wrapper import CustomRewardWrapper, load_bonus_fn

parser = argparse.ArgumentParser(description="Train PPO with a custom reward.")
parser.add_argument("--reward", required=True, help="Name of a file in rewards/, e.g. hover_hack")
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = CustomRewardWrapper(gym.make("LunarLander-v3"), load_bonus_fn(args.reward))
model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
model.learn(total_timesteps=args.steps)

Path("models").mkdir(exist_ok=True)
out = f"models/ppo_{args.reward}_seed{args.seed}"
model.save(out)
env.close()
print(f"Saved model to {out}.zip")
