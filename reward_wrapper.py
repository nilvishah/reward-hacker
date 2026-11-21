import importlib

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Adds a custom bonus on top of the built-in LunarLander reward."""

    def __init__(self, env: gym.Env, bonus_fn):
        super().__init__(env)
        self.bonus_fn = bonus_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = float(self.bonus_fn(obs, int(action), reward, terminated, truncated))
        info["base_reward"] = reward
        info["bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info


def load_bonus_fn(name: str):
    """Load compute_bonus from rewards/<name>.py."""
    module = importlib.import_module(f"rewards.{name}")
    return module.compute_bonus
