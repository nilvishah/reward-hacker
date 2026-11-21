import importlib

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Adds a custom bonus on top of the built-in LunarLander reward."""

    def __init__(self, env: gym.Env, bonus_fn):
        super().__init__(env)
        self.bonus_fn = bonus_fn
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = float(
            self.bonus_fn(obs, self.prev_obs, int(action), reward, terminated, truncated)
        )
        self.prev_obs = obs
        info["base_reward"] = reward
        info["bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info


def load_bonus_fn(name: str):
    """Load compute_bonus from rewards/<name>.py."""
    module = importlib.import_module(f"rewards.{name}")
    return module.compute_bonus
