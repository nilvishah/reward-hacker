import math

def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    if action in (1, 3):
        angle = obs[4]
        angvel = obs[5]
        if abs(angle) < 0.15 and abs(angvel) < 0.3:
            return -0.1
    return 0.0
