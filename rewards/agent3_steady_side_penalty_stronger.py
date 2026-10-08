import math

def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    if action == 1 or action == 3:
        angle = abs(obs[4])
        angvel = abs(obs[5])
        if angle < 0.25 and angvel < 0.5:
            return -0.5
    return 0.0
