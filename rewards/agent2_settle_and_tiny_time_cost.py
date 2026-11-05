def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    b = -0.03
    if obs[6] > 0.5 and obs[7] > 0.5 and action == 0:
        b += 1.0
    return float(b)
