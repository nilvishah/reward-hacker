def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    b = -0.1
    if terminated and base_reward >= 99.0:
        b += 10.0
    return float(max(-10.0, min(10.0, b)))
