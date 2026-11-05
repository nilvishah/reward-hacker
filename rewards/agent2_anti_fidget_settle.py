def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    b = 0.0
    if action == 1 or action == 3:
        if abs(obs[4]) < 0.15 and abs(obs[5]) < 0.3:
            b -= 0.1
    if obs[6] > 0.5 and obs[7] > 0.5 and action == 0:
        b += 1.0
    return float(b)
