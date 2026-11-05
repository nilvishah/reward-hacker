def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    b = 0.0
    if action == 1 or action == 3:
        if abs(obs[4]) < 0.2 and abs(obs[5]) < 0.4:
            b -= 0.2
    if obs[6] > 0.5 and obs[7] > 0.5 and action == 0:
        b += 1.0
    return float(b)
