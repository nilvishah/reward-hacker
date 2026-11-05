def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):
    b = 0.0
    if action == 1 or action == 3:
        b -= 0.03
    if terminated:
        if base_reward >= 99.0:
            b += 10.0
        elif base_reward <= -99.0:
            b -= 10.0
    return float(max(-10.0, min(10.0, b)))
