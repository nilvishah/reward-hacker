# IDEA: Penalize only unsafe descent speed, tilt and drift near the ground, so hovering earns nothing and the agent is steered toward soft touchdowns.
def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
    x, y, vx, vy, angle, angvel, left_leg, right_leg = obs[:8]
    bonus = 0.0

    # Speed limit on descent that tightens as the lander gets closer to the ground
    safe_down = 0.15 + 0.7 * max(y, 0.0)
    excess = -vy - safe_down
    if excess > 0.0:
        bonus -= 0.3 * min(excess, 1.5)

    # Near the ground, discourage tilt and sideways speed (only penalties, nothing to farm)
    if y < 0.4:
        tilt_excess = abs(angle) - 0.15
        if tilt_excess > 0.0:
            bonus -= 0.3 * min(tilt_excess, 1.0)
        drift_excess = abs(vx) - 0.3
        if drift_excess > 0.0:
            bonus -= 0.2 * min(drift_excess, 1.0)

    return float(bonus)
