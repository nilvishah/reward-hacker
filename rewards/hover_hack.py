def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
    """Deliberately broken: +1 every frame the lander is upright and slow.

    Hovering in place for 1,000 frames earns far more than landing,
    so the agent can learn to never land.
    """
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
    upright = abs(angle) < 0.2
    slow = abs(vx) < 0.2 and abs(vy) < 0.2
    return 1.0 if upright and slow else 0.0
