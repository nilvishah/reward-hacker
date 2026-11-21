def _stability(obs) -> float:
    """Higher when the lander is upright and slow (0 is perfect)."""
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
    return -(abs(angle) + abs(vx) + abs(vy))


def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
    """Fixed version of hover_hack.

    Pays for improving stability (change between frames), not for being stable,
    so hovering in place earns nothing. A small per-frame cost makes waiting expensive.
    """
    improvement = _stability(obs) - _stability(prev_obs)
    time_cost = 0.05
    return 10.0 * improvement - time_cost
