GAMMA = 0.99  # must match PPO's discount factor (SB3 default)


def _potential(obs) -> float:
    """Higher when the lander is upright and slow (0 is perfect)."""
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
    return -(abs(angle) + abs(vx) + abs(vy))


def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
    """Potential-based version of hover_fixed (Ng et al., 1999).

    bonus = gamma * potential(next) - potential(prev), with potential = 0 at the
    end of an attempt. The bonus guides learning but provably cannot change
    which behavior is best, so slamming into the ground earns nothing extra.
    """
    next_potential = 0.0 if terminated else _potential(obs)
    shaping = 10.0 * (GAMMA * next_potential - _potential(prev_obs))
    time_cost = 0.05
    return shaping - time_cost
