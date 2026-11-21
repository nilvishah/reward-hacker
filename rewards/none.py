def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
    """Baseline: no bonus, built-in reward only."""
    return 0.0
