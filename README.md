# RewardHacker

A pipeline where an LLM designs reward functions for a reinforcement learning agent, with a sandbox for running the generated code safely and a detector that flags reward hacking.

**Status:** in progress

## Environment

[Gymnasium LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/): the agent sees 8 numbers (position, velocity, tilt, spin, leg contact) and picks one of 4 actions each frame. An episode scoring 200+ counts as solved.

## Baseline

PPO (Stable-Baselines3, `MlpPolicy`) trained for 1M steps on the built-in reward, evaluated over 20 fixed seeds with no reward wrappers.

| Metric | Value |
|---|---|
| Mean score | 258.3 ± 44.3 |
| Solved rate (≥200) | 95% |
| Landed | 95% |
| Crashed | 5% |
| Timed out (hovering) | 0% |
| Mean episode length | 283 frames |
| Main engine frames (mean) | 105.5 |
| Side engine frames (mean) | 50.3 |

## Scripts

- `random_agent.py`: random actions, to see the untrained baseline
- `train_baseline.py`: train PPO on the built-in reward
- `watch_agent.py`: watch a trained agent fly
- `evaluate.py`: fixed evaluator used to judge every agent

## Roadmap

- [x] PPO baseline
- [x] Fixed evaluator
- [ ] Hand-written reward hack (hovering)
- [ ] LLM reward designer
- [ ] AST validator and Docker sandbox
- [ ] Reward-hacking detector
- [ ] Multi-seed experiments and results
