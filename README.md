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

## Reward hacking: break and fix

**Broken reward** (`rewards/hover_hack.py`): +1 per frame for being upright and slow. Hovering for 1,000 frames pays far more than landing, so the agent learned to never land.

**Fixed reward** (`rewards/hover_fixed.py`): pays for *improving* stability between frames instead of *being* stable, plus a 0.05 per-frame time cost. Hovering in place now earns nothing and costs time.

All agents evaluated on the built-in reward only, 20 fixed seeds.

| | Baseline | Hover hack | Fixed |
|---|---|---|---|
| Mean score | 258.3 ± 44.3 | -10.6 ± 22.2 | 249.6 ± 23.0 |
| Landed | 95% | 0% | 100% |
| Timed out | 0% | 100% | 0% |
| Mean episode length | 283 | 1,000 | 344 |

## Scripts

- `random_agent.py`: random actions, to see the untrained baseline
- `train_baseline.py`: train PPO on the built-in reward
- `train_with_reward.py`: train PPO with any reward from `rewards/`
- `reward_wrapper.py`: adds a custom bonus on top of the built-in reward
- `watch_agent.py`: watch a trained agent fly
- `record_video.py`: record episodes as video
- `evaluate.py`: fixed evaluator used to judge every agent
- `rewards/`: reward functions (`hover_hack.py` is broken on purpose, `hover_fixed.py` is the fix)

## Roadmap

- [x] PPO baseline
- [x] Fixed evaluator
- [x] Hand-written reward hack (hovering)
- [ ] LLM reward designer
- [ ] AST validator and Docker sandbox
- [ ] Reward-hacking detector
- [ ] Multi-seed experiments and results
