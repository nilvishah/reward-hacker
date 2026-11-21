import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate(model_path: str, episodes: int = 20, seed: int = 0) -> dict:
    """Score a trained model on the plain environment (built-in reward only)."""
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_path)

    scores, lengths, outcomes = [], [], []
    main_fires, side_fires = [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, steps, main, side = 0.0, 0, 0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            if action == 2:
                main += 1
            elif action in (1, 3):
                side += 1
            if terminated or truncated:
                # On the final frame the env sets reward to +100 (rested) or -100 (crash)
                if truncated:
                    outcomes.append("timeout")
                elif reward > 0:
                    outcomes.append("landed")
                else:
                    outcomes.append("crashed")
                break
        scores.append(total)
        lengths.append(steps)
        main_fires.append(main)
        side_fires.append(side)

    env.close()
    n = len(scores)
    return {
        "model": model_path,
        "episodes": n,
        "mean_score": round(float(np.mean(scores)), 1),
        "std_score": round(float(np.std(scores)), 1),
        "solved_rate": round(sum(s >= 200 for s in scores) / n, 2),
        "landed_rate": round(outcomes.count("landed") / n, 2),
        "crashed_rate": round(outcomes.count("crashed") / n, 2),
        "timeout_rate": round(outcomes.count("timeout") / n, 2),
        "mean_episode_length": round(float(np.mean(lengths)), 1),
        "mean_main_engine_frames": round(float(np.mean(main_fires)), 1),
        "mean_side_engine_frames": round(float(np.mean(side_fires)), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained LunarLander agent.")
    parser.add_argument("--model", required=True, help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = evaluate(args.model, args.episodes, args.seed)
    for key, value in results.items():
        print(f"{key:>26}: {value}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"eval_{Path(args.model).stem}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_file}")
