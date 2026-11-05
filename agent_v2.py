import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from agent import SYSTEM, TOOLS, detect_hacking
from evaluate import evaluate
from validator import validate

load_dotenv()
MODEL = "claude-sonnet-5-5"
MAX_EXPERIMENTS = 4
SEEDS = [0, 1, 2]
LOG = Path("results/agent_log_v2.md")

GOAL = f"""Goal: find a bonus that beats the no-bonus baseline. You have {MAX_EXPERIMENTS} experiments.
Each experiment now trains on {len(SEEDS)} seeds, and results are averaged across them.

Results so far (3 seeds each unless noted), judged on the built-in reward only:
- no bonus: score 222.5 (std 20.0 across seeds), landed 88%, timed out 2%
- +1 per frame while upright and slow: score -30.5, landed 3%, timed out 83% (learned to hover)
- 10 x improvement in -(|angle|+|vx|+|vy|) minus 0.05/frame: score 182.2, landed 78%
- potential-based version of that: score 168.4, landed 78% (too slow, burns fuel)

A previous agent tested ideas on ONE seed each and picked the best:
- +10 landing bonus, -0.1 per frame: 273.5 on seed 0, but only 194.2 (std 58.8) across 3 seeds,
  landed 88%. One seed took 678 frames per attempt and burned lots of fuel.
- On single seeds: -0.02/frame scored 239.1, -0.05/frame 265.7, -0.15/frame 231.2.
Lesson: single-seed results were mostly noise. Look at the per-seed scores, not just the mean."""


def log(text: str):
    print(text)
    with LOG.open("a") as f:
        f.write(text + "\n")


def aggregate(runs: list) -> dict:
    keys = ["mean_score", "landed_rate", "crashed_rate", "timeout_rate",
            "mean_episode_length", "mean_main_engine_frames", "mean_side_engine_frames"]
    out = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in keys}
    out["score_std_across_seeds"] = round(float(np.std([r["mean_score"] for r in runs])), 1)
    out["per_seed_scores"] = [r["mean_score"] for r in runs]
    return out


def run_experiment(name: str, idea: str, code: str, number: int) -> dict:
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:40]
    reward = f"agent2_{name}"
    log(f"\n## Experiment {number}: {name}")
    log(f"💡 Idea: {idea}")

    problems = validate(code)
    if problems:
        log("🛡️ Checker: ❌ BLOCKED")
        for p in problems:
            log(f"   - {p}")
        return {"status": "blocked", "problems": problems}
    log("🛡️ Checker: ✅ passed")

    Path(f"rewards/{reward}.py").write_text(code)
    log(f"🚀 Training {len(SEEDS)} seeds in parallel...")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = [
        subprocess.Popen(
            [sys.executable, "train_with_reward.py", "--reward", reward, "--seed", str(s)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        for s in SEEDS
    ]
    outputs = [p.communicate()[0] for p in procs]
    if any(p.returncode != 0 for p in procs):
        log("💥 Training failed")
        return {"status": "training_failed", "error": "\n".join(o[-1000:] for o in outputs)}

    runs = []
    for s in SEEDS:
        r = evaluate(f"models/ppo_{reward}_seed{s}")
        Path(f"results/eval_ppo_{reward}_seed{s}.json").write_text(json.dumps(r, indent=2))
        runs.append(r)
    results = aggregate(runs)
    results["flags"] = detect_hacking(results)

    log(
        f"📊 Result: score {results['mean_score']:.1f} ± {results['score_std_across_seeds']} "
        f"(seeds: {results['per_seed_scores']}) | landed {results['landed_rate']:.0%} | "
        f"timed out {results['timeout_rate']:.0%} | length {results['mean_episode_length']:.0f}"
    )
    for flag in results["flags"]:
        log(f"⚠️ Detector: {flag}")
    return results


def main():
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(f"# RewardHacker agent v2 (3 seeds per experiment), {datetime.now():%Y-%m-%d %H:%M}\n")
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": GOAL}]
    experiments = 0

    for _ in range(20):
        response = client.messages.create(
            model=MODEL, max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "text" and block.text.strip():
                log(f"\n🧠 Claude: {block.text.strip()}")
        if response.stop_reason != "tool_use":
            log("\nClaude stopped without calling finish.")
            return

        tool_results, finished = [], False
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "finish":
                log(f"\n🏁 Finished. Best: {block.input['best_experiment']}")
                log(f"📝 Summary: {block.input['summary']}")
                finished, output = True, {"status": "ok"}
            elif experiments >= MAX_EXPERIMENTS:
                output = {"status": "refused", "reason": "experiment budget used up, call finish"}
            else:
                experiments += 1
                output = run_experiment(block.input["name"], block.input["idea"], block.input["code"], experiments)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)})
        if finished:
            return
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
