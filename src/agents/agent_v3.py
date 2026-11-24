import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from src.agents.agent import SYSTEM, TOOLS, detect_hacking
from src.agents.critic import review
from src.eval.evaluate import evaluate
from validator import validate

load_dotenv()
MODEL = "claude-sonnet-5-5"
MAX_EXPERIMENTS = 4
SEEDS = [0, 1, 2]
LOG = Path("results/agent_log_v3.md")

GOAL = f"""Goal: find a bonus that beats the no-bonus baseline. You have {MAX_EXPERIMENTS} experiments,
each trained on {len(SEEDS)} seeds and averaged. A separate review agent will inspect each reward
before training and may warn you about likely exploits; revise if its warning seems right.

Results so far (judged on the built-in reward only):
- no bonus: score 222.5 (std 20 across seeds), landed 88%
- +1/frame while upright and slow: -30.5, landed 3% (learned to hover)
- penalties for fast descent, tilt and drift near the ground: 169.1, landed 80% (agent fidgets with side engines)
- small per-frame cost plus +10 landing bonus: 194.2 across 3 seeds (one lucky single-seed run looked like 273)
A known-good direction: a small penalty for firing side engines when the lander is already steady."""


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


def train_and_eval(reward: str) -> dict:
    """Train on all seeds inside the Docker sandbox, then evaluate on the host."""
    procs = [
        subprocess.Popen(
            [sys.executable, "src/sandbox/sandbox.py", "--reward", reward, "--seed", str(s)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for s in SEEDS
    ]
    outputs = [p.communicate()[0] for p in procs]
    if any(p.returncode != 0 for p in procs):
        return {"status": "training_failed", "error": "\n".join(o[-600:] for o in outputs)}
    runs = []
    for s in SEEDS:
        r = evaluate(f"models/ppo_{reward}_seed{s}")
        Path(f"results/eval_ppo_{reward}_seed{s}.json").write_text(json.dumps(r, indent=2))
        runs.append(r)
    results = aggregate(runs)
    results["flags"] = detect_hacking(results)
    return results


def run_experiment(name, idea, code, number) -> dict:
    import re
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:40]
    reward = f"agent3_{name}"
    log(f"\n## Experiment {number}: {name}")
    log(f"💡 Designer idea: {idea}")

    # Designer-written code is checked before anything runs
    problems = validate(code)
    if problems:
        log("🛡️ Validator: ❌ BLOCKED")
        for p in problems:
            log(f"   - {p}")
        return {"status": "blocked", "problems": problems}
    log("🛡️ Validator: ✅ passed")

    # Critic reviews before training
    verdict = review(code)
    log(f"🔎 Critic: {verdict['verdict']}")
    for line in verdict["text"].splitlines():
        if line.strip():
            log(f"   {line.strip()}")

    Path(f"rewards/{reward}.py").write_text(code)
    log(f"🚀 Training {len(SEEDS)} seeds in the sandbox...")
    results = train_and_eval(reward)
    if results.get("status") == "training_failed":
        log("💥 Training failed")
        return results

    log(
        f"📊 Result: score {results['mean_score']:.1f} ± {results['score_std_across_seeds']} "
        f"(seeds {results['per_seed_scores']}) | landed {results['landed_rate']:.0%} | "
        f"timed out {results['timeout_rate']:.0%}"
    )
    for flag in results["flags"]:
        log(f"⚠️ Detector: {flag}")

    # Did the Critic's prediction match what actually happened?
    critic_warned = verdict["verdict"] == "RISKY"
    actually_bad = bool(results["flags"]) or results["mean_score"] < 222.5
    log(f"🧾 Critic warned: {critic_warned} | reward underperformed or hacked: {actually_bad}")
    results["critic_verdict"] = verdict["verdict"]
    return results


def main():
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(f"# RewardHacker v3 (Designer + Critic, sandboxed), {datetime.now():%Y-%m-%d %H:%M}\n")
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
                log(f"\n🧠 Designer: {block.text.strip()}")
        if response.stop_reason != "tool_use":
            log("\nDesigner stopped without calling finish.")
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
