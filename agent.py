import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from evaluate import evaluate
from validator import validate

load_dotenv()
MODEL = "claude-sonnet-5-5"
MAX_EXPERIMENTS = 5
TRAIN_STEPS = 1_000_000
LOG = Path("results/agent_log.md")

SYSTEM = """You are a research agent designing reward bonuses for a PPO agent in Gymnasium's LunarLander-v3.

Each frame the agent sees obs = [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]:
x is horizontal distance from the pad, y is height, vx and vy are velocities,
angle is tilt in radians, and the legs are 1.0 when touching the ground.
Actions: 0 = nothing, 1 = left engine, 2 = main engine, 3 = right engine.

The built-in reward already pays for changes in a position score (distance, speed and tilt),
+10 per leg contact, -0.3 per main engine frame, -0.03 per side engine frame,
+100 for landing at rest and -100 for crashing. Episodes are cut off at 1,000 frames.

Your bonus is ADDED to the built-in reward during training. Agents are judged ONLY on the
built-in reward over 20 fixed seeds, so a bonus that teaches a trick (like hovering) scores badly.

Every bonus must be one function:
def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:
Rules: pure Python, only `import math` allowed, no file/network/system access, no global state,
no while loops, return a finite float with absolute value at most 10. A validator blocks anything else.

Use the run_experiment tool to test ideas (each takes about 4 minutes and uses one of your
limited experiments). Read the results and warning flags carefully, explain your reasoning
briefly before each experiment, and call finish when you are done."""

GOAL = f"""Goal: find a bonus that beats the no-bonus baseline. You have {MAX_EXPERIMENTS} experiments.

Results so far, judged on the built-in reward only:
- no bonus (3 seeds): score 222.5, landed 88%, timed out 2%, ~50 side-engine frames per episode
- +1 per frame while upright and slow: score -30.5, landed 3%, timed out 83% (learned to hover)
- 10 x improvement in -(|angle|+|vx|+|vy|) minus 0.05/frame: score 182.2, landed 78%
- potential-based version of that: score 168.4, landed 78% (too slow, burns fuel)
- your earlier attempt (penalties for fast descent, tilt and drift near the ground):
  score 169.1, landed 80%, crashed 15%, 152 side-engine frames (agent fidgets with side engines)"""

TOOLS = [
    {
        "name": "run_experiment",
        "description": "Validate a reward bonus, train PPO on it for 1M steps, and evaluate on the built-in reward over 20 fixed seeds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short snake_case name"},
                "idea": {"type": "string", "description": "one sentence describing the idea"},
                "code": {"type": "string", "description": "full Python source defining compute_bonus"},
            },
            "required": ["name", "idea", "code"],
        },
    },
    {
        "name": "finish",
        "description": "Call when you are done. Name your best experiment and summarize what you learned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "best_experiment": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["best_experiment", "summary"],
        },
    },
]


def log(text: str):
    print(text)
    with LOG.open("a") as f:
        f.write(text + "\n")


def detect_hacking(r: dict) -> list:
    """Automatic warning flags based on how attempts ended."""
    flags = []
    if r["timeout_rate"] >= 0.2:
        flags.append("HOVERING: many attempts hit the 1,000-frame limit without landing")
    if r["mean_episode_length"] >= 700:
        flags.append("STALLING: attempts are very long")
    if r["crashed_rate"] >= 0.2:
        flags.append("CRASHING: many attempts end in a crash")
    if r["mean_side_engine_frames"] > 100:
        flags.append("FIDGETING: side engines used more than twice the baseline")
    return flags


def run_experiment(name: str, idea: str, code: str, number: int) -> dict:
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:40]
    reward = f"agent_{name}"
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
    log("🚀 Training (about 4 minutes)...")
    proc = subprocess.run(
        [sys.executable, "train_with_reward.py", "--reward", reward, "--steps", str(TRAIN_STEPS)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log("💥 Training failed")
        return {"status": "training_failed", "error": (proc.stdout + proc.stderr)[-2000:]}

    results = evaluate(f"models/ppo_{reward}_seed0")
    results["flags"] = detect_hacking(results)
    Path(f"results/eval_ppo_{reward}_seed0.json").write_text(json.dumps(results, indent=2))
    log(
        f"📊 Result: score {results['mean_score']} | landed {results['landed_rate']:.0%} | "
        f"crashed {results['crashed_rate']:.0%} | timed out {results['timeout_rate']:.0%} | "
        f"side engine {results['mean_side_engine_frames']}"
    )
    for flag in results["flags"]:
        log(f"⚠️ Detector: {flag}")
    return results


def main():
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(f"# RewardHacker agent run, {datetime.now():%Y-%m-%d %H:%M}\n")
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": GOAL}]
    experiments = 0

    for _ in range(20):  # hard cap on agent turns
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

        tool_results = []
        finished = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "finish":
                log(f"\n�� Finished. Best: {block.input['best_experiment']}")
                log(f"📝 Summary: {block.input['summary']}")
                finished = True
                output = {"status": "ok"}
            elif experiments >= MAX_EXPERIMENTS:
                output = {"status": "refused", "reason": "experiment budget used up, call finish"}
            else:
                experiments += 1
                output = run_experiment(
                    block.input["name"], block.input["idea"], block.input["code"], experiments
                )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
            )

        if finished:
            return
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
