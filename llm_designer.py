import argparse
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You design reward bonuses for a PPO agent in Gymnasium's LunarLander-v3.

Each frame the agent sees obs = [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]:
x is horizontal distance from the pad, y is height, vx and vy are velocities,
angle is tilt in radians, and the legs are 1.0 when touching the ground.
Actions: 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine.

The built-in reward already pays for changes in a position score (distance, speed and tilt
to the pad), +10 per leg contact, -0.3 per main engine frame, -0.03 per side engine frame,
+100 for landing at rest and -100 for crashing. Episodes are cut off at 1,000 frames.

Your bonus is ADDED to the built-in reward. Agents are judged ONLY on the built-in reward
over 20 fixed seeds, so a bonus that teaches the agent a trick (like hovering to farm points)
will score badly. The goal is a bonus that helps PPO land more reliably within 1M steps.

Write exactly one function with this signature:

def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated) -> float:

Rules:
- Pure Python only. You may import math; no other imports.
- No file, network, or system access. No global state between calls.
- Return a float. Keep the bonus small relative to the built-in reward.
- The first line of the code must be a comment: # IDEA: <one short sentence describing the idea>

Reply with only the code in a single ```python block."""

PAST_RESULTS = """Results so far (3 seeds each, judged on built-in reward only):
- no bonus: score 222.5, landed 88%, timed out 2%
- +1 per frame while upright and slow: score -30.5, landed 3%, timed out 83% (agent learned to hover)
- 10 x frame-to-frame improvement in -(|angle|+|vx|+|vy|), minus 0.05 per frame: score 182.2, landed 78%
- potential-based version of that with gamma 0.99: score 168.4, landed 78% (descends too slowly, burns fuel)"""


def extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("No python code block found in the response")
    return match.group(1).strip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Claude to design a reward bonus.")
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": PAST_RESULTS + "\n\nDesign a new bonus that beats the no-bonus baseline."}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    code = extract_code(text)

    out = Path(f"rewards/llm_round{args.round}.py")
    out.write_text(code)
    print(code)
    print(f"Saved to {out} (NOT executed)")
