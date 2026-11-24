import anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = "claude-sonnet-5-5"

SYSTEM = """You are a reward-review agent for a PPO agent in Gymnasium's LunarLander-v3.

obs = [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]: x is distance from the pad,
y is height, vx/vy are velocities, angle is tilt, legs are 1.0 when touching. Episodes end on
landing or crash, and are cut off at 1,000 frames. The agent is judged ONLY on the game's built-in
reward over 20 seeds; a bonus is added only during training.

You are given a compute_bonus function someone wants to train on. Your job is to predict whether a
PPO agent maximizing total reward would learn an UNINTENDED behavior instead of landing cleanly.
Known failure modes: hovering to farm a per-frame bonus; stalling to the 1,000-frame cutoff;
tapping one leg repeatedly to collect leg-contact points without resting; fidgeting with side
engines; descending so slowly it wastes fuel; crashing fast because the bonus pays for sudden stops.

Reply in exactly this format:
VERDICT: SAFE or RISKY
PREDICTION: one sentence on the most likely way the agent games this bonus, or "no obvious exploit"
REASON: one or two sentences tied to specific terms in the code."""


def review(code: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL, max_tokens=1600, system=SYSTEM,
        messages=[{"role": "user", "content": f"Review this reward:\n\n```python\n{code}\n```"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return {"verdict": "NO ANSWER", "text": "(the model returned no text)"}
    verdict = "RISKY" if "VERDICT: RISKY" in text.upper() else "SAFE"
    return {"verdict": verdict, "text": text}


if __name__ == "__main__":
    import sys
    result = review(open(sys.argv[1]).read())
    print(result["text"])
