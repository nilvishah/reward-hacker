import argparse
import subprocess
import sys
from pathlib import Path

DOCKER = "/Users/nilvishah/.docker/bin/docker"
IMAGE = "reward-sandbox"
TIME_LIMIT_SECONDS = 30 * 60


def sandbox_command(reward: str, seed: int, steps: int) -> list:
    """Docker command that trains on a reward inside a locked-down container."""
    root = Path.cwd()
    return [
        DOCKER, "run", "--rm",
        "--name", f"sandbox_{reward}_seed{seed}",
        "--network", "none",                      # no internet
        "--memory", "2g", "--cpus", "1",          # resource caps
        "--pids-limit", "256",                    # no process explosions
        "--read-only", "--tmpfs", "/tmp",         # read-only filesystem
        "--cap-drop", "ALL",                      # no special system powers
        "--security-opt", "no-new-privileges",
        "-v", f"{root / 'rewards'}:/app/rewards:ro",  # rewards: read-only
        "-v", f"{root / 'models'}:/app/models",       # models: the only writable folder
        IMAGE,
        "python", "train_with_reward.py",
        "--reward", reward, "--seed", str(seed), "--steps", str(steps),
    ]


def start_training(reward: str, seed: int, steps: int = 1_000_000) -> subprocess.Popen:
    """Start sandboxed training in the background."""
    (Path.cwd() / "models").mkdir(exist_ok=True)
    return subprocess.Popen(
        sandbox_command(reward, seed, steps),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def wait_for(proc: subprocess.Popen, reward: str, seed: int) -> tuple:
    """Wait for a sandboxed run; kill it if it exceeds the time limit."""
    try:
        output, _ = proc.communicate(timeout=TIME_LIMIT_SECONDS)
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        subprocess.run([DOCKER, "kill", f"sandbox_{reward}_seed{seed}"], capture_output=True)
        proc.kill()
        return -1, f"killed: exceeded {TIME_LIMIT_SECONDS}s time limit"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on a reward inside the Docker sandbox.")
    parser.add_argument("--reward", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()

    proc = start_training(args.reward, args.seed, args.steps)
    code, output = wait_for(proc, args.reward, args.seed)
    print(output[-1500:])
    sys.exit(code)
