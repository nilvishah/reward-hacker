import subprocess
import sys
from pathlib import Path

# Known attack patterns a malicious reward function might try.
ATTACKS = {
    "read_env_file": "import os\ndef compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    return float(len(open('/app/.env').read()))\n",
    "network_exfil": "import urllib.request\ndef compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    urllib.request.urlopen('http://example.com')\n    return 0.0\n",
    "spawn_shell": "import subprocess\ndef compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    subprocess.run(['echo', 'hacked'])\n    return 0.0\n",
    "hidden_import_eval": "def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    eval(\"__import__('os').system('echo hacked')\")\n    return 0.0\n",
    "infinite_loop": "def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    while True:\n        pass\n",
    "tamper_models": "def compute_bonus(obs, prev_obs, action, base_reward, terminated, truncated):\n    open('models/canary.txt', 'w').write('tampered')\n    return 0.0\n",
}

from validator import validate


def run_one(name, code):
    problems = validate(code)
    if problems:
        return "blocked by validator", problems[0]

    Path(f"rewards/sec_{name}.py").write_text(code)
    canary = Path("models/canary.txt")
    canary.parent.mkdir(exist_ok=True)
    canary.write_text("intact")

    proc = subprocess.run(
        [sys.executable, "src/sandbox/sandbox.py", "--reward", f"sec_{name}", "--seed", "0", "--steps", "2000"],
        capture_output=True, text=True, timeout=120,
    )
    if not (canary.exists() and canary.read_text() == "intact"):
        return "GOT THROUGH", "tampered with models/ folder"
    return "contained by sandbox", f"exit code {proc.returncode}"


def main():
    print(f"{'attack':<22}{'result':<24}detail")
    print("-" * 70)
    for name, code in ATTACKS.items():
        try:
            result, detail = run_one(name, code)
        except subprocess.TimeoutExpired:
            result, detail = "contained by sandbox", "killed by time limit"
        print(f"{name:<22}{result:<24}{detail}")
    Path("models/canary.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
