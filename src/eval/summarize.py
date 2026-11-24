import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

groups = defaultdict(list)
for path in sorted(Path("results").glob("eval_ppo_*_seed*.json")):
    data = json.loads(path.read_text())
    reward = re.match(r"eval_ppo_(.+)_seed\d+", path.stem).group(1)
    groups[reward].append(data)

print(f"{'reward':<14}{'seeds':>6}{'score':>18}{'landed':>9}{'timeout':>9}{'length':>9}")
for reward, runs in groups.items():
    scores = [r["mean_score"] for r in runs]
    landed = np.mean([r["landed_rate"] for r in runs])
    timeout = np.mean([r["timeout_rate"] for r in runs])
    length = np.mean([r["mean_episode_length"] for r in runs])
    score = f"{np.mean(scores):.1f} ± {np.std(scores):.1f}"
    print(f"{reward:<14}{len(runs):>6}{score:>18}{landed:>9.0%}{timeout:>9.0%}{length:>9.0f}")
