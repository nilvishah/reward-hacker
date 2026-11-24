#!/bin/bash
# Train and evaluate each reward across 3 seeds.
set -e
for reward in none hover_hack hover_fixed; do
  for seed in 0 1 2; do
    model="models/ppo_${reward}_seed${seed}"
    if [ ! -f "${model}.zip" ]; then
      python3 train_with_reward.py --reward "$reward" --seed "$seed"
    fi
    python3 src/eval/evaluate.py --model "$model"
  done
done
