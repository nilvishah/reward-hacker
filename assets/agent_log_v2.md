# RewardHacker agent v2 (3 seeds per experiment), 2026-11-20 23:08

## Experiment 1: settle_and_tiny_time_cost
💡 Idea: Tiny -0.03/frame time cost plus +1 when both legs touch and the agent idles, to encourage cutting engines to settle after touchdown.
🛡️ Checker: ✅ passed
🚀 Training 3 seeds in parallel...
📊 Result: score 213.7 ± 30.9 (seeds: [252.7, 177.2, 211.3]) | landed 88% | timed out 3% | length 363
⚠️ Detector: FIDGETING: side engines used more than twice the baseline

## Experiment 2: anti_fidget_settle
💡 Idea: Penalize side-engine firing when tilt and spin are already small, keep the grounded-idle bonus, drop time cost.
🛡️ Checker: ✅ passed
🚀 Training 3 seeds in parallel...
📊 Result: score 245.9 ± 28.3 (seeds: [206.5, 259.6, 271.5]) | landed 95% | timed out 0% | length 260

## Experiment 3: anti_fidget_stronger
💡 Idea: Stronger and wider side-engine penalty when stable, same grounded-idle bonus.
🛡️ Checker: ✅ passed
🚀 Training 3 seeds in parallel...
📊 Result: score 232.4 ± 49.9 (seeds: [162.7, 257.9, 276.6]) | landed 88% | timed out 0% | length 289

## Experiment 4: anti_fidget_only
💡 Idea: Ablation: only the -0.1 stable side-engine penalty, no grounded-idle bonus.
🛡️ Checker: ✅ passed
🚀 Training 3 seeds in parallel...
📊 Result: score 242.3 ± 7.8 (seeds: [240.1, 252.7, 234.1]) | landed 98% | timed out 0% | length 361

🏁 Finished. Best: anti_fidget_only
📝 Summary: Best bonus: -0.1 per frame for firing a side engine when |angle|<0.15 and |angular velocity|<0.3, so only unnecessary corrections are penalized. It scored 242.3 (per-seed 240.1, 252.7, 234.1; std 7.8) and landed 98%, against the baseline's 222.5 (std 20). Adding a +1 grounded-idle bonus gave 245.9, but with a higher per-seed spread (206.5, 259.6, 271.5), so the two are statistically indistinguishable. I chose the simpler, more consistent version. Making the penalty stronger and wider (-0.2) hurt: 232.4 with std 49.9. A plain time cost plus the idle bonus gave 213.7 and caused fidgeting. Gains are modest, about 20 points, which is roughly one baseline std. Only 3 seeds per run were available, so the improvement is suggestive rather than conclusive.
