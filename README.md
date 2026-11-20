# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/iphone14mania-dot/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/iphone14mania-dot/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each PS5 trophy in week one: a logit-rate trend model over the recent window of Exophase reads, with an earners-over-players fallback for thin histories, scored by mean pinball loss on the logit scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=path/to/cases MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Without those variables it reads `/input` and writes `/output`, as the grader does. Python 3.13, standard library only, no network, deterministic.

## How it works

`main.py` reads every `*.in` case file (schema `makefun/gta6-trophies-case/v1`) and writes one `*.out` per case: for every trophy, five quantiles (0.05/0.25/0.5/0.75/0.95) of its earn rate at the case's target.

**The rate trend (primary).** Exophase refreshes discretely, so consecutive identical reads are collapsed into one state per distinct page. The logit of each trophy's rate is then fit linearly against log(1 + hours since release) over the last 24 hours of distinct pages (widened to 48 hours when the window holds fewer than 3 states). The recent window matters: the launch regime — pre-release cohort, first flood — is a different world from the regime being extrapolated, and one line through both biases the slope. Each slope is shrunk toward its tier's median slope (precision-weighted empirical Bayes), and the extrapolated move is capped at 3 logit units from the last observed rate, so a steep recent trend cannot run away.

**The earners-over-players split (fallback).** When the window is too thin to fit — a flat or single-state history, as in the practice cases — players tracked N(t) is forecast from log N against log(1 + hours) (linear and quadratic fits blended, so the launch flood can decelerate), each trophy's earners E(t) from log(E + 0.5) against log(1 + hours) and against log N, averaged, and the median rate is 100 · E(T) / N(T).

**The pinned trophy.** A trophy (nearly) every tracked player holds — the first-mission trophy, at 100% on launch night and 99.99% by day two — has its median pinned at 99.99%: its earners track players one-for-one, and letting a fitted slope sag it costs enormously on the logit scale.

**The spread.** Each trophy's logit spread combines the residual scatter of its own logit trend (which already includes Exophase's discrete refresh jumps), inflated by how far the target is beyond the data, and a horizon term of 0.15 · sqrt(days). Quantiles are the median shifted by normal z-scores on the logit scale, rounded to 4 decimals and forced non-decreasing. If anything fails, the case falls back to last rates with a baseline-style random-walk spread, so the output stays valid.

## Validation

- **Synthetic backtests** (cohort simulator: power-law-decaying joins, weekend bump, per-trophy maturation, discrete Exophase refreshes, 2-decimal rounding; final-like cases, checkpoint day 2.5, target day 7): mean pinball loss **0.12–0.14** across three regimes (base / calm / surge, 8 seeds each), vs **0.24** for the house curve baseline and **0.37–0.38** for the house baseline. 90% coverage 0.99.
- **Real-data backtest** (launch-night history → the live Exophase snapshot at 20 Nov 09:00 UTC, 14,028 players): loss **0.85** vs **2.23** for the house baseline, 90% coverage 0.91.
- **Practice cases** p00/p01 run clean; outputs are valid (55 rows, exact shape, ≤65,536 bytes).
- **Determinism:** two runs are byte-identical. **Runtime:** ~0.1 s for 6 cases (limit 8 s). **Edge cases** (empty history, single read, all-null rates, short rate lists, zero players) all produce valid output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fcek93em&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
