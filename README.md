# makefun-gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Alhafidm212/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Alhafidm212/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: a Python program that reads the earn rate of every Grand Theft Auto VI trophy on Exophase as the house recorded it up to the deadline and forecasts each trophy's rate at 2026-11-26 00:00 UTC as five quantiles (5%, 25%, 50%, 75%, 95%), for the lowest pinball loss on the logit scale.

Method (`main.py`, Python 3.13 standard library only): per trophy, a weighted least-squares line of logit(rate) against ln(1 + hours since release), with recency weights (24 h half-life) times (1 + players who had earned the trophy); slopes shrunk toward their tier group's slope (empirical-Bayes); the median is the last observed rate moved along the shrunk line to the target (move shrunk by 0.9, capped at 2.5 logit units); the spread combines the house random-walk spread with the fit's residual variance and slope uncertainty. On a flat history it reduces to the last rate, like the house persistence baseline — the edge is following the trend the recording actually shows.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` case in the input directory and writes `<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory. It runs offline with no network; two runs on the same case write identical bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=qy5sv934&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
