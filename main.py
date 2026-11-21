"""GTA VI trophy rarity forecast for make.fun's gta6-trophy-rarity-forecast-01.

A program (Python 3.13, standard library only) that reads one case
(schema "makefun/gta6-trophies-case/v1") from the input directory and writes,
for every trophy, five quantiles of its earn rate on Exophase at the case's
target, as /output/<id>.out:

    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

Method (per trophy, all in logit space, rates clamped to 0.01..99.99 exactly
as the scorer clamps them):

  1. Fit a weighted least-squares line of logit(rate) against ln(1 + hours
     since the game's release). Weights are recency weights (24 h half-life)
     times (1 + players who had earned the trophy), which is the
     inverse-variance weight of a logit estimate from a counted rate.
  2. Shrink each trophy's slope toward its tier group's slope
     (empirical-Bayes: a noisy trophy borrows strength from its tier).
  3. The median is the last observed rate moved along the shrunk line to the
     target, with the move shrunk by 0.9 and capped at 2.5 logit units, so a
     flat history (all reads identical, as in the practice cases) reduces to
     the last rate, and a wild extrapolation can never blow up.
  4. The spread is the house random-walk spread (0.5 logit units per
     square-root day) combined with the fit's residual variance and the
     slope's uncertainty over the extrapolation distance, so honest
     uncertainty about a 4.5-day forecast is wider than the house band.

The house persistence baseline says every rate stays where it is; this
program instead follows the trend the house's own recording shows, which is
what a prize needs: a loss strictly below the best house entry's.

Deterministic: no randomness, no wall-clock reads, no network, no imports
beyond the standard library. Two runs on the same case write identical bytes.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs `python3 -P -s /bot/main.py` with MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR unset (so /input and /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the house baseline's values).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a rate's move, in logit units per square-root day.
SIGMA = 0.5
# Half-life (hours) of the recency weight on history reads.
RECENCY_HALF_LIFE_H = 24.0
# Global shrink on the extrapolated logit move (guards against overshoot).
MOVE_SHRINK = 0.9
# Largest logit move allowed from the last observed rate.
MAX_MOVE = 2.5
# Spread bounds, in logit units.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.2


def clamp_pct(x: float) -> float:
    """A rate in percent, clamped to the range the scorer uses."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the house baseline rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation), for non-standard levels."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_values(levels) -> list:
    if list(levels) == LEVELS:
        return list(Z)
    return [norm_ppf(p) for p in levels]


def fit_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, se_b, resid_var)."""
    w_sum = sum(ws)
    if w_sum <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0, 0.0, 0.0, 0.0)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / w_sum
    y_bar = sum(w * y for w, y in zip(ws, ys)) / w_sum
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (y_bar, 0.0, 0.0, 0.0)
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = y_bar - b * x_bar
    dof = max(w_sum - 2.0, 0.5)
    resid_var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    se_b = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    return (a, b, se_b, resid_var)


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels", LEVELS)
    z = z_values(levels)
    trophies = case["trophies"]
    history = case.get("history") or []

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    t_target_h = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target_h, 0.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    base_spread = SIGMA * math.sqrt(max(horizon_days, 0.0))

    # Per-trophy series: (x, y, w) with x = ln(1 + hours since release).
    # The last read's time anchors the recency weights; compute it once.
    last_slot = None
    if history:
        last_read = history[-1]
        last_slot = last_read.get("slot") or last_read.get("at")
    t_last_h = None
    if last_slot:
        t_last_h = (parse_time(last_slot) - release).total_seconds() / 3600.0
    series = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            slot = read.get("slot") or read.get("at")
            if not slot:
                continue
            t_h = (parse_time(slot) - release).total_seconds() / 3600.0
            if t_h < -1.0:
                continue
            rate = rates[i]
            # A printed 0.00% means the true rate is below half a hundredth;
            # 0.005 is its midpoint. A printed 100% is capped at 99.99,
            # the same clamp the scorer applies.
            y = logit(0.005 if rate == 0 else rate)
            earned = (read.get("players_tracked") or 0) * max(rate, 0.0) / 100.0
            age_h = max(0.0, t_last_h - t_h) if t_last_h is not None else 0.0
            recency = 0.5 ** (age_h / RECENCY_HALF_LIFE_H)
            xs.append(math.log1p(max(t_h, 0.0)))
            ys.append(y)
            ws.append(recency * (1.0 + earned))
        series.append((xs, ys, ws))

    # Fit each trophy, then shrink its slope toward its tier's group slope.
    fits = []
    for (xs, ys, ws) in series:
        fits.append(fit_line(xs, ys, ws))
    groups: dict[str, list] = {}
    for trophy, fit in zip(trophies, fits):
        groups.setdefault(trophy.get("tier", "bronze"), []).append(fit)
    group_stats: dict[str, tuple] = {}
    for tier, tier_fits in groups.items():
        slopes = [f[1] for f in tier_fits]
        mean = sum(slopes) / len(slopes)
        var = sum((s - mean) ** 2 for s in slopes) / len(slopes)
        tau = min(max(math.sqrt(var), 0.15), 1.0)
        group_stats[tier] = (mean, tau)

    rows = []
    for idx, (trophy, (xs, ys, ws), (a, b, se_b, resid_var)) in enumerate(zip(trophies, series, fits)):
        tier = trophy.get("tier", "bronze")
        g_mean, g_tau = group_stats[tier]
        if se_b > 0:
            prec = 1.0 / (se_b * se_b)
            g_prec = 1.0 / (g_tau * g_tau)
            b_shrunk = (b * prec + g_mean * g_prec) / (prec + g_prec)
            se_shrunk = math.sqrt(1.0 / (prec + g_prec))
        else:
            b_shrunk = b
            se_shrunk = 0.0
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        if last_rate is None:
            last_rate = 4.7  # no history at all: a neutral low rate
        y_last = logit(0.005 if last_rate == 0 else last_rate)
        # Extrapolated move from the last observed rate, shrunk and capped.
        x_bar = (sum(w * x for w, x in zip(ws, xs)) / sum(ws)) if ws else x_target
        move = b_shrunk * (x_target - x_bar)
        move = max(-MAX_MOVE, min(MAX_MOVE, MOVE_SHRINK * move))
        median_logit = y_last + move
        # Spread: random walk + residual variance + slope uncertainty.
        var = base_spread * base_spread + resid_var
        var += (se_shrunk * (x_target - x_bar)) ** 2
        spread = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(max(var, 0.0))))
        q = [round4(clamp_pct(inv_logit(median_logit + zz * spread))) for zz in z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
