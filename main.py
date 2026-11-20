#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a target time.

Reads each case file in the input directory (schema
makefun/gta6-trophies-case/v1) and writes one forecast per case: for every
trophy, five quantiles (0.05/0.25/0.5/0.75/0.95) of its earn rate at the
case's target, scored by mean pinball loss on the logit scale.

Model
-----
The primary forecast models each trophy's earn rate directly: the logit
of the rate is fit linearly against log(1 + hours since release) over
the recent window (the last 24 hours of distinct Exophase pages, widened
to 48 hours when the window is thin).  The recent window matters because
the launch regime (pre-release cohort, first flood) is a different world
from the regime being extrapolated; fitting one line through both
regimes biases the slope.  Each slope is shrunk toward its tier's median
slope (precision-weighted empirical Bayes), so trophies with noisy
histories borrow strength from their tier instead of extrapolating
rounding noise.  The extrapolated move is capped at 3 logit units from
the last observed rate, so a steep recent trend cannot run away.

When the recent window is too thin to fit (a flat or single-state
history, as in the practice cases), the forecast falls back to an
earners-over-players decomposition: players tracked N(t) is forecast
from log N against log(1 + hours) (linear and quadratic fits blended),
and each trophy's earners E(t) from log(E + 0.5) against log(1 + hours)
and against log N, averaged; the median rate is 100 * E(T) / N(T).

Spread
------
Per-trophy logit spread combines the residual scatter of the rate's own
logit trend (which already includes Exophase's discrete refresh jumps),
inflated by how far the target is beyond the data, and a horizon term
that grows like sqrt(days).  Quantiles are the median shifted by normal
z-scores on the logit scale.

Only the standard library is used.  The program is deterministic: no
randomness, no clock reads, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the five levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

RATE_LO = 0.01
RATE_HI = 99.99

# Fallback slopes when a trophy has no fittable history: how fast its
# earners grow, per tier, in log(1 + hours).  Early/easy trophies grow
# fastest; the platinum grows slowest.
PRIOR_TIME_SLOPE = {"platinum": 0.2, "gold": 0.3, "silver": 0.4,
                    "bronze": 0.6}
# Fallback elasticity of earners w.r.t. players tracked.
PRIOR_N_SLOPE = 0.35
PRIOR_N_SLOPE_COMMON = 0.95  # trophies almost every player earns
# Prior slope for players-tracked growth when the history is flat.
PRIOR_N_GROWTH_SLOPE = 1.0
# Prior spread (in the slope) used to shrink fitted slopes toward the
# tier median.
SLOPE_PRIOR_SD = 0.5
# Rate-trend window: fit the logit-rate trend on the last day of
# distinct pages; widen to two days when the window is thin.
RATE_WINDOW_HOURS = 24.0
RATE_WINDOW_MIN_STATES = 3
RATE_WINDOW_EXPAND_HOURS = 48.0
# Cap on the extrapolated logit move from the last observed rate.
RATE_MOVE_CAP = 3.0
# Prior spread for rate-trend slopes (looser than the earner slopes:
# the rate trend is the primary model and its fits are usually clean).
RATE_SLOPE_PRIOR_SD = 0.75
# Horizon term of the logit spread: HORIZON_COEF * sqrt(days).
HORIZON_COEF = 0.15
# Floor on the per-trophy logit spread (except the >=99% branch).
SIGMA_FLOOR = 0.30


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit(rate):
    """Logit of a percent rate, clamped to [0.01, 99.99] first."""
    p = min(max(rate, RATE_LO), RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    """Inverse logit, returned as a percent."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def ols(xs, ys):
    """Ordinary least squares of ys on xs.

    Returns (intercept, slope, resid_sd, n, x_mean, sxx), or None when
    the fit is impossible (fewer than 2 points or no x variance).
    """
    n = len(xs)
    if n < 2:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    else:
        resid_sd = 0.0
    return (intercept, slope, resid_sd, n, x_mean, sxx)


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def shrink_slope(slope, se, prior_mean):
    """Shrink a fitted slope toward a prior mean, precision-weighted."""
    if se is None or se <= 0.0:
        se = 0.35
    w_fit = 1.0 / (se * se)
    w_prior = 1.0 / (SLOPE_PRIOR_SD * SLOPE_PRIOR_SD)
    return (slope * w_fit + prior_mean * w_prior) / (w_fit + w_prior)


def distinct_states(history):
    """Collapse consecutive identical reads into one state each.

    Exophase refreshes discretely, so long stretches of reads are the
    same page.  Keeping one state per distinct page stops flat stretches
    from dominating the fits.  Each state keeps its last read's time.
    """
    states = []
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if states and states[-1][0] == key:
            states[-1] = (key, read)
        else:
            states.append((key, read))
    return [read for _, read in states]


def forecast_players(states, x_target, x_last, ln_n_last):
    """Forecast log players tracked at the target.

    Returns (ln_n_target, sigma_ln_n).  Blends a linear and a quadratic
    fit of log N against log(1 + hours); falls back to a prior growth
    slope when the history has no usable shape.
    """
    pts = []
    for read in states:
        n = read.get("players_tracked")
        if n and n > 0:
            pts.append((read["_x"], math.log(n)))
    sigma = 0.6
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fit = ols(xs, ys)
        if fit is not None:
            a, b, resid_sd, n, x_mean, sxx = fit
            pred_lin = a + b * x_target
            pred = pred_lin
            if len(pts) >= 3:
                # Quadratic term, fit by least squares on the residuals
                # of the linear fit against (x - x_mean)^2.
                quad = ols([(x - x_mean) ** 2 for x in xs],
                           [y - (a + b * x) for x, y in zip(xs, ys)])
                if quad is not None:
                    qc = quad[1]
                    # Only trust a decelerating (negative) quadratic
                    # term; an accelerating one is usually noise.
                    if qc < 0.0:
                        pred_quad = (pred_lin + quad[0]
                                     + qc * (x_target - x_mean) ** 2)
                        pred = 0.5 * pred_lin + 0.5 * pred_quad
            # Extrapolation-aware uncertainty.
            lev = ((x_target - x_mean) ** 2 / sxx) + 1.0 / n
            sigma = max(0.25, resid_sd * math.sqrt(1.0 + lev))
            sigma = min(sigma, 1.0)
            return clamp_ln_n(pred, ln_n_last), sigma
    # Flat or single-point history: prior growth slope from the last read.
    pred = ln_n_last + PRIOR_N_GROWTH_SLOPE * max(0.0, x_target - x_last)
    return clamp_ln_n(pred, ln_n_last), sigma


def clamp_ln_n(value, ln_n_last):
    """Players tracked never falls; cap growth at 30x over the horizon."""
    return min(max(value, ln_n_last), ln_n_last + math.log(30.0))


def tier_medians(fits):
    """Median fitted slope per tier, plus a global median fallback."""
    by_tier = {}
    all_slopes = []
    for tier, slope in fits:
        by_tier.setdefault(tier, []).append(slope)
        all_slopes.append(slope)
    global_med = median(all_slopes) if all_slopes else 0.0
    out = {tier: median(slopes) for tier, slopes in by_tier.items()}
    return out, global_med


def forecast_earners(trophy, states, x_target, ln_n_target,
                     tier_time, global_time, tier_n, global_n):
    """Forecast a trophy's earners at the target.

    Returns (earners, resid_sd): the earner count and the residual
    scatter of the log-earner fits, for the spread.
    """
    idx = trophy["id"] - 1
    tier = trophy.get("tier", "bronze")
    pts_x, pts_u, pts_y = [], [], []
    last_e = None
    last_rate = None
    for read in states:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if idx >= len(rates) or rates[idx] is None or not n or n <= 0:
            continue
        rate = rates[idx]
        e = rate * n / 100.0
        pts_x.append(read["_x"])
        pts_u.append(math.log(n))
        pts_y.append(math.log(e + 0.5))
        last_e = e
        last_rate = rate
    if last_e is None:
        # Trophy never observed: tier rate prior times target players.
        prior_rate = {"platinum": 0.05, "gold": 0.3, "silver": 0.5,
                      "bronze": 1.0}.get(tier, 1.0)
        return prior_rate * math.exp(ln_n_target) / 100.0, 0.6

    preds, resids = [], []
    fit_x = ols(pts_x, pts_y)
    if fit_x is not None:
        a, b, resid_sd, n, x_mean, sxx = fit_x
        se = resid_sd / math.sqrt(sxx) if n > 2 else 0.5
        prior = tier_time.get(tier, global_time)
        b_s = shrink_slope(b, se, prior)
        b_s = min(max(b_s, -0.2), 2.5)
        a_s = sum(pts_y) / n - b_s * x_mean
        preds.append(a_s + b_s * x_target)
        resids.append(resid_sd)
    fit_u = ols(pts_u, pts_y)
    if fit_u is not None:
        a, b, resid_sd, n, u_mean, suu = fit_u
        se = resid_sd / math.sqrt(suu) if n > 2 else 0.5
        prior = tier_n.get(tier, global_n)
        b_s = shrink_slope(b, se, prior)
        b_s = min(max(b_s, 0.0), 1.5)
        a_s = sum(pts_y) / n - b_s * u_mean
        preds.append(a_s + b_s * ln_n_target)
        resids.append(resid_sd)
    if not preds:
        # One usable point: extrapolate with prior slopes from it.
        x0 = pts_x[-1]
        y0 = pts_y[-1]
        b_t = PRIOR_TIME_SLOPE.get(tier, 0.6)
        b_n = (PRIOR_N_SLOPE_COMMON if (last_rate or 0.0) > 50.0
               else PRIOR_N_SLOPE)
        preds.append(y0 + b_t * max(0.0, x_target - x0))
        preds.append(y0 + b_n * max(0.0, ln_n_target - pts_u[-1]))
        resids.append(0.4)
    log_e = sum(preds) / len(preds)
    earners = math.exp(log_e) - 0.5
    earners = max(earners, 0.5 * last_e)
    earners = min(earners, math.exp(ln_n_target))
    resid = sum(resids) / len(resids) if resids else 0.4
    return max(earners, 0.0), resid


def logit_trend_spread(trophy, states):
    """Residual scatter of the trophy's logit rate around a log-time trend."""
    idx = trophy["id"] - 1
    xs, ys = [], []
    for read in states:
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            xs.append(read["_x"])
            ys.append(logit(rates[idx]))
    fit = ols(xs, ys)
    if fit is None:
        return 0.4, 0.0
    _, _, resid_sd, _, _, _ = fit
    x_range = max(xs) - min(xs) if xs else 0.0
    return resid_sd, x_range


def window_states(states):
    """The recent window of distinct states for the rate trend.

    The last RATE_WINDOW_HOURS of states; widened to
    RATE_WINDOW_EXPAND_HOURS when the window is thin, and to all states
    when even that is thin.  A flat history yields one state.
    """
    if not states:
        return []
    h_last = states[-1]["_h"]
    win = [r for r in states if r["_h"] >= h_last - RATE_WINDOW_HOURS]
    if len(win) < RATE_WINDOW_MIN_STATES:
        win = [r for r in states
               if r["_h"] >= h_last - RATE_WINDOW_EXPAND_HOURS]
    if len(win) < RATE_WINDOW_MIN_STATES:
        win = states
    return win


def rate_trend_points(trophy, states):
    """(x, logit rate) points for one trophy over the given states."""
    idx = trophy["id"] - 1
    xs, ys = [], []
    for read in states:
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            xs.append(read["_x"])
            ys.append(logit(rates[idx]))
    return xs, ys


def rate_trend_forecast(trophy, states, x_target, tier_rate, global_rate):
    """Forecast a trophy's rate from its recent logit-rate trend.

    Returns (median_rate, resid_sd, x_range), or None when the window
    cannot be fit (too few points or no time variance).  The slope is
    shrunk toward the tier median; the extrapolated move from the last
    observed rate is capped at RATE_MOVE_CAP logit units.
    """
    tier = trophy.get("tier", "bronze")
    xs, ys = rate_trend_points(trophy, states)
    fit = ols(xs, ys)
    if fit is None or len(xs) < 3:
        return None
    a, b, resid_sd, n, x_mean, sxx = fit
    se = resid_sd / math.sqrt(sxx) if n > 2 else 0.5
    prior = tier_rate.get(tier, global_rate)
    w_fit = 1.0 / (se * se) if se > 0 else 1.0 / 0.1225
    w_prior = 1.0 / (RATE_SLOPE_PRIOR_SD * RATE_SLOPE_PRIOR_SD)
    b_s = (b * w_fit + prior * w_prior) / (w_fit + w_prior)
    a_s = sum(ys) / n - b_s * x_mean
    med_logit = a_s + b_s * x_target
    last_logit = ys[-1]
    med_logit = min(max(med_logit, last_logit - RATE_MOVE_CAP),
                    last_logit + RATE_MOVE_CAP)
    x_range = max(xs) - min(xs)
    return sigmoid(med_logit), resid_sd, x_range


def forecast_case(case):
    """Forecast one case. Returns the output object."""
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    x_target = math.log(1.0 + max((target - release).total_seconds()
                                  / 3600.0, 0.0))

    states = distinct_states(history)
    for read in states:
        when = parse_time(read["at"])
        hours = max((when - release).total_seconds() / 3600.0, 0.0)
        read["_h"] = hours
        read["_x"] = math.log(1.0 + hours)

    last = states[-1] if states else None
    n_last = (last.get("players_tracked") or 0) if last else 0
    if n_last <= 0:
        # No usable history at all: tier priors with a wide spread.
        return {"trophies": [prior_row(t) for t in trophies]}
    ln_n_last = math.log(n_last)
    x_last = last["_x"]
    ln_n_target, sigma_n = forecast_players(states, x_target, x_last,
                                            ln_n_last)
    n_target = math.exp(ln_n_target)
    win = window_states(states)

    # First pass: raw fitted slopes per trophy, for tier medians.  The
    # rate-trend slopes come from the recent window; the earner slopes
    # (fallback path) from the full history.
    time_slopes, n_slopes, rate_slopes = [], [], []
    for trophy in trophies:
        idx = trophy["id"] - 1
        xs, us, ys = [], [], []
        for read in states:
            rates = read.get("rates") or []
            n = read.get("players_tracked")
            if idx >= len(rates) or rates[idx] is None or not n or n <= 0:
                continue
            xs.append(read["_x"])
            us.append(math.log(n))
            ys.append(math.log(rates[idx] * n / 100.0 + 0.5))
        tier = trophy.get("tier", "bronze")
        fx = ols(xs, ys)
        if fx is not None:
            time_slopes.append((tier, fx[1]))
        fu = ols(us, ys)
        if fu is not None:
            n_slopes.append((tier, fu[1]))
        rx, ry = rate_trend_points(trophy, win)
        fr = ols(rx, ry)
        if fr is not None and len(rx) >= 3:
            rate_slopes.append((tier, fr[1]))
    tier_time, global_time = tier_medians(time_slopes)
    tier_n, global_n = tier_medians(n_slopes)
    tier_rate, global_rate = tier_medians(rate_slopes)

    horizon_term = HORIZON_COEF * math.sqrt(horizon_days)
    last_rates = {}
    if last is not None:
        for i, rate in enumerate(last.get("rates") or []):
            if rate is not None:
                last_rates[i + 1] = rate
    x_max = max((r["_x"] for r in states), default=x_target)
    rows = []
    for trophy in trophies:
        trend = rate_trend_forecast(trophy, win, x_target,
                                    tier_rate, global_rate)
        if trend is not None:
            rate_med, resid_logit, x_range = trend
            rate_med = min(max(rate_med, RATE_LO), RATE_HI)
            from_trend = True
        else:
            # Thin window: fall back to the earners-over-players split.
            earners, _ = forecast_earners(
                trophy, states, x_target, ln_n_target,
                tier_time, global_time, tier_n, global_n)
            rate_med = 100.0 * earners / n_target
            rate_med = min(max(rate_med, RATE_LO), RATE_HI)
            resid_logit, x_range = logit_trend_spread(trophy, states)
            from_trend = False
        if last_rates.get(trophy["id"], 0.0) >= 99.0:
            # A trophy (nearly) every tracked player holds: earners
            # track players one-for-one, so its rate stays pinned at
            # the top.  A fitted slope would wrongly let it sag as
            # players flood in, and on the logit scale that sag is
            # enormously expensive.
            rate_med = RATE_HI
        infl = math.sqrt(1.0 + max(0.0, x_target - x_max)
                         / max(x_range, 0.5))
        sigma_fit = max(0.15, resid_logit * infl)
        if rate_med >= 99.0:
            # Earners track players almost exactly here, so the
            # players-tracked uncertainty cancels; keep a modest spread.
            sigma = math.sqrt(sigma_fit ** 2 + horizon_term ** 2)
            sigma = min(max(sigma, 0.5), 1.0)
        elif from_trend:
            sigma = math.sqrt(sigma_fit ** 2 + horizon_term ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), 2.0)
        else:
            # The earner split leans on the players-tracked forecast,
            # so its uncertainty stays in the spread.
            mult = min(1.0 / (1.0 - rate_med / 100.0), 2.0)
            sigma = math.sqrt(sigma_fit ** 2 + horizon_term ** 2
                              + (sigma_n * mult) ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), 2.0)
        med_logit = logit(rate_med)
        quantiles = []
        prev = None
        for z in Z_SCORES:
            q = sigmoid(med_logit + z * sigma)
            q = round(min(max(q, RATE_LO), RATE_HI), 4)
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def prior_row(trophy):
    """Fallback forecast row when a case has no usable history."""
    tier = trophy.get("tier", "bronze")
    med = {"platinum": 0.05, "gold": 0.3, "silver": 0.5,
           "bronze": 1.0}.get(tier, 1.0)
    base = logit(med)
    quantiles = []
    prev = None
    for z in Z_SCORES:
        q = round(min(max(sigmoid(base + z * 1.5), RATE_LO), RATE_HI), 4)
        if prev is not None and q < prev:
            q = prev
        quantiles.append(q)
        prev = q
    return {"id": trophy["id"], "q": quantiles}


def fallback_case(case):
    """Safe forecast used when the main forecast fails: last rates with
    a baseline-style random-walk spread, so the case stays valid."""
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    spread = 0.5 * math.sqrt(horizon_days)
    last_rates = {}
    if history:
        last = history[-1].get("rates") or []
        for i, rate in enumerate(last):
            if rate is not None:
                last_rates[i + 1] = rate
    rows = []
    for trophy in trophies:
        rate = last_rates.get(trophy["id"])
        if rate is None:
            rows.append(prior_row(trophy))
            continue
        base = logit(rate)
        quantiles = []
        prev = None
        for z in Z_SCORES:
            q = round(min(max(sigmoid(base + z * spread),
                              RATE_LO), RATE_HI), 4)
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir)
                   if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
