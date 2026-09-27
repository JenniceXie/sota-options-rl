"""Fit the conventional baselines' parameters, strictly before the test window.

Writes one artifact --- ``configs/baselines/params.json`` --- holding everything
``env/policy/baselines.py`` needs and nothing it could refit at run time: a
pooled GARCH(1,1), a per-date causal volatility forecast, and the tercile
thresholds for the four state signals.

**The whole point of this script is the causality claim, so it is asserted and
not documented.**  ``--test-start`` splits the world.  Every quantity written
here is computed from dates strictly before it, and the script refuses to write
if that is violated.  The artifact carries the fit window, the test start and the
calibration date count, and ``load_params`` refuses an artifact that omits them.

Three specification choices, each forced by a measurement rather than a taste,
and each recorded in the artifact so a reader does not have to trust this file:

**Returns are close-to-close, reconstructed from two legs.**
``underlying_market_features`` carries ``spot_return_step`` twice per date with
``spot_return_interval`` telling you which half it is: ``previous_close_to_open``
at the AM row and ``open_to_close`` at the PM row.  The PM row alone is the
*intraday* return, and fitting on it understates annualised volatility by
1.2-1.5x per name because it drops every overnight gap.  The close-to-close
return is the sum, which reproduces ``ln(close_t / close_{t-1})`` exactly.

**Returns are winsorised at 4 unconditional standard deviations.**  Not
cosmetic.  Unwinsorised, Gaussian QML on this panel returns a pooled persistence
of 0.478 --- a 0.9-day half-life --- because earnings gaps give kurtosis up to
20 (META) and the Gaussian likelihood cannot represent a jump, so it pushes beta
to zero and charges the variance to noise.  At that persistence a 10-day
forecast equals the unconditional variance to three decimals, and the "GARCH
policy" would silently become a per-name constant.  Winsorising restores 0.822,
a 3.5-day half-life, and leaves 14% of a shock alive at the forecast horizon.

**Dynamics are pooled; only the variance level is per name.**  Per-name MLE is
degenerate on this sample: MSFT and META both return beta = 0, which would make
their forecasts constants for the same reason as above.  One shared ``(alpha,
beta)`` with per-name variance targeting is two parameters plus ten scales, all
estimated pre-test, and it does not collapse for any name.

The surface dataset is ``option_delta_point_surface``, because that is what
``statespace_v1.DATASET_SURFACE`` reads.  ``option_iv_surface_features`` carries
columns of the same names and different values --- on one spot-checked cell,
``butterfly_25d`` differs by 34% and ``term_slope`` by 30% --- so calibrating
against it would put the thresholds on a quantity the policy never sees.

Usage:
    python scripts/analysis/fit_baseline_params.py configs/baselines/params.json \
        --data-root "$PM_DATA_ROOT" \
        --calibration-dates configs/dates_sft.txt configs/dates_rl.txt \
        --test-dates configs/dates_eval.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date as _date
from pathlib import Path
from typing import Any

WINSORIZE_SD = 4.0
#: Trading days matching the ``8_30`` bucket's 14-DTE anchor.
HORIZON_TRADING_DAYS = 10
TRADING_DAYS_PER_YEAR = 252.0

SURFACE = "option_delta_point_surface"
EQUITY = "underlying_market_features"
CLOSE_STEP = "market_close"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read_jsonl(root: Path, dataset: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    base = root / "features" / dataset
    if not base.is_dir():
        raise SystemExit(f"no such dataset directory: {base}")
    for part in sorted(base.glob("date=*/part-*.jsonl")):
        day = part.parent.name.split("=", 1)[1]
        rows = out.setdefault(day, [])
        with part.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return out


def close_to_close_returns(equity: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """``{ticker: {date: log return}}``, summed over the two half-day legs."""
    legs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for day, rows in equity.items():
        for row in rows:
            interval = row.get("spot_return_interval")
            value = row.get("spot_return_step")
            if interval is None or value is None:
                continue
            legs[(row["underlying"], day)][interval] = float(value)
    panel: dict[str, dict[str, float]] = defaultdict(dict)
    dropped = Counter()
    for (ticker, day), found in legs.items():
        if "previous_close_to_open" in found and "open_to_close" in found:
            panel[ticker][day] = found["previous_close_to_open"] + found["open_to_close"]
        else:
            dropped[ticker] += 1
    if dropped:
        # Reported, never silently imputed: a half-day return standing in for a
        # full one is a 1.3x error in the fitted variance level.
        print(f"  dropped dates missing a return leg: {dict(dropped)}")
    return dict(panel)


# ---------------------------------------------------------------------------
# GARCH(1,1), Gaussian QML with variance targeting
# ---------------------------------------------------------------------------

def _loglik(z: list[float], alpha: float, beta: float) -> float:
    """On unit-variance data, so the targeted omega is ``1 - alpha - beta``."""
    omega = 1.0 - alpha - beta
    if omega <= 0.0 or alpha <= 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        return -math.inf
    h = 1.0
    total = 0.0
    for x in z:
        total += -0.5 * (math.log(2.0 * math.pi) + math.log(h) + x * x / h)
        h = omega + alpha * x * x + beta * h
    return total


def _grid(objective, a_range, b_range, step) -> tuple[float, float, float]:
    """A deterministic grid, because there is no scipy here and no need for one.

    Reproducibility is worth more than the last decimal of the optimum: the same
    inputs give the same parameters on any machine, with no optimiser seed or
    convergence tolerance in the provenance.
    """
    best = (-math.inf, 0.0, 0.0)
    a = a_range[0]
    while a <= a_range[1] + 1e-12:
        b = b_range[0]
        while b <= b_range[1] + 1e-12:
            if a > 0.0 and b >= 0.0 and a + b < 0.9995:
                value = objective(a, b)
                if value > best[0]:
                    best = (value, a, b)
            b += step
        a += step
    return best


def fit_pooled_garch(panel: dict[str, dict[str, float]], test_start: str) -> dict[str, Any]:
    scales: dict[str, dict[str, float]] = {}
    standardised: dict[str, list[float]] = {}
    for ticker, series in panel.items():
        pre = [v for day, v in sorted(series.items()) if day < test_start]
        if len(pre) < 60:
            raise SystemExit(
                f"{ticker}: only {len(pre)} returns before {test_start}; refusing to "
                "fit a volatility model on that. Widen the history or drop the name."
            )
        mu = sum(pre) / len(pre)
        centred = [v - mu for v in pre]
        sd = math.sqrt(sum(v * v for v in centred) / len(centred))
        scales[ticker] = {"mu": mu, "sd": sd, "sigma2_bar": sd * sd,
                          "ann_uncond_vol": math.sqrt(TRADING_DAYS_PER_YEAR) * sd,
                          "n_fit": len(pre)}
        standardised[ticker] = [
            max(-WINSORIZE_SD, min(WINSORIZE_SD, v / sd)) for v in centred
        ]

    def pooled(alpha: float, beta: float) -> float:
        return sum(_loglik(z, alpha, beta) for z in standardised.values())

    loglik, alpha, beta = _grid(pooled, (0.01, 0.40), (0.00, 0.98), 0.01)
    loglik, alpha, beta = _grid(
        pooled, (max(0.002, alpha - 0.02), alpha + 0.02),
        (max(0.0, beta - 0.02), beta + 0.02), 0.002,
    )
    persistence = alpha + beta
    return {
        "specification": "GARCH(1,1), Gaussian QML, variance targeting; pooled "
                         "(alpha, beta), per-name unconditional variance",
        "returns": "close-to-close log return = previous_close_to_open + open_to_close",
        "winsorize_sd": WINSORIZE_SD,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "persistence": round(persistence, 6),
        "shock_half_life_days": round(math.log(0.5) / math.log(persistence), 3),
        "shock_surviving_at_horizon": round(persistence ** HORIZON_TRADING_DAYS, 4),
        "loglik": loglik,
        "n_obs": sum(len(z) for z in standardised.values()),
        "per_name": scales,
    }


def forecast_table(panel: dict[str, dict[str, float]], garch: dict[str, Any]) -> dict[str, float]:
    """Annualised volatility forecast over the horizon, keyed ``date|ticker``.

    The recursion at date ``t`` has consumed returns up to and including ``t``,
    and the forecast is for ``t+1 .. t+H``.  At a close-only decision grid that
    is causal: the PM decision happens at the close that produced ``ret_t``.
    """
    alpha = garch["alpha"]
    beta = garch["beta"]
    persistence = alpha + beta
    horizon = garch["horizon_trading_days"]
    out: dict[str, float] = {}
    for ticker, series in panel.items():
        scale = garch["per_name"][ticker]
        sigma2 = scale["sigma2_bar"]
        mu = scale["mu"]
        sd = scale["sd"]
        omega = sigma2 * (1.0 - alpha - beta)
        bound = WINSORIZE_SD * sd
        h = sigma2
        for day, value in sorted(series.items()):
            eps = max(-bound, min(bound, value - mu))
            h_next = omega + alpha * eps * eps + beta * h
            mean_var = sigma2 + (h_next - sigma2) * (
                1.0 - persistence ** horizon
            ) / (horizon * (1.0 - persistence))
            out[f"{day}|{ticker}"] = math.sqrt(TRADING_DAYS_PER_YEAR * mean_var)
            h = h_next
    return out


# ---------------------------------------------------------------------------
# signal thresholds
# ---------------------------------------------------------------------------

def _quantile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * frac
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def state_signals(
    surface: dict[str, list[dict[str, Any]]],
    equity: dict[str, list[dict[str, Any]]],
    forecast: dict[str, float],
    days: set[str],
) -> dict[str, list[float]]:
    """The four state signals plus the GARCH ratio, exactly as the policy reads them.

    ``w`` is recomputed as ``iv - rv`` from the same two source columns
    ``statespace_v1._market_fields`` differences, rather than read from anywhere
    else, so the threshold sits on the quantity that reaches the wire.
    """
    rv: dict[tuple[str, str], float] = {}
    for day, rows in equity.items():
        for row in rows:
            if row.get("step") != CLOSE_STEP:
                continue
            value = row.get("realized_vol_21d")
            if value is not None:
                rv[(day, row["underlying"])] = float(value)

    out: dict[str, list[float]] = defaultdict(list)
    for day, rows in surface.items():
        if day not in days:
            continue
        for row in rows:
            if row.get("step") != CLOSE_STEP:
                continue
            ticker = row["underlying"]
            iv = row.get("atm_iv_30d")
            realized = rv.get((day, ticker))
            if iv is not None and realized is not None:
                out["wedge"].append(float(iv) - realized)
            if iv is not None:
                g = forecast.get(f"{day}|{ticker}")
                if g and g > 0.0:
                    out["garch_ratio"].append(float(iv) / g)
            for key, column in (("term_slope", "term_slope"),
                                ("skew", "skew_25d"),
                                ("butterfly", "butterfly_25d")):
                value = row.get(column)
                if value is not None:
                    out[key].append(float(value))
    return dict(out)


# ---------------------------------------------------------------------------

def _read_dates(paths: list[Path]) -> list[str]:
    days: list[str] = []
    for path in paths:
        days.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return sorted(set(days))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-dates", type=Path, nargs="+", required=True,
                        help="train and validation date files; thresholds use these only")
    parser.add_argument("--test-dates", type=Path, required=True,
                        help="held out; used ONLY to locate the boundary and to assert it")
    args = parser.parse_args(argv)

    calibration = _read_dates(args.calibration_dates)
    test = _read_dates([args.test_dates])
    test_start = min(test)

    overlap = sorted(set(calibration) & set(test))
    if overlap:
        raise SystemExit(
            f"calibration and test dates overlap on {len(overlap)} dates "
            f"(first {overlap[:3]}). Refusing: every threshold would be "
            "partly fitted on the sample it is reported on."
        )
    late = [d for d in calibration if d >= test_start]
    if late:
        raise SystemExit(
            f"{len(late)} calibration dates fall on or after the test start "
            f"{test_start} (first {late[:3]}). Refusing."
        )

    print(f"calibration {len(calibration)} dates {calibration[0]} .. {calibration[-1]}")
    print(f"test        {len(test)} dates {test[0]} .. {test[-1]}")

    equity = _read_jsonl(args.data_root, EQUITY)
    surface = _read_jsonl(args.data_root, SURFACE)
    print(f"read {EQUITY}: {len(equity)} dates | {SURFACE}: {len(surface)} dates")

    panel = close_to_close_returns(equity)
    fit_days = sorted({d for s in panel.values() for d in s if d < test_start})
    if not fit_days:
        raise SystemExit("no return dates before the test start")
    garch = fit_pooled_garch(panel, test_start)
    print(f"GARCH pooled alpha={garch['alpha']} beta={garch['beta']} "
          f"persistence={garch['persistence']} half-life={garch['shock_half_life_days']}d "
          f"surviving@{garch['horizon_trading_days']}d={garch['shock_surviving_at_horizon']}")

    forecast = forecast_table(panel, garch)
    signals = state_signals(surface, equity, forecast, set(calibration))
    thresholds = {}
    print(f"\n{'signal':12s} {'n':>6s} {'p33 (lo)':>11s} {'p50':>11s} {'p67 (hi)':>11s}")
    for name in ("garch_ratio", "wedge", "term_slope", "skew", "butterfly"):
        values = signals.get(name) or []
        if len(values) < 100:
            raise SystemExit(f"signal {name!r} has only {len(values)} calibration "
                             "observations; refusing to set a tercile on that")
        thresholds[name] = {"lo": round(_quantile(values, 1 / 3), 8),
                            "med": round(_quantile(values, 0.5), 8),
                            "hi": round(_quantile(values, 2 / 3), 8),
                            "n": len(values)}
        print(f"{name:12s} {len(values):6d} {thresholds[name]['lo']:11.6f} "
              f"{thresholds[name]['med']:11.6f} {thresholds[name]['hi']:11.6f}")

    payload = {
        "artifact": "baseline_params.v1",
        "built": _date.today().isoformat(),
        "thresholds": thresholds,
        "garch": garch,
        "forecast": forecast,
        "provenance": {
            "fit_window": [fit_days[0], fit_days[-1]],
            "test_start": test_start,
            "calibration_dates": len(calibration),
            "calibration_window": [calibration[0], calibration[-1]],
            "test_dates": len(test),
            "surface_dataset": SURFACE,
            "equity_dataset": EQUITY,
            "threshold_rule": "pooled terciles (p33 / p67) over calibration dates x names",
            "note": "Thresholds and GARCH parameters use calibration dates only. "
                    "The forecast table spans every date, but the forecast for date t "
                    "consumes returns up to and including t and no further.",
        },
    }

    # The assertion the whole script exists for.
    assert all(d < test_start for d in fit_days), "fit window leaked past the test start"
    bad = [d for d in calibration if d >= test_start]
    assert not bad, bad

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB, "
          f"{len(forecast)} forecast cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
