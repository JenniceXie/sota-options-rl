"""Point-in-time option IV inversion and fixed-node surface estimation.

The module is deliberately independent from SpiderRock's fitted volatility
values.  A caller may pass those values alongside the independently inverted
quote-mid observations for comparison, but they never enter the project
inversion.  European contracts use Black--Scholes with continuous carry.
American contracts use a Cox--Ross--Rubinstein tree.  SpiderRock exposes only
an aggregate discrete-dividend amount, not a dated dividend schedule, so the
v1 American implementation uses an explicitly reported escrowed-dividend
approximation.  It is OptionMetrics-like, not an exact replication of the
vendor's proprietary multi-thousand-step implementation and dividend/borrow
inputs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


OptionRight = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]

MIN_VOLATILITY = 1.0e-4
MAX_VOLATILITY = 5.0
EXTREME_MAX_VOLATILITY = 20.0

#: Days a reported theta is divided into.  252 and not 365, so that theta and
#: the one-sigma move it is netted against are measured over the same day.
#: ``SizeResolver._scenario_limit`` takes the daily move as ``sigma/sqrt(252)``
#: and then adds ``-theta``; under a 365-day theta the two terms are 1.448x
#: apart and the gamma/theta cancellation that holds for a delta-hedged book
#: (``theta_day = -0.5*gamma*(sigma_day*S)**2``) leaves 31% of the gamma term
#: standing -- a standing credit to long gamma and charge to short gamma that
#: no risk view intended.  It also matches the SpiderRock ``prtTh`` column that
#: sits beside it in the chain: measured over 32,608 clean rows on five dates,
#: ``vendor/model`` was 1.4528 against 365/252 = 1.4484.
#:
#: The cost of the choice is that "one day" is now 1/252 of a *calendar* year
#: -- ``time_to_expiry`` is ACT/365 -- so this theta is the decay over one
#: trading day only in the business-time reading where sigma is a trading-year
#: vol.  That reading is the one the scenario uses, so the pair is coherent;
#: read as calendar decay it is 1.448x too large, which is why the emitted
#: ``model_theta_units`` names the convention instead of leaving it to be
#: guessed.
THETA_DAYS_PER_YEAR = 252.0

#: The string every builder stamps into ``model_theta_units``.  It lives beside
#: the divisor because the two must move together: ``env/chain.py`` converts a
#: stored theta on the strength of this label, so a label that lags the divisor
#: is worse than no label at all.
MODEL_THETA_UNITS = "option_price_per_trading_day"


@dataclass(frozen=True, slots=True)
class PriceDelta:
    """Option value and first two spot derivatives under one pricing model."""

    price: float
    delta: float
    gamma: float | None = None


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityResult:
    """Result of a bounded implied-volatility inversion."""

    implied_volatility: float | None
    delta: float | None
    status: str
    iterations: int
    model_price_error: float | None
    gamma: float | None = None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _right(value: str) -> OptionRight:
    normalized = value.strip().lower()
    if normalized in {"c", "call"}:
        return "call"
    if normalized in {"p", "put"}:
        return "put"
    raise ValueError(f"unsupported option right: {value!r}")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _escrowed_spot(
    spot: float,
    *,
    discrete_dividend: float = 0.0,
) -> float:
    """Return spot net of the vendor's aggregate discrete-dividend amount.

    A dated dividend schedule is required for an exact early-exercise model.
    The normalized SpiderRock print has only one ``ddiv`` scalar.  Treating it
    as an escrowed present value is transparent and deterministic, while the
    output schema records this approximation for downstream cross-checking.
    """

    return max(spot - max(discrete_dividend, 0.0), 1.0e-12)


def black_scholes_price_delta(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_right: str,
    discrete_dividend: float = 0.0,
) -> PriceDelta:
    """Price a European option and return its spot delta."""

    right = _right(option_right)
    if min(spot, strike) <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0:
        intrinsic = (
            max(spot - strike, 0.0)
            if right == "call"
            else max(strike - spot, 0.0)
        )
        if right == "call":
            delta = 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        else:
            delta = -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)
        return PriceDelta(intrinsic, delta, None)
    if volatility <= 0:
        raise ValueError("volatility must be positive")

    adjusted_spot = _escrowed_spot(
        spot,
        discrete_dividend=discrete_dividend,
    )
    root_time = math.sqrt(time_to_expiry)
    denominator = volatility * root_time
    d1 = (
        math.log(adjusted_spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_expiry
    ) / denominator
    d2 = d1 - denominator
    spot_discount = math.exp(-dividend_yield * time_to_expiry)
    strike_discount = math.exp(-rate * time_to_expiry)
    if right == "call":
        price = (
            adjusted_spot * spot_discount * _normal_cdf(d1)
            - strike * strike_discount * _normal_cdf(d2)
        )
        delta = spot_discount * _normal_cdf(d1)
    else:
        price = (
            strike * strike_discount * _normal_cdf(-d2)
            - adjusted_spot * spot_discount * _normal_cdf(-d1)
        )
        delta = -spot_discount * _normal_cdf(-d1)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    gamma = spot_discount * density / (adjusted_spot * denominator)
    return PriceDelta(max(price, 0.0), delta, gamma)


def crr_american_price_delta(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_right: str,
    discrete_dividend: float = 0.0,
    steps: int = 256,
) -> PriceDelta:
    """Price an American option with a CRR tree and early exercise.

    ``discrete_dividend`` is handled with the documented escrowed-dividend
    approximation because the input record lacks projected ex-dividend dates.
    Increase ``steps`` for convergence studies and OptionMetrics cross-checks.
    """

    right = _right(option_right)
    if min(spot, strike) <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0:
        return black_scholes_price_delta(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=max(volatility, MIN_VOLATILITY),
            option_right=right,
            discrete_dividend=discrete_dividend,
        )
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if steps < 2:
        raise ValueError("CRR steps must be at least two")

    adjusted_spot = _escrowed_spot(
        spot,
        discrete_dividend=discrete_dividend,
    )
    dt = time_to_expiry / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    denominator = up - down
    if denominator <= 0:
        raise ValueError("degenerate CRR tree")
    # Above this the tree's top node is genuinely not a double, not merely
    # badly factored, and there is nothing to compute.  It is a ``ValueError``
    # and not an ``OverflowError`` so that it joins the two degeneracies above
    # as something the volatility solver already catches while it widens its
    # bracket: the alternative is an exception class the caller does not handle,
    # which is what took out a whole day's chain rather than one contract's IV.
    if math.log(adjusted_spot) + steps * volatility * math.sqrt(dt) > 709.0:
        raise ValueError("CRR terminal spot overflows at this volatility")
    growth = math.exp((rate - dividend_yield) * dt)
    probability = (growth - down) / denominator
    if not 0.0 <= probability <= 1.0:
        raise ValueError("CRR risk-neutral probability is outside [0, 1]")
    discount = math.exp(-rate * dt)

    # The node spot is written as one power of ``up`` rather than as a low node
    # times ``(up/down)**index``.  The two are equal in exact arithmetic --
    # ``down == 1/up``, so ``down**steps * (up/down)**index == up**(2*index -
    # steps)`` -- but the factored form computes a huge intermediate and then
    # multiplies it by a tiny one, and the intermediate overflows a double long
    # before the product it belongs to does.
    #
    # ``(up/down)**steps`` is ``exp(2*sigma*sqrt(T*steps))``, which passes 709 at
    # ``sigma*sqrt(T) > 31.4`` -- reachable at ``steps=128`` by any contract that
    # is both longer than 2.46 years and priced in the extreme bracket, since
    # ``EXTREME_MAX_VOLATILITY`` is 20.  The top terminal spot itself is
    # ``exp(sigma*sqrt(T*steps))``, half the exponent, and stays representable.
    # This is not a hypothetical: 8 dates of the 246-date evaluation window had
    # no chain because one Jan-2028 LEAP per date reached that bracket during the
    # solver's upper expansion, and the OverflowError killed the whole build.
    terminal_spots = [
        adjusted_spot * up ** (2 * index - steps) for index in range(steps + 1)
    ]
    if right == "call":
        values = [max(value - strike, 0.0) for value in terminal_spots]
    else:
        values = [max(strike - value, 0.0) for value in terminal_spots]

    first_step_values: tuple[float, float] | None = None
    second_step_values: tuple[float, float, float] | None = None
    for level in range(steps - 1, -1, -1):
        next_values: list[float] = []
        for index in range(level + 1):
            continuation = discount * (
                probability * values[index + 1]
                + (1.0 - probability) * values[index]
            )
            node_spot = adjusted_spot * up ** (2 * index - level)
            exercise = (
                max(node_spot - strike, 0.0)
                if right == "call"
                else max(strike - node_spot, 0.0)
            )
            next_values.append(max(continuation, exercise))
        values = next_values
        if level == 2:
            second_step_values = (values[0], values[1], values[2])
        if level == 1:
            first_step_values = (values[0], values[1])

    if first_step_values is None or second_step_values is None:
        raise RuntimeError("CRR tree did not produce Greek-estimation values")
    first_down = adjusted_spot * down
    first_up = adjusted_spot * up
    delta = (first_step_values[1] - first_step_values[0]) / (
        first_up - first_down
    )
    second_down = adjusted_spot * down * down
    second_middle = adjusted_spot
    second_up = adjusted_spot * up * up
    delta_down = (second_step_values[1] - second_step_values[0]) / (
        second_middle - second_down
    )
    delta_up = (second_step_values[2] - second_step_values[1]) / (
        second_up - second_middle
    )
    gamma = (delta_up - delta_down) / (
        0.5 * (second_up - second_down)
    )
    return PriceDelta(values[0], delta, gamma)


def implied_volatility_from_midpoint(
    *,
    midpoint: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_right: str,
    exercise_style: ExerciseStyle,
    discrete_dividend: float = 0.0,
    american_steps: int = 256,
    minimum_volatility: float = MIN_VOLATILITY,
    maximum_volatility: float = MAX_VOLATILITY,
    extreme_maximum_volatility: float = EXTREME_MAX_VOLATILITY,
    price_tolerance: float = 1.0e-6,
    max_iterations: int = 36,
) -> ImpliedVolatilityResult:
    """Invert a causal quote midpoint with adaptive bounded bisection.

    ``maximum_volatility`` is the ordinary numerical bracket, not an
    economic price bound.  If the quote requires a larger volatility, the
    bracket expands up to ``extreme_maximum_volatility`` and a solved value is
    returned with status ``extreme_iv``.  Quotes still above the expanded
    model bracket remain unresolved instead of being mislabeled as ordinary
    invalid inputs.
    """

    if midpoint <= 0 or min(spot, strike, time_to_expiry) <= 0:
        return ImpliedVolatilityResult(None, None, "invalid_input", 0, None)
    if exercise_style not in {"european", "american"}:
        raise ValueError(f"unsupported exercise style: {exercise_style!r}")
    if not 0 < minimum_volatility < maximum_volatility:
        raise ValueError("invalid ordinary volatility bracket")
    if extreme_maximum_volatility < maximum_volatility:
        raise ValueError("extreme maximum volatility must cover ordinary maximum")

    def price(volatility: float) -> PriceDelta:
        kwargs = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "rate": rate,
            "dividend_yield": dividend_yield,
            "volatility": volatility,
            "option_right": option_right,
            "discrete_dividend": discrete_dividend,
        }
        if exercise_style == "european":
            return black_scholes_price_delta(**kwargs)
        return crr_american_price_delta(**kwargs, steps=american_steps)

    low = minimum_volatility
    low_value: PriceDelta | None = None
    while low < maximum_volatility:
        try:
            low_value = price(low)
            break
        except ValueError:
            # A very small CRR volatility can imply an invalid one-step
            # risk-neutral probability.  Raise the numerical lower bracket;
            # do not clamp the probability and silently change the model.
            low *= 2.0
    if low_value is None:
        return ImpliedVolatilityResult(None, None, "pricing_error", 0, None)
    if midpoint < low_value.price - price_tolerance:
        return ImpliedVolatilityResult(
            None,
            None,
            "below_model_lower_bound",
            0,
            midpoint - low_value.price,
        )
    high = maximum_volatility
    try:
        high_value = price(high)
    except ValueError:
        return ImpliedVolatilityResult(None, None, "pricing_error", 0, None)
    while (
        midpoint > high_value.price + price_tolerance
        and high < extreme_maximum_volatility
    ):
        high = min(high * 2.0, extreme_maximum_volatility)
        try:
            high_value = price(high)
        except ValueError:
            return ImpliedVolatilityResult(None, None, "pricing_error", 0, None)
    if midpoint > high_value.price + price_tolerance:
        return ImpliedVolatilityResult(
            None,
            None,
            "above_extreme_volatility_cap",
            0,
            midpoint - high_value.price,
        )

    extreme = high > maximum_volatility
    result = low_value
    for iteration in range(1, max_iterations + 1):
        guess = 0.5 * (low + high)
        try:
            result = price(guess)
        except ValueError:
            return ImpliedVolatilityResult(
                None,
                None,
                "pricing_error",
                iteration,
                None,
            )
        error = result.price - midpoint
        if abs(error) <= price_tolerance or high - low <= 1.0e-7:
            return ImpliedVolatilityResult(
                guess,
                result.delta,
                "extreme_iv" if extreme and guess > maximum_volatility else "ok",
                iteration,
                error,
                result.gamma,
            )
        if error > 0:
            high = guess
        else:
            low = guess
    guess = 0.5 * (low + high)
    result = price(guess)
    return ImpliedVolatilityResult(
        guess,
        result.delta,
        (
            "extreme_iv"
            if extreme and guess > maximum_volatility
            else "max_iterations"
        ),
        max_iterations,
        result.price - midpoint,
        result.gamma,
    )


def model_vega(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_right: str,
    exercise_style: ExerciseStyle,
    discrete_dividend: float = 0.0,
    american_steps: int = 256,
) -> float | None:
    """Return price sensitivity per one unit of annualized volatility.

    European vega is analytic.  American vega is a central finite difference
    of the same versioned CRR model used for inversion, avoiding dependence on
    vendor-specific vega units.
    """

    if min(spot, strike, time_to_expiry, volatility) <= 0:
        return None
    if exercise_style == "european":
        adjusted_spot = _escrowed_spot(
            spot,
            discrete_dividend=discrete_dividend,
        )
        root_time = math.sqrt(time_to_expiry)
        d1 = (
            math.log(adjusted_spot / strike)
            + (rate - dividend_yield + 0.5 * volatility * volatility)
            * time_to_expiry
        ) / (volatility * root_time)
        density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        return adjusted_spot * math.exp(-dividend_yield * time_to_expiry) * density * root_time
    if exercise_style != "american":
        raise ValueError(f"unsupported exercise style: {exercise_style!r}")

    step = max(1.0e-4, min(0.02, volatility * 0.01))
    low = max(volatility - step, MIN_VOLATILITY)
    high = volatility + step
    kwargs = {
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "rate": rate,
        "dividend_yield": dividend_yield,
        "option_right": option_right,
        "discrete_dividend": discrete_dividend,
        "steps": american_steps,
    }
    try:
        low_price = crr_american_price_delta(volatility=low, **kwargs).price
        high_price = crr_american_price_delta(volatility=high, **kwargs).price
    except ValueError:
        return None
    value = (high_price - low_price) / (high - low)
    return value if math.isfinite(value) and value > 0 else None


def model_theta(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_right: str,
    exercise_style: ExerciseStyle,
    discrete_dividend: float = 0.0,
    american_steps: int = 256,
) -> float | None:
    """Return model theta per trading day under the selected project pricer.

    Theta is computed by moving valuation time forward while holding the other
    causal pricing inputs fixed.  European and American contracts therefore
    use the same sign and units.  The finite-difference result is deliberately
    independent of vendor Greek conventions.

    The derivative is per year; ``THETA_DAYS_PER_YEAR`` divides it into days.
    See that constant for why the divisor is 252 and what it costs.
    """

    if min(spot, strike, time_to_expiry, volatility) <= 0:
        return None
    if exercise_style not in {"european", "american"}:
        raise ValueError(f"unsupported exercise style: {exercise_style!r}")

    # Bump width, not a reporting unit: ``time_to_expiry`` is ACT/365, so one
    # calendar day is the natural symmetric interval, and the division below by
    # ``upper - lower`` recovers a per-year derivative whatever width is used.
    # The derivative is with respect to elapsed time, so it is the negative
    # derivative with respect to remaining maturity.
    maturity_step = min(1.0 / 365.0, time_to_expiry * 0.25)
    if maturity_step <= 0:
        return None

    def price(maturity: float) -> float:
        kwargs = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": maturity,
            "rate": rate,
            "dividend_yield": dividend_yield,
            "volatility": volatility,
            "option_right": option_right,
            "discrete_dividend": discrete_dividend,
        }
        if exercise_style == "european":
            return black_scholes_price_delta(**kwargs).price
        return crr_american_price_delta(
            **kwargs,
            steps=american_steps,
        ).price

    lower_maturity = max(time_to_expiry - maturity_step, 0.0)
    upper_maturity = time_to_expiry + maturity_step
    try:
        lower_price = price(lower_maturity)
        upper_price = price(upper_maturity)
    except ValueError:
        return None
    annual_theta = (lower_price - upper_price) / (
        upper_maturity - lower_maturity
    )
    daily_theta = annual_theta / THETA_DAYS_PER_YEAR
    return daily_theta if math.isfinite(daily_theta) else None


def weighted_median(values: Sequence[tuple[float, float]]) -> float | None:
    """Return a deterministic weighted median of finite positive weights."""

    clean = sorted(
        (value, weight)
        for value, weight in values
        if math.isfinite(value) and math.isfinite(weight) and weight > 0
    )
    if not clean:
        return None
    total = sum(weight for _, weight in clean)
    threshold = 0.5 * total
    cumulative = 0.0
    for value, weight in clean:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return clean[-1][0]


def _solve_three_by_three(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float] | None:
    augmented = [
        list(row) + [float(value)]
        for row, value in zip(matrix, vector, strict=True)
    ]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(augmented[row][3] for row in range(3))  # type: ignore[return-value]


def fit_atm_total_variance(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, str]:
    """Fit local quadratic total variance and evaluate it at forward ATM.

    Required fields are ``log_forward_moneyness``, ``time_to_expiry``,
    ``implied_volatility``, and ``weight``.  A vega/spread/recency weighting
    policy belongs to the caller because those values are source-specific.
    """

    points: list[tuple[float, float, float, float, float]] = []
    for row in observations:
        k = _finite(row.get("log_forward_moneyness"))
        maturity = _finite(row.get("time_to_expiry"))
        volatility = _finite(row.get("implied_volatility"))
        weight = _finite(row.get("weight"))
        if (
            k is None
            or maturity is None
            or maturity <= 0
            or volatility is None
            or volatility <= 0
            or weight is None
            or weight <= 0
        ):
            continue
        points.append(
            (k, volatility * volatility * maturity, weight, maturity, volatility)
        )
    if (
        len(points) < 3
        or not any(k <= 0 for k, _, _, _, _ in points)
        or not any(k >= 0 for k, _, _, _, _ in points)
    ):
        fallback = weighted_median(
            [
                (volatility, weight)
                for k, _, weight, _, volatility in points
                if abs(k) <= 0.10
            ]
        )
        maturity_values = [maturity for _, _, _, maturity, _ in points]
        maturity = statistics.median(maturity_values) if maturity_values else None
        if fallback is None or maturity is None:
            return None, None, "insufficient_atm_support"
        return fallback * fallback * maturity, None, "weighted_median_fallback"

    sums = [0.0] * 5
    rhs = [0.0] * 3
    for k, total_variance, weight, _, _ in points:
        powers = (1.0, k, k * k, k * k * k, k**4)
        for index in range(5):
            sums[index] += weight * powers[index]
        rhs[0] += weight * total_variance
        rhs[1] += weight * k * total_variance
        rhs[2] += weight * k * k * total_variance
    coefficients = _solve_three_by_three(
        (
            (sums[0], sums[1], sums[2]),
            (sums[1], sums[2], sums[3]),
            (sums[2], sums[3], sums[4]),
        ),
        rhs,
    )
    if coefficients is None or coefficients[0] <= 0:
        return None, None, "singular_surface_fit"
    weighted_squared_error = 0.0
    total_weight = 0.0
    for k, total_variance, weight, _, _ in points:
        fitted = coefficients[0] + coefficients[1] * k + coefficients[2] * k * k
        weighted_squared_error += weight * (total_variance - fitted) ** 2
        total_weight += weight
    rmse = math.sqrt(weighted_squared_error / total_weight) if total_weight else None
    return coefficients[0], rmse, "quadratic_total_variance"


def interpolate_delta_iv(
    observations: Sequence[Mapping[str, Any]],
    *,
    option_right: str,
    target_absolute_delta: float = 0.25,
    nearest_tolerance: float = 0.05,
) -> tuple[float | None, str]:
    """Interpolate independently inverted IV in absolute-delta coordinates."""

    right = _right(option_right)
    points: list[tuple[float, float]] = []
    for row in observations:
        if _right(str(row.get("option_right") or "")) != right:
            continue
        delta = _finite(row.get("delta"))
        volatility = _finite(row.get("implied_volatility"))
        if delta is None or volatility is None or volatility <= 0:
            continue
        points.append((abs(delta), volatility))
    points.sort()
    if not points:
        return None, "missing_delta_support"
    below = [item for item in points if item[0] <= target_absolute_delta]
    above = [item for item in points if item[0] >= target_absolute_delta]
    if below and above:
        lower_delta, lower_iv = below[-1]
        upper_delta, upper_iv = above[0]
        if upper_delta == lower_delta:
            return lower_iv, "exact_delta"
        fraction = (target_absolute_delta - lower_delta) / (
            upper_delta - lower_delta
        )
        return lower_iv + fraction * (upper_iv - lower_iv), "interpolated_delta"
    nearest_delta, nearest_iv = min(
        points,
        key=lambda item: abs(item[0] - target_absolute_delta),
    )
    if abs(nearest_delta - target_absolute_delta) <= nearest_tolerance:
        return nearest_iv, "nearest_delta_fallback"
    return None, "insufficient_delta_support"


def interpolate_total_variance(
    points: Sequence[tuple[float, float]],
    *,
    target_time: float,
) -> tuple[float | None, str]:
    """Interpolate volatility by linearly interpolating total variance."""

    clean = sorted(
        (maturity, volatility)
        for maturity, volatility in points
        if maturity > 0
        and volatility > 0
        and math.isfinite(maturity)
        and math.isfinite(volatility)
    )
    if not clean or target_time <= 0:
        return None, "missing_maturity_support"
    exact = min(clean, key=lambda item: abs(item[0] - target_time))
    if abs(exact[0] - target_time) <= 1.0 / 3650.0:
        return exact[1], "exact_maturity"
    lower = [item for item in clean if item[0] < target_time]
    upper = [item for item in clean if item[0] > target_time]
    if not lower or not upper:
        return None, "missing_bracketing_expiry"
    left_t, left_iv = lower[-1]
    right_t, right_iv = upper[0]
    left_variance = left_iv * left_iv * left_t
    right_variance = right_iv * right_iv * right_t
    fraction = (target_time - left_t) / (right_t - left_t)
    target_variance = left_variance + fraction * (right_variance - left_variance)
    if target_variance <= 0:
        return None, "nonpositive_interpolated_variance"
    return math.sqrt(target_variance / target_time), "interpolated_total_variance"
