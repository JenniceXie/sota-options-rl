"""Algorithmic policies: rules that read the same bytes the LLM reads.

These exist to answer one question the LLM arms cannot answer on their own.
Every policy arm run so far has ended the window below the hold arm, and there
are two very different explanations for that:

1. the environment admits profitable trajectories and the policy failed to
   find them, or
2. the environment does not admit them at all -- the cost model, the size
   resolver, the tenor ladder and the fill convention together make the
   expected after-cost return of *any* package negative, and the LLM was
   playing a game it could not win.

Explanation 2 is not a hypothesis anyone should carry into a GRPO run.  If the
reward is negative for every reachable action, gradient ascent on it selects
for abstaining and nothing else, and the entire training signal is a study of
the cost model.  So this module runs the environment against rules that are not
learned at all, sweeps them, and reports whether *any* trajectory clears zero.

**These are baselines, not proposals.**  Every rule here is a two-line
heuristic evaluated on the same window it is reported on, which means a rule
that wins does not generalise and is not evidence about that rule.  What it is
evidence about is the environment: a positive trajectory existing is a
statement about the reachable set, and that statement survives the fact that
the rule was chosen with hindsight.  A rule that loses is the stronger result,
because it is not selected on.

Three deliberate constraints:

**The rules read the rendered observation, not the book.**  It would be easier
to hand a rule the ``StepContext``, and it would also make the comparison
meaningless: the rule would be acting on fields the LLM never saw, at a
precision the wire format does not carry.  Decoding the observation back to
floats is the only way "the rule beat the model" is a sentence about policies
rather than about information.  The scale constants are *imported* from
``statespace_v1`` rather than restated, so a change to the encoding moves both
sides at once instead of silently desyncing the rule from the prompt.

**Size is still the resolver's.**  A rule that sized its own positions would be
testing a different environment.  Every order here goes out at the family
default coordinates and lets ``SizeResolver`` decide the count, exactly as an
LLM order does.

**The clock is the policy's own.**  ``reset`` clears the episode context for an
LLM; for these it clears nothing, because a rule that holds for ten steps has
to count steps across a month boundary the book itself crosses.  This makes the
rules *less* memoryless than the LLM arms, which is a difference to state, not
to hide.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..spec import EnvConfig
from ..statespace import Observation
from ..statespace_v1 import (
    MISSING,
    SCALE_DP,
    SCALE_HDOLLAR,
    SCALE_KDOLLAR,
    SCALE_PCT,
)
from . import PolicyResponse

__all__ = [
    "MarketRow",
    "AccountRow",
    "PositionRow",
    "ObservedState",
    "read_observation",
    "RuleParams",
    "RulePolicy",
    "SIGNALS",
]

#: Column order of the ``M`` row, and the scale each column was divided by.
#: Mirrors ``StateSpaceV1._market_fields``; ``tests/test_rule_policies.py``
#: asserts a round trip rather than trusting the mirror.
_MARKET_COLUMNS: tuple[tuple[str, float], ...] = (
    ("ret", SCALE_DP),
    ("rv", SCALE_DP),
    ("iv", SCALE_DP),
    ("dv", SCALE_DP),
    ("w", SCALE_DP),
    ("ts", SCALE_DP),
    ("sk", SCALE_DP),
    ("bf", SCALE_DP),
    ("fi", SCALE_PCT),
    ("doi", SCALE_PCT),
)

#: ``A nlv cash bp util rlz unrlz dd npos nD nV``.
_ACCOUNT_COLUMNS: tuple[tuple[str, float], ...] = (
    ("nlv", SCALE_KDOLLAR),
    ("cash", SCALE_KDOLLAR),
    ("bp", SCALE_KDOLLAR),
    ("util", SCALE_PCT),
    ("rlz", SCALE_HDOLLAR),
    ("unrlz", SCALE_HDOLLAR),
    ("dd", SCALE_PCT),
    ("npos", 1.0),
    ("nD", SCALE_PCT),
    ("nV", SCALE_PCT),
)


#: Tokens the anonymizer substitutes for something it judged to be a date or a
#: year (``statespace_v1.py`` prints "``<d>`` and ``<y>`` are a date and a year
#: that were removed").  They arrive with the original sign still attached, as
#: ``-<y>``.
_REDACTED = ("<d>", "<y>")


def _decode(cell: str, scale: float) -> float | None:
    """Decode one wire cell, or ``None`` where the value is not available.

    Strict about structure and lenient about *absence*, and a redaction is
    absence.  ``<y>`` is emitted by this system's own renderer, so refusing to
    read it is the parser disagreeing with the writer: every policy built on
    ``read_observation`` -- the rule arms, ``GarchVolPremiumPolicy``,
    ``EconomicRulePolicy`` -- would raise ``could not convert string to float:
    '-<y>'`` the first time it held a position whose unrealized PnL happened to
    round to something in the 1900-2100 range.  That is **2.3% of P rows and
    4.0% of A rows** in the shipped astra2 corpus, present in all 142 runs, so
    it is a crash waiting on the book rather than an exotic input.

    ``None`` and not ``0.0``: the callers already branch on ``is not None``
    (``_closes`` skips its PnL targets), whereas a zero would read as "flat"
    and silently suppress a stop.

    This does not make the underlying redaction harmless -- an unrealized PnL of
    about -$2,0xx really is gone from the state the policy sees, and that is a
    fault in the anonymizer, not here. It makes the loss survivable and visible
    instead of fatal.
    """
    if cell == MISSING:
        return None
    if cell.lstrip("+-") in _REDACTED:
        return None
    return float(cell) / scale


@dataclass(frozen=True, slots=True)
class MarketRow:
    ticker: str
    values: Mapping[str, float | None]

    def get(self, name: str) -> float | None:
        return self.values.get(name)


@dataclass(frozen=True, slots=True)
class AccountRow:
    values: Mapping[str, float | None]

    @property
    def nav(self) -> float:
        return self.values.get("nlv") or 0.0

    @property
    def n_positions(self) -> int:
        return int(self.values.get("npos") or 0)

    @property
    def utilization(self) -> float:
        return self.values.get("util") or 0.0


@dataclass(frozen=True, slots=True)
class PositionRow:
    position_id: str
    underlying: str
    family: str
    orientation: str
    dte: int
    quantity: int
    mark: float | None
    unrealized_pnl: float | None
    delta_share: float | None


@dataclass(frozen=True, slots=True)
class ObservedState:
    step_index: int
    market: Mapping[str, MarketRow]
    account: AccountRow
    positions: tuple[PositionRow, ...]
    #: From the ``T`` header, ``T <step_index> <date> <session>``.  Defaulted so
    #: that callers constructing a state positionally keep working, and carried
    #: because a policy holding a frozen per-date table -- a volatility
    #: forecast, say -- has no other way to know which row is its own.  Reading
    #: the date off the header is the only causal way to get it: it is the date
    #: the observation was rendered for.
    date: str = ""
    session: str = ""


def read_observation(observation: Observation) -> ObservedState:
    """Decode ``state_space.v1`` bytes back into numbers.

    Lenient about rows it does not recognise and strict about the ones it does:
    a truncated ``M`` row raises rather than yielding a row with nine fields,
    because a rule reading ``w`` out of the ``ts`` column would still trade and
    would still produce a plausible-looking equity curve.
    """
    step_index = -1
    date = ""
    session = ""
    market: dict[str, MarketRow] = {}
    account = AccountRow({})
    positions: list[PositionRow] = []

    for line in observation.text.splitlines():
        cells = line.split()
        if not cells:
            continue
        tag = cells[0]
        if tag == "T" and len(cells) >= 2:
            step_index = int(cells[1])
            if len(cells) >= 3:
                date = cells[2]
            if len(cells) >= 4:
                session = cells[3]
        elif tag == "M":
            ticker, *rest = cells[1:]
            if len(rest) != len(_MARKET_COLUMNS):
                raise ValueError(
                    f"market row for {ticker} has {len(rest)} fields, "
                    f"expected {len(_MARKET_COLUMNS)}: {line!r}"
                )
            market[ticker] = MarketRow(
                ticker,
                {
                    name: _decode(cell, scale)
                    for (name, scale), cell in zip(_MARKET_COLUMNS, rest, strict=True)
                },
            )
        elif tag == "A":
            rest = cells[1:]
            if len(rest) != len(_ACCOUNT_COLUMNS):
                raise ValueError(
                    f"account row has {len(rest)} fields, "
                    f"expected {len(_ACCOUNT_COLUMNS)}: {line!r}"
                )
            account = AccountRow(
                {
                    name: _decode(cell, scale)
                    for (name, scale), cell in zip(_ACCOUNT_COLUMNS, rest, strict=True)
                }
            )
        elif tag == "P" and len(cells) in (10, 11):
            # 11 cells on a *rolled* position: ``_position_cells`` appends an
            # optional trailing ``src`` (``<rolled_from>g<generation>``).  The
            # old ``== 10`` dropped those rows silently, which is the worst
            # available failure: the position stayed open and invisible, so no
            # age, dte, target or stop rule could ever close it, and
            # ``len(positions)`` under-reported against the ``npos`` cell and
            # inflated the room left for new opens.  A rolled position is still
            # a position.
            _, pid, ticker, family, orientation, dte, qty, mark, upnl, delta = cells[:10]
            positions.append(
                PositionRow(
                    position_id=pid,
                    underlying=ticker,
                    family=family,
                    orientation=orientation,
                    dte=int(dte),
                    quantity=int(qty),
                    mark=_decode(mark, SCALE_PCT),
                    unrealized_pnl=_decode(upnl, 1.0),
                    delta_share=_decode(delta, SCALE_PCT),
                )
            )

    return ObservedState(
        step_index, market, account, tuple(positions), date=date, session=session
    )


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

#: Signal names admitted by ``RuleParams.signal``.
#:
#: ``none`` is the important one and the least interesting to read.  It opens
#: on every eligible step with no view at all, which makes its return the pure
#: cost drag of the package it trades: if ``none`` on a credit family and
#: ``none`` on a debit family both lose by the same amount, the loss is the
#: fill convention and not the direction.
SIGNALS = ("none", "wedge", "wedge_short", "wedge_long", "trend", "revert", "random")


@dataclass(frozen=True, slots=True)
class RuleParams:
    """One point in the sweep.

    Thresholds are in the natural units the wire carries: ``wedge_threshold``
    and ``ret_threshold`` are in vol points and return fraction respectively,
    not in the scaled integers, so a threshold here reads the same as the
    number in ``docs/state_space.md``.
    """

    signal: str = "wedge"
    tenor: str = "8_30"
    #: Family used when the signal says vol is rich, and when it says cheap.
    short_vol_family: str = "ic n"
    long_vol_family: str = "lg n"
    #: Family used by the directional signals; orientation is chosen per step.
    directional_family: str = "dv"
    wedge_threshold: float = 0.05
    ret_threshold: float = 0.01
    #: Open at most this many packages per step, and hold at most this many.
    max_opens_per_step: int = 1
    max_positions: int = 6
    #: Close after this many *decision steps*, ``None`` to never close on age.
    hold_steps: int | None = 4
    #: Close when the contract has this many days left, whatever the age.
    min_dte: int = 3
    #: Close on unrealised PnL crossing ±x basis points of NAV.  ``None`` off.
    take_profit_bp: float | None = None
    stop_loss_bp: float | None = None
    #: Only used by ``signal='random'``; also seeds the tie-break everywhere.
    seed: int = 0
    open_probability: float = 0.5
    #: Names the rule may trade.  Empty means "whatever the config admits".
    names: tuple[str, ...] = ()

    def label(self) -> str:
        parts = [self.signal, self.tenor, f"h{self.hold_steps}", f"p{self.max_positions}"]
        if self.signal in ("wedge", "wedge_short", "wedge_long"):
            parts.append(f"w{self.wedge_threshold:g}")
        if self.signal in ("trend", "revert"):
            parts.append(f"r{self.ret_threshold:g}")
        if self.signal == "random":
            parts.append(f"s{self.seed}")
        return ".".join(parts)


_RANDOM_FAMILIES: tuple[str, ...] = (
    "ol b", "ol r", "dv b", "dv r", "cv b", "cv r", "dg b", "dg r",
    "bf b", "bf r", "ls n", "lg n", "ib n", "ic n",
)


class RulePolicy:
    """A deterministic rule over the decoded observation.

    Deterministic given ``seed``: the random arm draws from its own
    ``random.Random`` rather than the global one, so two rule arms running in
    the same process cannot perturb each other's draws.  That mattered here --
    the sweep runs dozens of arms back to back in one interpreter.
    """

    def __init__(self, params: RuleParams, config: EnvConfig) -> None:
        if params.signal not in SIGNALS:
            raise ValueError(f"unknown signal {params.signal!r}; expected one of {SIGNALS}")
        if params.tenor not in config.tenor.admitted:
            raise ValueError(f"tenor {params.tenor!r} is not admitted")
        self.params = params
        self.name = f"rule.{params.label()}"
        self._tradeable = tuple(params.names or config.universe.tradeable)
        unknown = [n for n in self._tradeable if n not in config.universe.tradeable]
        if unknown:
            raise ValueError(f"not tradeable under this config: {unknown}")
        self._rng = random.Random(params.seed)
        self._opened_at: dict[str, int] = {}

    # ``reset`` does not clear ``_opened_at``: see the module docstring.
    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        # ``tools`` is ignored for the same reason ``system`` and ``grammar``
        # are: a rule arm reads the parsed state, never the prompt, so nothing
        # here is a capability that could be silently lost.
        return None

    def act(self, observation: Observation) -> PolicyResponse:
        state = read_observation(observation)
        params = self.params

        lines = list(self._closes(state))
        closing = {line.split()[1] for line in lines}
        room = params.max_positions - (len(state.positions) - len(closing))
        if room > 0:
            lines.extend(self._opens(state, min(room, params.max_opens_per_step)))

        text = "\n".join(lines) if lines else "H"
        return PolicyResponse(text=text, model=self.name, extra={"signal": params.signal})

    # -- closing ---------------------------------------------------------

    def _closes(self, state: ObservedState) -> Sequence[str]:
        """Age, time-to-expiry and PnL targets, in that order of precedence.

        Age first because it is the only one that fires on a position that has
        not moved, and a rule with no age limit degenerates into buy-and-hold
        with extra steps -- which is a real arm, but it is ``hold_steps=None``
        and should be asked for rather than arrived at.
        """
        params = self.params
        nav = state.account.nav
        out: list[str] = []
        for position in state.positions:
            opened = self._opened_at.setdefault(position.position_id, state.step_index)
            age = state.step_index - opened
            reasons = []
            if params.hold_steps is not None and age >= params.hold_steps:
                reasons.append("age")
            if position.dte <= params.min_dte:
                reasons.append("dte")
            pnl = position.unrealized_pnl
            if pnl is not None and nav > 0:
                bp = 10_000.0 * pnl / nav
                if params.take_profit_bp is not None and bp >= params.take_profit_bp:
                    reasons.append("target")
                if params.stop_loss_bp is not None and bp <= -params.stop_loss_bp:
                    reasons.append("stop")
            if reasons:
                out.append(f"C {position.position_id}")
        return out

    # -- opening ---------------------------------------------------------

    def _opens(self, state: ObservedState, room: int) -> Sequence[str]:
        params = self.params
        held = {p.underlying for p in state.positions}
        candidates = [
            (name, self._score(state.market.get(name)))
            for name in self._tradeable
            if name not in held
        ]
        ranked = sorted(
            ((n, s) for n, s in candidates if s is not None),
            key=lambda item: -abs(item[1]),
        )

        out: list[str] = []
        for name, score in ranked[:room]:
            order = self._order_for(name, score)
            if order is not None:
                out.append(order)
        return out

    def _score(self, row: MarketRow | None) -> float | None:
        """The signal, signed.  ``None`` means "do not trade this name now".

        Sign convention is uniform and load-bearing: positive means the rule
        wants the *first* branch of its family choice (rich vol, or bullish),
        negative the second.  Magnitude is only used to rank names, never to
        size -- sizing is the resolver's.
        """
        params = self.params
        if row is None:
            return None
        if params.signal == "none":
            # Ranked at random rather than at a constant.  A constant makes the
            # sort stable, which makes ``none`` trade the alphabetically first
            # tradeable name and only that name -- a single-name arm wearing a
            # universe-wide label.
            return self._rng.random()
        if params.signal == "random":
            if self._rng.random() >= params.open_probability:
                return None
            return self._rng.random()

        if params.signal in ("wedge", "wedge_short", "wedge_long"):
            wedge = row.get("w")
            if wedge is None or abs(wedge) < params.wedge_threshold:
                return None
            if params.signal == "wedge_short" and wedge < 0:
                return None
            if params.signal == "wedge_long" and wedge > 0:
                return None
            return wedge

        ret = row.get("ret")
        if ret is None or abs(ret) < params.ret_threshold:
            return None
        return ret if params.signal == "trend" else -ret

    def _order_for(self, name: str, score: float) -> str | None:
        params = self.params
        if params.signal == "random":
            head = self._rng.choice(_RANDOM_FAMILIES)
        elif params.signal in ("wedge", "wedge_short", "wedge_long"):
            head = params.short_vol_family if score > 0 else params.long_vol_family
        elif params.signal == "none":
            head = params.short_vol_family
        else:
            head = f"{params.directional_family} {'b' if score > 0 else 'r'}"
        return f"O {name} {head} {params.tenor}"
