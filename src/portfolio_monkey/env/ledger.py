"""The T1 ledger: what the environment emits and metrics consume.

``docs/evaluation_protocol.md`` section 5.  The environment does **not** compute
the reward.  It writes this ledger; ``eval/metrics.py`` reads it and computes
`R = Σ log(V_t / V_{t-1})`.  Section 11.1 of ``docs/env_contract.md`` insists on
that separation for one reason: the training objective and the reported metric
have to be literally the same quantity, which is only true if there is exactly
one implementation of it.

Four streams, written as JSONL so a run can be inspected mid-flight and a
partial run is still readable:

``position_steps.jsonl``
    T1.  One row per open position per **mark** grid point — twice a day,
    including the AM points where no decision is made.  NAV is measured on the
    mark grid, so a ledger written only on the decision grid would measure a
    different path from the one the policy is being scored on.
``nav_panel.jsonl``
    T2, written directly rather than aggregated later.  It is redundant with T1
    by construction, and that is the point: ``reconcile()`` checks the two
    against each other and a mismatch means a cash flow escaped
    ``ExecutionModel``.
``fills.jsonl``
    Every cash-moving event, for the cost attribution of section 8.1.
``decisions.jsonl``
    The raw completion, the parse result and the reasoning trace.  Kept because
    a refused order is data: a policy whose orders are 30% ungrammatical is a
    prompt problem, and that is invisible from PnL alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

from .book import Book, Position
from .execution import Fill

__all__ = [
    "PositionStep",
    "NavRow",
    "DecisionRow",
    "LedgerWriter",
    "read_jsonl",
    "reconcile",
    "LEDGER_VERSION",
]

LEDGER_VERSION = "ledger.v1"

#: NAV reconciles to the cent.  Anything larger is a cash flow that did not go
#: through ``ExecutionModel``, not float noise.
RECONCILE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class PositionStep:
    """One position at one mark grid point (T1)."""

    arm: str
    track: str
    step_ts: str
    session: str
    episode_id: str
    position_id: str
    underlying: str
    strategy_family: str
    orientation: str
    strategy_handle: str
    opened_ts: str
    quantity: int
    dte: int
    mtm_value: float
    mtm_pnl_step: float
    realized_pnl_step: float
    cost_half_spread: float
    cost_fees: float
    delta: float
    gamma: float
    vega: float
    theta: float
    collateral: float
    mark_quality: str
    legs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class NavRow:
    """One account snapshot at one mark grid point (T2)."""

    arm: str
    track: str
    step_ts: str
    session: str
    episode_id: str
    is_decision_point: bool
    nlv: float
    cash: float
    share_value: float
    rf_accrual: float
    mtm_pnl: float
    realized_pnl: float
    cost_half_spread: float
    cost_fees: float
    net_dollar_delta: float
    gross_dollar_delta: float
    dollar_gamma: float
    dollar_vega: float
    dollar_theta: float
    notional_traded: float
    n_positions: int
    collateral_used: float
    stale_mark_share: float
    drawdown: float
    terminated: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """What the policy was shown, what it said, and what came of it.

    ``observation`` is stored in full.  It is the single largest field in the
    ledger and it is worth it: without the exact bytes the policy saw, a
    surprising action cannot be distinguished from a state-space bug, and the
    state space is explicitly a swappable component here.

    ``results`` is the environment's own account of what each order did, and it
    already contains the parse failures — ``OptionsEnv._act`` emits one result
    per rejected order before it touches the book.  A separate ``errors`` column
    would be the same rows written twice and would go stale the first time the
    parser gained a code.

    ``turn`` separates the two completions one step can produce once ``Q`` is
    enabled: the policy asks for prices, is answered, and then acts.  Both are
    written, and the proposal turn is in some ways the more valuable — it
    carries the reasoning that *chose* what to price, and it is the only record
    of the candidates the policy considered and then rejected on price, which
    never reach ``fills`` because they were never opened.

    Adding it makes ``(episode_id, step_index)`` non-unique for the first time.
    That is why it is a column rather than a flag buried in ``extra``: a reader
    that wants the trade must filter ``turn == "act"``, and a filter nobody can
    see is one nobody applies.
    """

    arm: str
    track: str
    step_ts: str
    episode_id: str
    step_index: int
    state_space_id: str
    observation: str
    completion: str
    reasoning: str
    results: tuple[Mapping[str, Any], ...]
    #: ``"act"`` or ``"quote"``.  Defaults to ``"act"`` so that every existing
    #: writer and every ledger already on disk reads as what it is -- a run
    #: with one turn per step -- rather than as an absent field.
    turn: str = "act"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    finish_reason: str = ""
    #: The provider failure the policy swallowed, if any.  ``DeepSeekPolicy``
    #: turns an outage into a hold rather than losing the episode, which means
    #: an unrecorded outage would be indistinguishable from a deliberate hold.
    error: str = ""
    #: Whatever the policy wants kept — cached-token counts, context pressure.
    extra: Mapping[str, Any] = field(default_factory=dict)


class LedgerWriter:
    """Streams the four tables to a run directory.

    Append-mode is deliberate: a run that crashes at step 400 should leave 399
    steps of usable ledger behind, and buffering the whole run in memory to
    write it at the end would throw that away for no gain.
    """

    version = LEDGER_VERSION

    #: Every table is opened up front, so an empty file means "this arm did
    #: nothing" and a *missing* file means the ledger is damaged.  Lazy creation
    #: conflates the two: the hold arm never opens a position, so its
    #: ``position_steps.jsonl`` would be absent and ``reconcile`` would raise
    #: ``FileNotFoundError`` on the one arm that is guaranteed to reconcile.
    TABLES = ("position_steps", "nav_panel", "fills", "decisions", "strategies")

    def __init__(self, root: Path | str, *, arm: str, track: str) -> None:
        self.root = Path(root)
        self.arm = arm
        self.track = track
        self.root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, TextIO] = {}
        self._prev_mtm: dict[str, float] = {}
        for name in self.TABLES:
            self._handle(name)

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "LedgerWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def path(self, name: str) -> Path:
        return self.root / f"{name}.jsonl"

    def _handle(self, name: str) -> TextIO:
        handle = self._handles.get(name)
        if handle is None:
            handle = self.path(name).open("a", encoding="utf-8")
            self._handles[name] = handle
        return handle

    def _write(self, name: str, payload: Mapping[str, Any]) -> None:
        handle = self._handle(name)
        handle.write(json.dumps(payload, default=_encode, separators=(",", ":")) + "\n")
        handle.flush()

    # -- writing ---------------------------------------------------------

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps({"ledger_version": LEDGER_VERSION, **payload}, indent=2, default=_encode),
            encoding="utf-8",
        )

    def write_step(
        self,
        book: Book,
        *,
        episode_id: str,
        step_ts: datetime,
        session: str,
        is_decision_point: bool,
        fills: Sequence[Fill] = (),
        rf_accrual: float = 0.0,
        terminated: str | None = None,
    ) -> NavRow:
        """Write the T1 rows and the T2 row for one mark grid point.

        Returns the NAV row so the caller can carry ``nlv`` forward without
        re-reading the file.
        """
        stamp = step_ts.isoformat()
        by_position: dict[str, list[Fill]] = {}
        for fill in fills:
            by_position.setdefault(fill.position_id, []).append(fill)

        seen: set[str] = set()
        for position in sorted(book.positions.values(), key=lambda p: p.position_id):
            seen.add(position.position_id)
            self._write(
                "position_steps",
                asdict(self._position_step(
                    position,
                    episode_id=episode_id,
                    stamp=stamp,
                    session=session,
                    fills=by_position.get(position.position_id, ()),
                )),
            )

        # A position closed this step is gone from the book but still had a
        # step: dropping it would make the realized PnL of the exit invisible in
        # T1 and break the reconciliation against T2.
        for position_id, position_fills in sorted(by_position.items()):
            if position_id in seen or position_id == "-":
                continue
            self._write(
                "position_steps",
                asdict(self._closed_step(
                    position_fills,
                    episode_id=episode_id,
                    stamp=stamp,
                    session=session,
                )),
            )
            self._prev_mtm.pop(position_id, None)

        row = self._nav_row(
            book,
            episode_id=episode_id,
            stamp=stamp,
            session=session,
            is_decision_point=is_decision_point,
            fills=fills,
            rf_accrual=rf_accrual,
            terminated=terminated,
        )
        self._write("nav_panel", asdict(row))
        for fill in fills:
            self._write("fills", {"step_ts": stamp, "episode_id": episode_id, **asdict(fill)})
        return row

    def write_decision(self, row: DecisionRow) -> None:
        self._write("decisions", asdict(row))

    def write_strategies(
        self,
        proposals: Sequence[Mapping[str, Any]],
        *,
        episode_id: str,
        step_index: int,
    ) -> int:
        """One row per strategy the policy named, whether or not it filled.

        Deliberately **not** reconciled against ``fills``: the whole point of
        this table is the rows that have no fill.  ``reconcile`` therefore
        ignores it, and a reader joins on ``position_id`` -- empty on every
        proposal that died before it reached the book.

        Held apart from ``decisions`` rather than nested inside it because the
        offline questions are per *strategy* (which families does this policy
        reach for, what fraction of its intents the ladder can actually fill,
        how often it names a coordinate versus taking the default) and a
        column-per-step table answers none of them without a flattening pass.
        """
        for proposal in proposals:
            self._write(
                "strategies",
                {
                    "arm": self.arm,
                    "track": self.track,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    **proposal,
                },
            )
        return len(proposals)

    # -- row construction ------------------------------------------------

    def _position_step(
        self,
        position: Position,
        *,
        episode_id: str,
        stamp: str,
        session: str,
        fills: Sequence[Fill],
    ) -> PositionStep:
        mtm = position.mark * position.quantity
        previous = self._prev_mtm.get(position.position_id)
        # A position opened this step has no previous mark, and its MTM change
        # is zero by construction — it was filled and marked at the same
        # snapshot.  Seeding from the current value rather than from zero is
        # what keeps the entry premium out of ``mtm_pnl_step``, where it would
        # read as an instant gain or loss the size of the position.
        self._prev_mtm[position.position_id] = mtm
        return PositionStep(
            arm=self.arm,
            track=self.track,
            step_ts=stamp,
            session=session,
            episode_id=episode_id,
            position_id=position.position_id,
            underlying=position.underlying,
            strategy_family=position.family,
            orientation=position.orientation,
            strategy_handle=position.strategy_handle,
            opened_ts=position.opened_at.isoformat(),
            quantity=position.quantity,
            dte=position.min_dte,
            mtm_value=mtm,
            mtm_pnl_step=0.0 if previous is None else mtm - previous,
            realized_pnl_step=sum(f.realized for f in fills),
            cost_half_spread=sum(f.half_spread for f in fills),
            cost_fees=sum(f.fees for f in fills),
            delta=position.dollar_delta,
            gamma=position.dollar_gamma,
            vega=position.dollar_vega,
            theta=position.dollar_theta,
            collateral=position.collateral,
            mark_quality=position.mark_quality,
            legs=tuple(
                {
                    "contract_id": leg.contract_id,
                    "right": leg.right,
                    "strike": leg.strike,
                    "expiry": leg.expiry.isoformat(),
                    "qty": leg.ratio * position.quantity,
                    "entry_price": leg.entry_price,
                }
                for leg in position.legs
            ),
        )

    def _closed_step(
        self, fills: Sequence[Fill], *, episode_id: str, stamp: str, session: str
    ) -> PositionStep:
        head = fills[0]
        previous = self._prev_mtm.get(head.position_id, 0.0)
        return PositionStep(
            arm=self.arm,
            track=self.track,
            step_ts=stamp,
            session=session,
            episode_id=episode_id,
            position_id=head.position_id,
            underlying=head.underlying,
            strategy_family=head.family,
            orientation="",
            strategy_handle=head.strategy_handle,
            opened_ts="",
            quantity=0,
            dte=0,
            mtm_value=0.0,
            mtm_pnl_step=-previous,
            realized_pnl_step=sum(f.realized for f in fills),
            cost_half_spread=sum(f.half_spread for f in fills),
            cost_fees=sum(f.fees for f in fills),
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            collateral=0.0,
            mark_quality=head.mark_quality,
            legs=(),
        )

    def _nav_row(
        self,
        book: Book,
        *,
        episode_id: str,
        stamp: str,
        session: str,
        is_decision_point: bool,
        fills: Sequence[Fill],
        rf_accrual: float,
        terminated: str | None,
    ) -> NavRow:
        view = book.view(datetime.fromisoformat(stamp))
        return NavRow(
            arm=self.arm,
            track=self.track,
            step_ts=stamp,
            session=session,
            episode_id=episode_id,
            is_decision_point=is_decision_point,
            nlv=view.nav,
            cash=view.cash,
            share_value=book.share_value,
            rf_accrual=rf_accrual,
            mtm_pnl=view.unrealized_pnl,
            realized_pnl=view.realized_pnl,
            cost_half_spread=sum(f.half_spread for f in fills),
            cost_fees=sum(f.fees for f in fills),
            net_dollar_delta=view.net_dollar_delta,
            gross_dollar_delta=sum(abs(p.dollar_delta) for p in view.positions),
            dollar_gamma=view.net_dollar_gamma,
            dollar_vega=view.net_dollar_vega,
            dollar_theta=view.net_dollar_theta,
            notional_traded=sum(abs(f.mid_value) for f in fills),
            n_positions=view.n_positions,
            collateral_used=view.collateral_used,
            stale_mark_share=view.stale_mark_share,
            drawdown=view.drawdown,
            terminated=terminated,
        )


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    steps: int
    max_error: float
    worst_step: str | None
    ok: bool
    detail: tuple[str, ...] = ()
    by_identity: Mapping[str, float] = field(default_factory=dict)


def reconcile(root: Path | str, *, tolerance: float = RECONCILE_TOLERANCE) -> Reconciliation:
    """Check both accounting identities of ``docs/env_contract.md`` section 8.2.

    The doc states two and calls for both to "reconcile to the cent at every
    step".  Only the first used to be checked here, and the gap mattered: the
    balance sheet can be exact while the *attribution* is wrong, and every cost
    metric in ``eval/metrics.py`` reads the attribution rather than the balance
    sheet.

    ``balance``
        ``NLV = cash + share_value + Σ mtm_value``.  Fails only if a cash flow
        bypassed ``ExecutionModel`` — the failure mode that is otherwise silent,
        because a book that mints money still produces a monotone,
        plausible-looking NAV path.  ``share_value`` is a separate column rather
        than folded into ``cash`` so this is exact: a hedge is the one thing
        that puts value outside both cash and the option positions, and one
        booked once in cash but twice in shares would net to zero in any
        aggregate that combined them.
    ``cash``
        ``Δcash = rf_accrual + Σ fill.cash_delta``.  Every cash movement is
        either interest or a fill; nothing else may move it.
    ``realized``
        ``Δrealized_pnl = rf_accrual + Σ fill.realized``.  This is the one the
        report leans on.  It is what makes "costs explain the NAV change" a
        checked claim rather than a docstring.
    ``fill``
        Per fill, ``cash_delta = mid_value - half_spread - fees``.  Holds for
        options, expiries and hedges alike, and is what lets the cost columns be
        summed across kinds.
    ``orphan``
        Every fill's ``step_ts`` appears in the NAV panel.  A fill stamped at a
        step the panel never recorded is invisible to all three sums above.

    The three differenced identities start at the *second* panel row, since the
    first has no predecessor to difference against; that row is covered by
    ``balance``.
    """
    root = Path(root)
    mtm_by_step: dict[str, float] = {}
    for row in read_jsonl(root / "position_steps.jsonl"):
        mtm_by_step[row["step_ts"]] = mtm_by_step.get(row["step_ts"], 0.0) + row["mtm_value"]

    errors: list[str] = []
    worst = 0.0
    worst_step: str | None = None
    by_identity: dict[str, float] = {}

    def record(identity: str, step: str, residual: float, scale: float) -> None:
        nonlocal worst, worst_step
        if abs(residual) <= max(tolerance, abs(scale) * 1e-9):
            return
        by_identity[identity] = max(by_identity.get(identity, 0.0), abs(residual))
        if abs(residual) > abs(worst):
            worst, worst_step = residual, step
        if len(errors) < 10:
            errors.append(f"{identity} {step}: residual {residual:+.4f}")

    cash_by_step: dict[str, float] = {}
    realized_by_step: dict[str, float] = {}
    fills_path = root / "fills.jsonl"
    if fills_path.exists():
        for fill in read_jsonl(fills_path):
            step = fill["step_ts"]
            cash_by_step[step] = cash_by_step.get(step, 0.0) + fill["cash_delta"]
            realized_by_step[step] = realized_by_step.get(step, 0.0) + fill["realized"]
            record(
                "fill",
                step,
                fill["cash_delta"] - (fill["mid_value"] - fill["half_spread"] - fill["fees"]),
                scale=abs(fill["mid_value"]),
            )

    steps = 0
    seen: set[str] = set()
    previous: dict[str, Any] | None = None
    for row in read_jsonl(root / "nav_panel.jsonl"):
        steps += 1
        step = row["step_ts"]
        seen.add(step)
        record(
            "balance",
            step,
            row["nlv"] - row["cash"] - row.get("share_value", 0.0) - mtm_by_step.get(step, 0.0),
            scale=row["nlv"],
        )
        if previous is not None:
            rf = row.get("rf_accrual", 0.0)
            flow = cash_by_step.get(step, 0.0)
            record(
                "cash",
                step,
                (row["cash"] - previous["cash"]) - rf - flow,
                scale=abs(row["cash"]) + abs(flow),
            )
            realized_flow = realized_by_step.get(step, 0.0)
            record(
                "realized",
                step,
                (row["realized_pnl"] - previous["realized_pnl"]) - rf - realized_flow,
                scale=abs(rf) + abs(realized_flow) + abs(row["realized_pnl"]),
            )
        previous = row

    for step in sorted(set(cash_by_step) - seen):
        record("orphan", step, cash_by_step[step], scale=0.0)

    return Reconciliation(
        steps=steps,
        max_error=abs(worst),
        worst_step=worst_step,
        ok=not errors,
        detail=tuple(errors),
        by_identity=dict(by_identity),
    )


def _encode(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__} into the ledger")
