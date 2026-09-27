"""The seam between the environment and whatever the policy is shown.

``docs/state_space.md`` is one state space.  It will not be the last one: the
tenor policy, the field list and the AM/PM grid in it are all still open
questions (`⟨Q7a⟩`…`⟨Q7h⟩`).  So the environment is written against the
*protocol* in this module and never against that document.

The contract in one sentence: **the environment computes numbers, the state
space decides bytes.**  Nothing below the seam knows a field name, a unit, a
scale factor or a block letter; nothing above it knows what a position costs to
close.  Swapping ``state_space.v1`` for ``state_space.v2`` is a change to
``EnvConfig.state_space_id`` and a new module in this package — the resolvers,
the book, the ledger and the metrics are untouched by construction.

Three things live here rather than in an implementation because every state
space needs them and none should reimplement them:

1. ``BookView`` / ``PositionView`` — the neutral view of the book.  A state
   space renders these; it never reaches into ``Book``.
2. ``FeatureSource`` and the point-in-time assertion.  ``docs/state_space.md``
   section 5 asks for ``available_time <= decision_time`` to be *enforced per
   field per row* rather than trusted.  That check is here, once, so a new
   state space cannot forget it.
3. The registry, keyed by the same string that goes into the run manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from .spec import EnvConfig

__all__ = [
    "PointInTimeViolation",
    "StateSpaceError",
    "UnknownStateSpace",
    "FeatureRecord",
    "FeatureSource",
    "PositionView",
    "BookView",
    "EpisodeContext",
    "StepContext",
    "Observation",
    "StateSpace",
    "register_state_space",
    "build_state_space",
    "registered_state_spaces",
    "assert_point_in_time",
    "estimate_tokens",
]


class StateSpaceError(RuntimeError):
    """Base class for failures raised while serializing a state."""


class UnknownStateSpace(StateSpaceError):
    """Raised when ``EnvConfig.state_space_id`` names nothing registered."""


class PointInTimeViolation(StateSpaceError):
    """Raised when a field would enter the prompt before it was knowable.

    This is deliberately fatal rather than a warning.  ``docs/state_space.md``
    section 5: "Any field that cannot support the assertion does not enter the
    state — that is the admission rule."  A step that cannot be serialized
    causally is a step that must not be scored.
    """

    def __init__(self, dataset: str, key: str, field_name: str, available: Any, decision: Any) -> None:
        super().__init__(
            f"{dataset}:{key}:{field_name} became available at {available!r}, "
            f"after the decision time {decision!r}"
        )
        self.dataset = dataset
        self.key = key
        self.field_name = field_name
        self.available = available
        self.decision = decision


# --------------------------------------------------------------------------
# What a state space reads from
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One row of one feature dataset, with its point-in-time provenance.

    ``values`` is intentionally untyped: the whole point of the seam is that
    the environment does not enumerate the fields a state space may want.  The
    provenance fields are typed, because those are what gets asserted.

    ``available_time`` may be a single timestamp for the whole row, or a
    per-field mapping when the builder tracks it at that granularity.  Both
    shapes are accepted and ``assert_point_in_time`` handles either.
    """

    dataset: str
    key: str
    values: Mapping[str, Any]
    available_time: datetime | Mapping[str, datetime] | None = None
    decision_time: datetime | None = None
    point_in_time_rule: str | None = None

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@runtime_checkable
class FeatureSource(Protocol):
    """Point-in-time access to the materialized feature datasets.

    Kept as narrow as it can be: one method, one dataset, one grid point.  A
    backtest source reads parquet; a live source would hit a cache; neither
    difference reaches the state space.
    """

    def fetch(
        self,
        dataset: str,
        *,
        trade_date: date,
        session: str,
        keys: Sequence[str],
        decision_time: datetime,
    ) -> Mapping[str, FeatureRecord]:
        """Return one record per requested key, keyed by that key.

        Missing keys are simply absent from the mapping — a state space decides
        whether that is fatal or an ``na`` on the wire, because that judgement
        differs by field and is exactly the kind of thing this seam must not
        centralize.
        """
        ...


# --------------------------------------------------------------------------
# What a state space reads about the book
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionView:
    """One open package as the state space sees it.

    Everything is in account currency, already aggregated to the package: a
    state space never multiplies by a contract multiplier or sums over legs.
    Greeks are dollar greeks for the whole package including sign.

    ``docs/state_space.md`` section 6.3 proposes serializing only
    ``dollar_delta`` per position and moving the rest to the account line.
    That is a *rendering* decision and lives in the implementation; all four
    greeks are carried here so a state space that disagrees can use them.
    """

    position_id: str
    underlying: str
    family: str
    orientation: str
    dte: int
    quantity: int
    mark: float
    entry_cost: float
    unrealized_pnl: float
    dollar_delta: float
    dollar_gamma: float
    dollar_vega: float
    dollar_theta: float
    collateral: float
    mark_quality: str
    opened_at: datetime | None = None
    #: The position this one replaced via ``X``, or "" if it was opened outright.
    #: Carried to the view because it is the only record that survives into
    #: *later* steps: the ``R`` receipt announcing the roll scrolls out of the
    #: append-only context, and after that the ``P`` block is the policy's only
    #: evidence that what it holds is a continuation rather than a fresh
    #: commitment — which is the difference between "I have been wrong about NVDA
    #: for six weeks" and "I opened NVDA on Tuesday".
    rolled_from: str = ""
    roll_generation: int = 0


@dataclass(frozen=True, slots=True)
class BookView:
    """The account line, pre-aggregated.

    Every quantity here is computed by ``MarketResolver``/``Book`` and merely
    formatted downstream.  ``drawdown`` is from the running peak of ``nav``,
    signed negative.  ``stale_mark_share`` is the fraction of gross position
    value marked at anything other than a live mid — ``RiskControls`` gates on
    it, and a state space may or may not choose to show it.
    """

    nav: float
    cash: float
    buying_power: float
    collateral_used: float
    collateral_utilization: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown: float
    net_dollar_delta: float
    net_dollar_gamma: float
    net_dollar_vega: float
    net_dollar_theta: float
    stale_mark_share: float
    shares: Mapping[str, float] = field(default_factory=dict)
    positions: tuple[PositionView, ...] = ()

    @property
    def n_positions(self) -> int:
        return len(self.positions)


# --------------------------------------------------------------------------
# Context objects
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """What is true for a whole episode and stated once.

    ``carried_in`` is the ``BookState`` handoff of ``docs/env_contract.md``
    section 1.4: under monthly episodes the book crosses the boundary even
    though the context does not, and the episode header is the only place the
    policy can be told what it inherited.
    """

    episode_id: str
    config: EnvConfig
    start_date: date
    end_date: date
    decision_points: int
    carried_in: BookView | None = None


@dataclass(frozen=True, slots=True)
class StepContext:
    """Everything a state space may look at to serialize one step."""

    episode: EpisodeContext
    step_index: int
    trade_date: date
    session: str
    decision_time: datetime
    book: BookView
    features: FeatureSource
    last_result: Mapping[str, Any] | None = None
    previous_trade_date: date | None = None

    @property
    def config(self) -> EnvConfig:
        return self.episode.config

    def fetch(self, dataset: str, keys: Sequence[str], *, assert_causal: bool = True) -> Mapping[str, FeatureRecord]:
        """Fetch and, by default, assert causality on every field returned."""
        return self._fetch(
            dataset,
            keys,
            trade_date=self.trade_date,
            session=self.session,
            assert_causal=assert_causal,
        )

    def fetch_previous_close(
        self, dataset: str, keys: Sequence[str], *, assert_causal: bool = True
    ) -> Mapping[str, FeatureRecord]:
        """The same fetch against the previous trading date's close.

        A field whose intraday change is fitted open-to-close says nothing at
        the open, and the overnight move is where earnings and news land.  This
        is the only way to reach across the session boundary, so it is spelled
        out rather than left to a state space to improvise with ``features``.

        Empty before the first date in coverage, and after any gap in it.
        """
        if self.previous_trade_date is None:
            return {}
        return self._fetch(
            dataset,
            keys,
            trade_date=self.previous_trade_date,
            session="PM",
            assert_causal=assert_causal,
        )

    @property
    def skipped_sessions(self) -> tuple[str, ...]:
        """Sessions of *this* date that this step's grid passed over, in order.

        Session-partitioned datasets are addressed by the session the policy is
        standing on, so a session the decision grid does not visit is never
        read: its rows exist, are causal, and are invisible.  For most datasets
        that is exactly right -- an AM quote is superseded by the PM quote of
        the same date, so skipping it loses nothing.

        It is wrong for an append-only dataset, where a row is a distinct *item*
        rather than a restatement of one.  ``news_catalyst_rows`` is the case:
        the builder places each item at the first session whose decision time it
        precedes, so under the shipped ``decision_sessions = ("PM",)`` every
        item stamped AM -- about three quarters of the corpus -- is dropped
        rather than deferred.

        This names those sessions so a block that needs them can backfill.  It
        is deliberately *not* "every earlier session": a session the grid does
        visit renders its own rows, and re-serving them into an append-only
        conversation is pure token cost (``docs/state_space.md``).  Under an
        AM+PM grid it is therefore empty and the PM block is unchanged.
        """
        grid = self.config.grid
        marks = grid.mark_sessions
        if self.session not in marks:
            return ()
        decisions = set(grid.decision_sessions)
        return tuple(
            s for s in marks[: marks.index(self.session)] if s not in decisions
        )

    def fetch_session(
        self,
        dataset: str,
        keys: Sequence[str],
        *,
        session: str,
        assert_causal: bool = True,
    ) -> Mapping[str, FeatureRecord]:
        """The same fetch against another session of the same trade date.

        Causality is asserted against *this* step's ``decision_time``, not the
        fetched session's, which is the whole safety argument: an earlier
        session's rows carry their own ``available_time``, so the guard passes
        only because they really were public first.  Pointing this at a later
        session raises rather than leaking.
        """
        return self._fetch(
            dataset,
            keys,
            trade_date=self.trade_date,
            session=session,
            assert_causal=assert_causal,
        )

    def _fetch(
        self,
        dataset: str,
        keys: Sequence[str],
        *,
        trade_date: date,
        session: str,
        assert_causal: bool,
    ) -> Mapping[str, FeatureRecord]:
        records = self.features.fetch(
            dataset,
            trade_date=trade_date,
            session=session,
            keys=keys,
            decision_time=self.decision_time,
        )
        if assert_causal:
            for record in records.values():
                assert_point_in_time(record, self.decision_time)
        return records


@dataclass(frozen=True, slots=True)
class Observation:
    """The bytes for one step, plus what it took to produce them.

    ``text`` is what is appended to the conversation.  ``blocks`` is the same
    content split by block name, which is what tests assert against and what
    the ledger records for reproducibility.  ``missing`` names every field that
    was rendered as unavailable — that count is a run-quality metric, not a
    debug aid, because ``docs/state_space.md`` section 7 shows fields going
    missing in calendar-correlated blocks.
    """

    text: str
    blocks: Mapping[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    token_estimate: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# The protocol itself
# --------------------------------------------------------------------------


@runtime_checkable
class StateSpace(Protocol):
    """Everything the environment is allowed to ask about what the policy sees.

    Five methods, and the split between them is the append-only rule of
    ``docs/state_space.md`` R7: ``system_block`` and ``episode_block`` are
    emitted once and must be byte-identical across steps so that prefix caching
    bills only the per-step delta, while ``step_block`` is the delta.

    ``action_grammar`` is here rather than in ``env/actions.py`` on purpose.
    The wire format of an order is a serialization concern that has to match
    the schema the state space declared — a state space that renames the
    position block has renamed the token the policy uses to close a position.
    The *parser* stays in ``actions.py``; this returns the human-readable
    grammar the system prompt states.
    """

    @property
    def state_space_id(self) -> str:
        """Matches ``EnvConfig.state_space_id`` and goes into the manifest."""
        ...

    def system_block(self, config: EnvConfig) -> str:
        """The schema declaration.  Emitted once, before the first step."""
        ...

    def action_grammar(self, config: EnvConfig) -> str:
        """How an order is written, in the same encoding as the state."""
        ...

    def episode_block(self, ctx: EpisodeContext) -> str:
        """Episode header: window, starting book, what was carried in."""
        ...

    def step_block(self, ctx: StepContext) -> Observation:
        """The per-step observation.  The only thing emitted more than once."""
        ...

    def quote_block(self, results: Sequence[Mapping[str, Any]]) -> Observation:
        """The answer to a ``Q`` turn, emitted between a step and its action.

        Here rather than in the runner because it is a wire format: the rows a
        quote returns have to read the same as the rows a fill returns, and the
        state space is the only thing that knows how an ``R`` row is written.
        """
        ...

    def quote_tool_schema(self, config: EnvConfig) -> Mapping[str, Any] | None:
        """``action_grammar``'s machine-readable half, for the ``Q`` verb.

        Next to ``action_grammar`` for the reason given there: the wire format
        is the state space's to declare, and a schema is that declaration in
        the form a provider can enforce. Returning it here rather than building
        it in the policy is what keeps one vocabulary -- a policy that wrote its
        own schema would be describing a grammar it does not own.

        ``None`` when the arm does not run the quote round on the tool channel,
        so that the caller has one thing to test and the choice between the two
        channels is made once, by whoever also writes the grammar.
        """
        ...

    def required_datasets(self) -> tuple[str, ...]:
        """Datasets this state space reads.

        The episode driver checks coverage against these *before* a run rather
        than discovering a missing dataset on step 40 of 42.
        """
        ...


# --------------------------------------------------------------------------
# Point-in-time enforcement
# --------------------------------------------------------------------------


def assert_point_in_time(record: FeatureRecord, decision_time: datetime) -> None:
    """Fail unless every field in ``record`` was knowable at ``decision_time``.

    ``docs/state_space.md`` section 5 is explicit that this must be an
    assertion and not a belief about the builder.  Rows with no
    ``available_time`` at all are *not* silently accepted: a dataset that
    cannot say when it knew something cannot be admitted to the state, which is
    the same rule that keeps the earnings date out.
    """
    available = record.available_time
    if available is None:
        raise PointInTimeViolation(record.dataset, record.key, "*", None, decision_time)

    if isinstance(available, Mapping):
        for name, stamp in available.items():
            if stamp is None or stamp > decision_time:
                raise PointInTimeViolation(record.dataset, record.key, name, stamp, decision_time)
        for name in record.values:
            if name not in available:
                raise PointInTimeViolation(record.dataset, record.key, name, None, decision_time)
        return

    if available > decision_time:
        raise PointInTimeViolation(record.dataset, record.key, "*", available, decision_time)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for budget checks when no tokenizer is loaded.

    Deliberately crude and deliberately *not* used for any measured number in
    ``docs/state_space.md``, all of which were counted with the real Qwen3-4B
    tokenizer.  This exists so the episode driver can trip a budget guard
    without a 400 MB dependency; anything reported as a token count comes from
    the provider's usage field instead.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[EnvConfig], StateSpace]] = {}


def register_state_space(state_space_id: str, factory: Callable[[EnvConfig], StateSpace]) -> None:
    """Register a factory under an id.  Re-registering the same id is an error.

    Silent replacement would make a run manifest ambiguous: two runs stamped
    ``state_space.v1`` could have seen different bytes depending on import
    order.
    """
    existing = _REGISTRY.get(state_space_id)
    if existing is not None and existing is not factory:
        raise StateSpaceError(f"state space {state_space_id!r} is already registered")
    _REGISTRY[state_space_id] = factory


def registered_state_spaces() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_state_space(config: EnvConfig) -> StateSpace:
    """Instantiate the state space named by ``config.state_space_id``."""
    _load_builtin_state_spaces()
    try:
        factory = _REGISTRY[config.state_space_id]
    except KeyError as exc:
        raise UnknownStateSpace(
            f"no state space registered as {config.state_space_id!r}; "
            f"known: {registered_state_spaces()}"
        ) from exc
    state_space = factory(config)
    if state_space.state_space_id != config.state_space_id:
        raise StateSpaceError(
            f"state space factory for {config.state_space_id!r} produced "
            f"{state_space.state_space_id!r}"
        )
    return state_space


def _load_builtin_state_spaces() -> None:
    """Import the shipped implementations so they self-register.

    Import is deferred to call time to keep ``statespace`` importable by the
    implementations themselves without a cycle.
    """
    from . import statespace_v1  # noqa: F401


def format_signed(value: int) -> str:
    """Render an integer with an explicit ``+`` on non-negatives.

    Shared because the sign convention has to be uniform across every block of
    every state space: a bare ``0`` and a ``+0`` in different blocks is the
    kind of inconsistency the policy will read as meaningful.
    """
    return f"{value:+d}"


def scaled(value: float | None, scale: float, *, signed: bool = False, missing: str = "na") -> str:
    """Encoding rule R2: one integer with a declared scale, or ``na``.

    ``None`` and NaN both render as ``missing``.  NaN is checked explicitly
    because several feature columns carry NaN rather than null for an unfitted
    value, and ``float('nan') > x`` is False for every ``x``, so an unguarded
    comparison downstream would pass silently.
    """
    if value is None or value != value:
        return missing
    integer = int(round(value * scale))
    return format_signed(integer) if signed else str(integer)


def render_rows(rows: Iterable[Sequence[str]]) -> str:
    """Encoding rule R3: positional, space-delimited, one line per row."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in rows)
