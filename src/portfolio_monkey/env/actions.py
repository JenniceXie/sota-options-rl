"""Parsing and validating what the policy emits.

The wire format is the one ``statespace_v1.action_grammar`` declares:

.. code-block:: text

    O NVDA cv b 8_30 35/10   credit vertical, 8-30 DTE, short 35 delta, 10 wide
    O AAPL ol b 31_90        outright, 31-90 DTE, at the family default delta
    C p07                    close position p07
    X p07 31_90              roll p07 out to 31-90 DTE, same view
    H                        hold

**Tenor is required and coordinates are optional**, which is the reverse of
how it reads.  A bucket is a first-order choice -- it sets gamma, theta and the
cost per unit of risk -- whereas the coordinates perturb a package the policy
has already committed to.  Making the first-order choice omittable is what
produced a year of 15-DTE trading with no evidence the policy ever considered
anything else.

Three properties this module is built around, all of them consequences of
``docs/env_contract.md``:

**Size is not in the grammar.**  Section 4.5 moves position size out of the
action space and into ``SizeResolver`` entirely.  An order that mentions a
contract count is a grammar error, not a large trade.

**Coordinates are free inside per-family bounds** (``⟨Q9⟩``, parametric action
space).  The "standard" values in the taxonomy are *defaults* used when a
coordinate is omitted — that is what ``EnvConfig.coordinates`` encodes, and it
is why an omitted coordinate is legal while an out-of-bounds one is not.

**Every rejection is named and returned, never swallowed.**  The error codes
below are what the ``RES`` block shows the policy on the next step.  A policy
that cannot see *why* an order was refused re-emits it every step for the rest
of the episode, which is the single most expensive failure mode in a
context-limited rollout.

Parsing is deliberately tolerant of the surrounding prose — a reasoning model
will narrate — but strict about the order lines themselves.  Tolerance is at
line granularity: a line that starts with an order verb must parse completely
or be rejected with a code.  It is never partially interpreted.

**Why the roll verb is ``X`` and not ``R``.**  ``R`` is already the environment's
*receipt* prefix — ``R OK O NVDA lg b 8_30 p01 c366 e22``.  The policy reads its
own prior actions and the environment's replies from one append-only stream, so
a ``R p07 31_90`` action line and an ``R OK ...`` receipt would share a first
token.  ``statespace_v1._market_block`` records what that costs: a reader that
learns "the first token names the row type" and then meets a row where it does
not decodes every following cell one column out of phase.  ``X`` collides with
no block letter (``S``/``M``/``N``/``A``/``P``/``R``), no family code and no
orientation code.

**A roll inherits everything it does not restate.**  ``X`` names a position and
a new tenor; the underlying, family and orientation come from the position being
replaced.  That is what separates a roll from a close plus an unrelated open —
there is no spelling of ``X`` that turns an NVDA vertical into an AAPL condor —
and it means the *link* between the two positions is structural rather than an
inference some later reader has to make from timestamps.

Omitted coordinates default to **the replaced position's** coordinates, not to
the family defaults.  A policy that deliberately opened at ``d30`` and then
writes ``X p07 31_90`` is asking to move the expiry, and resetting it to the
``d50`` default would be the environment overriding a choice the policy had
already made and paid for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ``coordinate_order``/``coordinate_token`` are imported, not defined: this
# module used to carry a second copy of the family->coordinate table, and the
# two copies disagreed about ``long_straddle`` (one prefix here, two there) for
# long enough to become ``<Q47>``.  The table lives in the feature layer because
# this module already depends on it for ``TEMPLATE_SPECS``, so the dependency
# only runs one way.
from portfolio_monkey.data.features.strategy_templates import (
    HANDLE_VERSION,
    TEMPLATE_SPECS,
    coordinate_order,
    coordinate_token,
)

from .spec import EnvConfig

__all__ = [
    "ActionError",
    "Order",
    "OpenOrder",
    "QuoteOrder",
    "CloseOrder",
    "RollOrder",
    "RollBasis",
    "HoldOrder",
    "ParsedAction",
    "parse_action",
    "has_quote_lines",
    "render_quote_call",
    "strip_real_name_orders",
    "coordinate_order",
    "coordinate_token",
    "build_strategy_handle",
    "FAMILY_CODES",
    "ORIENTATION_CODES",
    "ERROR_CODES",
]

# Wire code -> canonical name.  Kept here rather than imported from
# ``statespace_v1`` because a different state space may use different codes and
# the parser must agree with whichever one declared the schema; the env wires
# the two together, and ``tests/test_actions.py`` asserts they match for v1.
FAMILY_CODES: Mapping[str, str] = {
    "ol": "outright",
    "dv": "debit_vertical",
    "cv": "credit_vertical",
    "dg": "defined_risk_reversal",
    "bf": "butterfly",
    "ls": "long_straddle",
    "lg": "long_strangle",
    "ib": "iron_butterfly",
    "ic": "iron_condor",
}

ORIENTATION_CODES: Mapping[str, str] = {"b": "bullish", "r": "bearish", "n": "neutral"}


def render_quote_call(arguments: Mapping[str, Any]) -> str:
    """One ``quote_package`` tool call as the ``Q`` line the parser reads.

    Here, next to the parser, rather than in the policy that receives the call
    or the state space that declares the schema.  The DSL is this module's, and
    a renderer kept anywhere else would be a second statement of the field
    order -- which is the failure ``COORDINATE_ORDER`` already records the cost
    of.  It also puts the round trip in one place, so
    ``parse_action(render_quote_call(args))`` is a test and not an integration.

    **Nothing here validates.**  Codes are written through unchanged even when
    they are not in ``FAMILY_CODES``, coordinates are not bounded, and a missing
    field renders as an empty cell.  That is deliberate: a policy that repaired
    a malformed call would hide it, and the whole point of normalizing to text
    is that the call meets exactly the checks a typed ``Q`` line meets --
    ``E_UNKNOWN_FAMILY``, ``E_ORIENTATION``, ``E_TENOR``, ``E_COORD_BOUNDS`` --
    and lands in the ledger with the same refusal code either way.  The
    provider's enums are a first filter, not a substitute; ``additionalModel``
    fields on this endpoint are unvalidated and a schema can be accepted and
    ignored, so the parser stays the authority.

    Coordinates are rendered with ``%g``, matching ``_pct``: the model sends
    ``30`` or ``30.0`` for the same delta and ``30.0`` would be two bytes of
    completion the parser reads identically.  An empty or absent list renders no
    cell at all, which is the grammar's "take the family defaults".
    """
    coordinates = arguments.get("coordinates") or ()
    cells = [
        "Q",
        str(arguments.get("underlying", "")),
        str(arguments.get("family", "")),
        str(arguments.get("orientation", "")),
        str(arguments.get("tenor", "")),
    ]
    if coordinates:
        cells.append("/".join(f"{float(c):g}" for c in coordinates))
    return " ".join(cell for cell in cells if cell)

ERROR_CODES = (
    "E_GRAMMAR",
    "E_UNKNOWN_NAME",
    "E_UNKNOWN_FAMILY",
    "E_NOT_TRADEABLE",
    "E_ORIENTATION",
    "E_TENOR",
    "E_COORD_ARITY",
    "E_COORD_BOUNDS",
    "E_UNKNOWN_POS",
    "E_DUPLICATE",
    "E_TOO_MANY_ORDERS",
    # Only reachable under ``flags.anonymize``: the completion named a real
    # underlying where a label was expected.  See ``real_name_orders``.
    "E_REAL_NAME",
    # Only reachable on a turn that carried at least one ``Q``: the same
    # completion also tried to trade.  Refused rather than silently dropped,
    # because a policy that cannot see why its order vanished will re-send it
    # every step -- and refused rather than executed, because an order written
    # *before* the quotes came back was written without the information the
    # quote round exists to supply.
    "E_QUOTE_TURN",
    #: Per *name*, not per completion — see ``max_quotes_per_name``.
    "E_TOO_MANY_QUOTES",
)

# An order occupies a whole line and nothing else.  Anchoring both ends is what
# keeps a narrated completion from being mined for orders: "Overall, I hold" is
# prose, not an ``O`` order, and "C." is a sentence, not a close.
#
# ``Q`` joined the set on 2026-09-24.  It is in the *same* regex as the trading
# verbs rather than in one of its own precisely so that refuse-and-count
# (``strip_real_name_orders``) covers it without being told: a quote naming a
# real ticker is the same leak as an order naming one, and a separate pattern
# would have had to remember that.
_ORDER_LINE = re.compile(r"^(?P<verb>[OCHXQ])(?:\s+(?P<rest>\S.*?))?\s*$")
_POSITION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,15}$")

#: Markdown emphasis a model wraps around a line it means as a command.  Live
#: DeepSeek completions end ``Command:\n`` ``` `O MSFT cv b` ```, and treating
#: the backticks as prose scored a deliberate, well-argued trade as an abstain.
#: Stripping them does not relax the both-ends anchor above — what is left must
#: still be a whole line and nothing else — and a line the model has explicitly
#: marked as code is stronger evidence of an order than a bare line, not weaker.
_EMPHASIS = ("```", "``", "`", "**", "*")

#: An order line may not be longer than this many whitespace tokens.  A model
#: that appends a size, a limit price or a comment to an order is emitting
#: something this grammar does not have, and guessing which part to keep is how
#: a "sell 1 lot" becomes a hundred.
_MAX_ORDER_TOKENS = 6


def strip_real_name_orders(
    text: str, names: Iterable[str]
) -> tuple[str, list[tuple[str, str]]]:
    """Remove order lines that name a real underlying instead of its label.

    Returns the completion with those lines gone and the ``(line, name)`` pairs
    that were taken out, so the caller can refuse each one individually rather
    than discarding a completion whose other lines were properly masked.

    Only meaningful under ``flags.anonymize``, and it runs on the completion
    *before* unmasking, because afterwards a guessed ``NVDA`` and an unmasked
    ``U03`` are the same six characters and nothing downstream can tell them
    apart.  An unknown *label* was already refused -- ``U44`` survives unmasking
    and dies at ``E_UNKNOWN_NAME`` -- but a real *name* passed straight through
    and filled, which is the hole this closes.

    Scoped to order lines and to whole tokens, both deliberately.  Prose is not
    an order and ``parse_action`` already discards it, so refusing a paragraph
    that happens to contain ``SPY`` would inflate the count with something that
    could never have traded; and a whole-token match is what keeps ``MU`` from
    firing on a word that contains it.  Case-sensitive because the parser is:
    ``tradeable`` is an exact-match set, so ``O nvda ls n 8_30`` is already
    refused as ``E_UNKNOWN_NAME`` and does not need a second code.
    """
    wanted = set(names)
    kept: list[str] = []
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = _strip_emphasis(line)
        named = ""
        if stripped and _ORDER_LINE.match(stripped) is not None:
            named = next((t for t in stripped.split() if t in wanted), "")
        if named:
            found.append((stripped, named))
        else:
            kept.append(line)
    return "\n".join(kept), found


@dataclass(frozen=True, slots=True)
class ActionError:
    """A rejected order, in the shape the ``RES`` block renders."""

    code: str
    detail: str
    raw: str

    def as_result(self) -> dict[str, str]:
        return {"order": self.raw, "status": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class OpenOrder:
    """A validated intent to open one package.  Carries no size."""

    underlying: str
    family: str
    orientation: str
    tenor_bucket: str
    coordinates: Mapping[str, float]
    raw: str
    defaulted: tuple[str, ...] = ()
    snapped: tuple[str, ...] = ()

    @property
    def strategy_handle(self) -> str:
        return build_strategy_handle(
            self.underlying, self.family, self.orientation, self.tenor_bucket, self.coordinates
        )


@dataclass(frozen=True, slots=True)
class QuoteOrder:
    """A request to price one package *without* opening it.

    Carries exactly the fields of :class:`OpenOrder` because it is the same
    intent asked in a different mood, and :meth:`as_open` hands back the real
    thing.  The quote path resolves that ``OpenOrder`` rather than resolving a
    ``QuoteOrder`` through a parallel code path, so "what you were quoted" and
    "what you would be filled" are the same object flowing through the same
    resolvers -- see ``OptionsEnv._resolve_and_size``.
    """

    underlying: str
    family: str
    orientation: str
    tenor_bucket: str
    coordinates: Mapping[str, float]
    raw: str
    defaulted: tuple[str, ...] = ()
    snapped: tuple[str, ...] = ()

    @property
    def strategy_handle(self) -> str:
        return build_strategy_handle(
            self.underlying, self.family, self.orientation, self.tenor_bucket, self.coordinates
        )

    def as_open(self) -> OpenOrder:
        return OpenOrder(
            underlying=self.underlying,
            family=self.family,
            orientation=self.orientation,
            tenor_bucket=self.tenor_bucket,
            coordinates=self.coordinates,
            raw=self.raw,
            defaulted=self.defaulted,
            snapped=self.snapped,
        )


@dataclass(frozen=True, slots=True)
class CloseOrder:
    position_id: str
    raw: str


@dataclass(frozen=True, slots=True)
class RollBasis:
    """What a roll inherits from the position it replaces.

    Passed in per open id rather than looked up, because the parser must not
    reach into the book: ``parse_action`` is pure, and the tests that drive the
    grammar do not build a ``Book``.  ``coordinates`` is carried rather than
    re-derived from ``strategy_handle`` — the handle renders deltas as integer
    percent, so at any ``delta_step`` finer than 0.01 it is a lossy encoding and
    round-tripping through it would silently move the strike a roll was supposed
    to hold fixed.
    """

    underlying: str
    family: str
    orientation: str
    coordinates: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RollOrder:
    """Close ``position_id`` and reopen the same view at a new point.

    Carries the full replacement intent, so the environment never has to
    re-derive what the new package should be: ``as_open()`` is the ``OpenOrder``
    the reopen resolves, and it is built here so that a roll and an equivalent
    hand-written ``O`` reach ``ContractResolver`` as the same object.
    """

    position_id: str
    underlying: str
    family: str
    orientation: str
    tenor_bucket: str
    coordinates: Mapping[str, float]
    raw: str
    defaulted: tuple[str, ...] = ()
    snapped: tuple[str, ...] = ()

    def as_open(self) -> OpenOrder:
        return OpenOrder(
            underlying=self.underlying,
            family=self.family,
            orientation=self.orientation,
            tenor_bucket=self.tenor_bucket,
            coordinates=self.coordinates,
            raw=self.raw,
            defaulted=self.defaulted,
            snapped=self.snapped,
        )

    @property
    def strategy_handle(self) -> str:
        return self.as_open().strategy_handle


@dataclass(frozen=True, slots=True)
class HoldOrder:
    raw: str = "H"


Order = OpenOrder | CloseOrder | RollOrder | HoldOrder


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """Everything one completion turned into.

    ``rationale`` is kept because the traces are retained for later
    distillation (``⟨Q41⟩``); it is never parsed and never scored.
    """

    orders: tuple[Order, ...] = ()
    errors: tuple[ActionError, ...] = ()
    rationale: str = ""
    raw: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_hold(self) -> bool:
        # A completion that only asks for quotes is *not* an abstain -- it is
        # the first half of a decision.  Counting it as a hold would report the
        # abstain rate of a two-turn arm as roughly double the truth.
        return not self.opens and not self.closes and not self.rolls and not self.quotes

    @property
    def opens(self) -> tuple[OpenOrder, ...]:
        return tuple(o for o in self.orders if isinstance(o, OpenOrder))

    @property
    def quotes(self) -> tuple[QuoteOrder, ...]:
        return tuple(o for o in self.orders if isinstance(o, QuoteOrder))

    @property
    def closes(self) -> tuple[CloseOrder, ...]:
        return tuple(o for o in self.orders if isinstance(o, CloseOrder))

    @property
    def rolls(self) -> tuple[RollOrder, ...]:
        return tuple(o for o in self.orders if isinstance(o, RollOrder))


# ---------------------------------------------------------------------------


def build_strategy_handle(
    underlying: str,
    family: str,
    orientation: str,
    tenor_bucket: str,
    coordinates: Mapping[str, float],
    version: str = HANDLE_VERSION,
) -> str:
    """``AAPL:debit_vertical:bullish:8_30:d55:w10:v2``.

    The handle is what ``ContractResolver`` consumes, so it is also the join key
    between an emitted order and every downstream row.  It is built here rather
    than in the resolver so that exactly one function decides what an order is
    called.

    The version defaults to ``HANDLE_VERSION`` rather than a literal: this used
    to read ``version: str = "v1"``, and a literal in the emitter is exactly how
    the feature layer and the env came to disagree about what ``v1`` meant.
    """
    return ":".join(
        [underlying, family, orientation, tenor_bucket, coordinate_token(family, coordinates), version]
    )


# ---------------------------------------------------------------------------


def _strip_emphasis(line: str) -> str:
    """Remove *matched* markdown emphasis from both ends of a line.

    Only matched pairs are removed, so a prose line that happens to open with a
    backtick ("``O`` means open") keeps its trailing text and stays prose.
    """
    text = line.strip()
    changed = True
    while changed:
        changed = False
        for mark in _EMPHASIS:
            if len(text) > 2 * len(mark) and text.startswith(mark) and text.endswith(mark):
                text = text[len(mark) : -len(mark)].strip()
                changed = True
    return text


def has_quote_lines(text: str) -> bool:
    """Whether this completion asks for prices, decided without config or book.

    The two-turn shape needs this answered *before* anything else runs: a
    completion with no ``Q`` must go straight to ``env.step``, because routing
    it through the quote path would refuse its orders as ``E_QUOTE_TURN`` and
    count its anonymization refusals a second time.  So the test has to be
    cheap and side-effect free -- no chain, no positions, no pseudonyms.

    It shares ``_ORDER_LINE`` and ``_strip_emphasis`` with the parser rather
    than scanning for a leading ``"Q"``, because the two must agree about what
    a quote line is.  A completion writing ``**Q t3 ...**`` is one the parser
    accepts, and a naive prefix check would route it past the quote turn and
    then refuse it as an order -- the policy would have asked a legal question
    and been told it broke the rules.
    """
    for line in text.splitlines():
        stripped = _strip_emphasis(line)
        if not stripped:
            continue
        match = _ORDER_LINE.match(stripped)
        if match is not None and match.group("verb") == "Q":
            return True
    return False


def parse_action(
    text: str,
    config: EnvConfig,
    *,
    open_positions: Mapping[str, RollBasis] | None = None,
    max_orders: int | None = None,
    max_quotes: int | None = None,
) -> ParsedAction:
    """Turn one completion into validated orders plus named rejections.

    ``open_positions`` maps every open position id to what a roll against it
    would inherit.  It is required rather than optional: a close order for a
    position that does not exist has to fail as ``E_UNKNOWN_POS`` at parse time,
    because by the time it reaches the book the distinction between "already
    closed" and "never existed" is gone.

    It is one mapping and not an id list plus a lookup, because ``X`` validates
    coordinates against the *replaced position's* family — arity, bounds and
    topology all depend on it.  Splitting "which ids are open" from "what they
    are" would let the two disagree, and the failure mode is a roll validated
    against the wrong family's coordinate order, which parses cleanly and
    resolves to the wrong strikes.

    ``max_orders`` defaults to ``config.max_orders_per_step`` rather than to a
    literal, so the cap the grammar prints is the cap this enforces.  It stays
    overridable only because the parser tests need to reach the branch without
    building a 9-order completion.  ``max_quotes`` is the same arrangement for
    ``Q``, but it is a **per-underlying** cap, not a per-completion one.
    """
    if max_orders is None:
        max_orders = config.max_orders_per_step
    if max_quotes is None:
        max_quotes = config.max_quotes_per_name
    basis = dict(open_positions or {})
    orders: list[Order] = []
    errors: list[ActionError] = []
    prose: list[str] = []
    known = set(basis)
    # One set for both verbs.  A ``C p07`` and an ``X p07 31_90`` in the same
    # completion are two unwinds of one position, and letting the second through
    # would close it twice -- the first ``book.close`` removes it, so the second
    # either raises or (worse, once rolls reopen) closes the *replacement*.
    closing: set[str] = set()
    tradeable = set(config.universe.tradeable)

    for line in text.splitlines():
        stripped = _strip_emphasis(line)
        if not stripped:
            continue
        match = _ORDER_LINE.match(stripped)
        if match is None:
            prose.append(line)
            continue

        verb = match.group("verb")
        tokens = stripped.split()
        if len(tokens) > _MAX_ORDER_TOKENS:
            errors.append(
                ActionError(
                    "E_GRAMMAR",
                    f"order line has {len(tokens)} fields, at most {_MAX_ORDER_TOKENS} are defined",
                    stripped,
                )
            )
            continue

        if verb == "H":
            if len(tokens) > 1:
                errors.append(ActionError("E_GRAMMAR", "H takes no arguments", stripped))
            else:
                orders.append(HoldOrder(stripped))
            continue

        if verb == "C":
            order_or_error = _parse_close(tokens, stripped, known, closing)
        elif verb == "X":
            order_or_error = _parse_roll(tokens, stripped, config, basis, closing)
        elif verb == "Q":
            # Parsed by ``_parse_open`` and not by a sibling of it: a quote that
            # validated more loosely than the order it previews would quote a
            # package the policy then could not open, which is a worse failure
            # than refusing the quote.
            order_or_error = _parse_open(tokens, stripped, config, tradeable)
            if isinstance(order_or_error, OpenOrder):
                order_or_error = QuoteOrder(
                    underlying=order_or_error.underlying,
                    family=order_or_error.family,
                    orientation=order_or_error.orientation,
                    tenor_bucket=order_or_error.tenor_bucket,
                    coordinates=order_or_error.coordinates,
                    raw=order_or_error.raw,
                    defaulted=order_or_error.defaulted,
                    snapped=order_or_error.snapped,
                )
        else:
            order_or_error = _parse_open(tokens, stripped, config, tradeable)

        if isinstance(order_or_error, ActionError):
            errors.append(order_or_error)
        else:
            orders.append(order_or_error)
            if isinstance(order_or_error, (CloseOrder, RollOrder)):
                closing.add(order_or_error.position_id)

    # The per-name quote budget, enforced before the trading limit and counted
    # apart from it.  Quotes move no cash, so they are not a risk control and
    # must not consume the order allowance; and the cap is per underlying
    # because the round exists to compare structures *within* a name, where one
    # crowded shortlist would otherwise starve the other nine names.
    seen_quotes: dict[str, int] = {}
    over_quota: set[int] = set()
    for order in orders:
        if not isinstance(order, QuoteOrder):
            continue
        seen_quotes[order.underlying] = seen_quotes.get(order.underlying, 0) + 1
        if seen_quotes[order.underlying] > max_quotes:
            over_quota.add(id(order))
            errors.append(
                ActionError(
                    "E_TOO_MANY_QUOTES",
                    f"at most {max_quotes} quotes per name per step",
                    order.raw,
                )
            )
    if over_quota:
        orders = [o for o in orders if id(o) not in over_quota]

    actionable = [
        o for o in orders if not isinstance(o, (HoldOrder, QuoteOrder))
    ]
    if len(actionable) > max_orders:
        for extra in actionable[max_orders:]:
            errors.append(
                ActionError(
                    "E_TOO_MANY_ORDERS",
                    f"at most {max_orders} orders per step",
                    getattr(extra, "raw", ""),
                )
            )
        keep = set(id(o) for o in actionable[:max_orders])
        # ``QuoteOrder`` is named here beside ``HoldOrder`` because it is no
        # longer in ``actionable``: without it, tripping the *order* limit would
        # silently discard every quote on the turn.
        orders = [
            o
            for o in orders
            if isinstance(o, (HoldOrder, QuoteOrder)) or id(o) in keep
        ]

    return ParsedAction(
        orders=tuple(orders),
        errors=tuple(errors),
        rationale="\n".join(prose).strip(),
        raw=text,
    )


def _parse_close(
    tokens: Sequence[str], raw: str, known: set[str], closing: set[str]
) -> CloseOrder | ActionError:
    if len(tokens) != 2:
        return ActionError("E_GRAMMAR", "expected: C <position_id>", raw)
    position_id = tokens[1]
    if not _POSITION_ID.match(position_id):
        return ActionError("E_GRAMMAR", f"{position_id!r} is not a position id", raw)
    if position_id not in known:
        return ActionError("E_UNKNOWN_POS", f"no open position {position_id}", raw)
    if position_id in closing:
        return ActionError("E_DUPLICATE", f"{position_id} is already being closed this step", raw)
    return CloseOrder(position_id, raw)


def _parse_open(
    tokens: Sequence[str], raw: str, config: EnvConfig, tradeable: set[str]
) -> OpenOrder | ActionError:
    if len(tokens) not in (5, 6):
        return ActionError("E_GRAMMAR", "expected: O <t> <fam> <or> <tenor> [<coords>]", raw)

    _, ticker, family_code, orientation_code, tenor_token, *rest = tokens

    if ticker not in tradeable:
        # Observed-but-not-tradeable is its own code even though the shipped
        # universe currently has no such name -- ``observed`` and ``tradeable``
        # are identical since SPY replaced SPX, and SPX was dropped rather than
        # demoted.  The distinction is kept because the two mistakes are
        # different: trying to trade a name the policy can see is a scope error,
        # inventing a ticker is a hallucination, and collapsing them would hide
        # which one a run is making.  ``UniverseSpec`` permits the split, so a
        # config can reintroduce it without touching this file.
        code = "E_NOT_TRADEABLE" if ticker in config.universe.observed else "E_UNKNOWN_NAME"
        return ActionError(code, f"{ticker} is not tradeable", raw)

    family = FAMILY_CODES.get(family_code)
    if family is None:
        return ActionError("E_UNKNOWN_FAMILY", f"{family_code!r} is not a family code", raw)
    if family not in config.admitted_families:
        return ActionError("E_UNKNOWN_FAMILY", f"{family} is not admitted", raw)

    orientation = ORIENTATION_CODES.get(orientation_code)
    if orientation is None:
        return ActionError("E_ORIENTATION", f"{orientation_code!r} is not an orientation", raw)
    allowed = TEMPLATE_SPECS[family].orientations
    if orientation not in allowed:
        return ActionError(
            "E_ORIENTATION", f"{family} admits {'/'.join(allowed)}, not {orientation}", raw
        )

    # Required rather than optional, and the evidence is the coordinate channel
    # sitting right below this.  Across the 957 opens of ``policy_zeroshot.all``,
    # 601 omitted coordinates entirely, 352 emitted values numerically equal to
    # the default, and 4 were genuinely different -- of which 2 were rejected on
    # bounds.  Two orders in 957 used the channel.  An optional tenor would have
    # been the same dead parameter, and the arm would have cost a full run to
    # learn nothing about tenor.  The price is E_TENOR rejects in the opening
    # steps, which are cheap and land in the ledger where they can be counted.
    if tenor_token not in config.tenor.admitted:
        return ActionError(
            "E_TENOR",
            f"{tenor_token!r} is not an admitted tenor; "
            f"expected one of {' '.join(config.tenor.admitted)}",
            raw,
        )

    parsed = _parse_coordinates(
        family, rest, config, raw, base=config.coordinates.defaults_for(family)
    )
    if isinstance(parsed, ActionError):
        return parsed
    coordinates, defaulted, snapped = parsed

    return OpenOrder(
        underlying=ticker,
        family=family,
        orientation=orientation,
        tenor_bucket=tenor_token,
        coordinates=coordinates,
        raw=raw,
        defaulted=defaulted,
        snapped=snapped,
    )


def _parse_roll(
    tokens: Sequence[str],
    raw: str,
    config: EnvConfig,
    basis: Mapping[str, RollBasis],
    closing: set[str],
) -> RollOrder | ActionError:
    """``X <id> <tenor> [<coords>]`` — replace one position with another.

    Reuses ``_parse_coordinates`` with the replaced position's coordinates as
    the base rather than the family defaults, which is the whole difference
    between "move this position's expiry" and "open a fresh package that happens
    to be the same family".

    The tenor is required for the same reason it is on ``O``, and *not* rejected
    for being unchanged: the canonical roll is ``X p07 8_30`` on a position that
    was opened at ``8_30`` and has decayed to 9 DTE.  The bucket is identical,
    the expiry is not, and refusing the no-op-looking case would refuse the only
    case that matters.
    """
    if len(tokens) not in (3, 4):
        return ActionError("E_GRAMMAR", "expected: X <position_id> <tenor> [<coords>]", raw)

    _, position_id, tenor_token, *rest = tokens
    if not _POSITION_ID.match(position_id):
        return ActionError("E_GRAMMAR", f"{position_id!r} is not a position id", raw)
    replaced = basis.get(position_id)
    if replaced is None:
        return ActionError("E_UNKNOWN_POS", f"no open position {position_id}", raw)
    if position_id in closing:
        return ActionError(
            "E_DUPLICATE", f"{position_id} is already being closed this step", raw
        )

    # Checked even though the position proves the family was admitted when it
    # opened: ``admitted_families`` is per-arm config, so a book carried into an
    # arm that no longer admits a family would otherwise let a roll reintroduce
    # it -- an open the same arm would reject.
    if replaced.family not in config.admitted_families:
        return ActionError(
            "E_UNKNOWN_FAMILY", f"{replaced.family} is not admitted", raw
        )
    if tenor_token not in config.tenor.admitted:
        return ActionError(
            "E_TENOR",
            f"{tenor_token!r} is not an admitted tenor; "
            f"expected one of {' '.join(config.tenor.admitted)}",
            raw,
        )

    parsed = _parse_coordinates(
        replaced.family, rest, config, raw, base=replaced.coordinates
    )
    if isinstance(parsed, ActionError):
        return parsed
    coordinates, defaulted, snapped = parsed

    return RollOrder(
        position_id=position_id,
        underlying=replaced.underlying,
        family=replaced.family,
        orientation=replaced.orientation,
        tenor_bucket=tenor_token,
        coordinates=coordinates,
        raw=raw,
        defaulted=defaulted,
        snapped=snapped,
    )


def _parse_coordinates(
    family: str,
    rest: Sequence[str],
    config: EnvConfig,
    raw: str,
    *,
    base: Mapping[str, float],
) -> tuple[Mapping[str, float], tuple[str, ...], tuple[str, ...]] | ActionError:
    """The coordinate half of ``O`` and ``X``, which differ only in ``base``.

    ``defaulted`` names the coordinates the caller did not state, and it is
    reported even when ``base`` is a previous position rather than the family
    defaults: "this came from the position I replaced" and "this came from the
    family default" are both *not chosen by the policy this step*, which is what
    the ledger's ``defaulted`` column exists to say.
    """
    declared = coordinate_order(family)
    coordinates: dict[str, float] = dict(base)
    defaulted: list[str] = []
    snapped: list[str] = []

    if rest:
        values = rest[0].split("/")
        if len(values) > len(declared):
            return ActionError(
                "E_COORD_ARITY",
                f"{family} takes at most {len(declared)} coordinates "
                f"({'/'.join(n for n, _ in declared)}), got {len(values)}",
                raw,
            )
        for (name, _), token in zip(declared, values, strict=False):
            if token in ("", "-", "_"):
                defaulted.append(name)
                continue
            try:
                percent = float(token)
            except ValueError:
                return ActionError("E_GRAMMAR", f"{token!r} is not a delta in percent", raw)
            value = percent / 100.0
            low, high = config.coordinates.bounds[name]
            if not low <= value <= high:
                return ActionError(
                    "E_COORD_BOUNDS",
                    f"{name}={value:.2f} outside [{low:.2f}, {high:.2f}]",
                    raw,
                )
            quantized = _snap(value, config.coordinates.delta_step)
            if quantized != value:
                snapped.append(f"{name}->{quantized:.2f}")
            coordinates[name] = quantized
        defaulted.extend(name for name, _ in declared[len(values) :])
    else:
        defaulted.extend(name for name, _ in declared)

    invalid = _check_topology(family, coordinates)
    if invalid is not None:
        return ActionError("E_COORD_BOUNDS", invalid, raw)

    return coordinates, tuple(defaulted), tuple(snapped)


def _snap(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 10)


def _check_topology(family: str, coordinates: Mapping[str, float]) -> str | None:
    """Reject coordinate sets that are individually in range but jointly absurd.

    Per-coordinate bounds cannot catch these: a credit vertical with a 0.10
    short delta and a 0.30 width asks for a long leg at delta −0.20, which does
    not exist.  Caught here rather than in ``ContractResolver`` so that the
    policy gets a coordinate-shaped error instead of a resolution failure it
    cannot act on.
    """
    if family in ("credit_vertical", "iron_condor"):
        if coordinates["short_delta"] - coordinates["width_delta"] <= 0.01:
            return "short_delta - width_delta leaves no long leg"
    if family == "iron_butterfly":
        if coordinates["short_delta"] - coordinates["wing_delta"] <= 0.01:
            return "short_delta - wing_delta leaves no wing"
    if family == "debit_vertical":
        if coordinates["long_delta"] - coordinates["width_delta"] <= 0.01:
            return "long_delta - width_delta leaves no short leg"
    if family == "butterfly":
        center = coordinates["center_delta"]
        if center - coordinates["upper_width_delta"] <= 0.01:
            return "center_delta - upper_width_delta leaves no upper wing"
        if center + coordinates["lower_width_delta"] >= 0.99:
            return "center_delta + lower_width_delta exceeds delta 1"
    if family == "defined_risk_reversal":
        if coordinates["directional_delta"] <= coordinates["tail_wing_delta"]:
            return "tail_wing_delta must be further out than directional_delta"
    return None
