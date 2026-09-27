"""``state_space.v1`` — the encoding specified by ``docs/state_space.md``.

Ten market fields per name, positional, scaled integers, schema declared once.
Every scale and every field here traces to that document's section 4 table; the
measured token counts it reports (30 tok/name, 184 tok for the six-name block,
37 for ``ACCT``, 23 per ``POS`` row) are the acceptance criterion for this
module, and ``tests/test_statespace_v1.py`` asserts the worked example
byte-for-byte.

Three places where this implementation is stricter than the document, all of
them because the document's illustrative rows disagree with its own rules:

* **Signed fields always carry a sign, including ``+0``.**  The worked index row
  in section 6.2 writes ``0`` for ``ret`` and ``dv`` while the ``NVDA`` row
  writes ``+0`` for ``dv``.  (That block is labelled ``SPX``; it predates the
  SPY swap and section 6.2 says so.  The inconsistency it illustrates is in the
  formatting, not the ticker, so it survives the swap untouched.)  A policy
  reading a column that is sometimes ``0`` and sometimes ``+0`` will look for
  meaning in the difference.  One convention, applied everywhere.
* **``fi`` is ``na`` below ``MIN_FLOW_TRADES`` prints.**  Section 0 rule C1:
  the obvious guard, ``contract_count > 0``, is inverted — it admits the
  thinnest windows, where the ratio is pinned at ±1.0 by a single lot, and
  marks the rest missing.  Either way the field is effectively PM-only for the
  nine single names; the difference is whether the AM column says so.
* **``dv`` is ``na`` at the AM session**, not absent.  The column is null on
  all 378 AM rows (section 4.1).  Emitting the field as ``na`` keeps the row
  arity fixed, which is what makes a positional encoding safe.

The AM/PM asymmetry is encoded rather than resolved.  ``⟨Q7h⟩`` proposes
collapsing to PM only, and ``EnvConfig.grid.decision_sessions`` already
defaults to ``("PM",)`` for an unrelated and harder reason — the option quote
snapshot exists only at 20:00 UTC.  This module stays correct either way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..data.context_catalysts import CATALYST_CODES
from ..data.features.strategy_templates import TEMPLATE_SPECS, coordinate_order
from .datasets import DatasetError
from .spec import SIZE_RULE_LIMITS, EnvConfig
from .statespace import (
    BookView,
    EpisodeContext,
    Observation,
    PositionView,
    StepContext,
    estimate_tokens,
    register_state_space,
    render_rows,
    scaled,
)

STATE_SPACE_ID = "state_space.v1"

#: The tool the ``Q`` verb becomes on a provider that speaks tool calls.  Named
#: once, here, because the policy sends it and the policy's normalizer matches
#: on it coming back -- a literal on both sides is a rename away from a silent
#: "the model called an unknown tool", which looks like a refusal to quote.
QUOTE_TOOL_NAME = "quote_package"


def _orientation_codes(family: str) -> tuple[str, ...]:
    """The wire codes for the orientations one family admits.

    ``TEMPLATE_SPECS`` states orientations in full words and the wire carries
    single letters, so the inversion has to happen somewhere; it happens here
    once rather than in the schema builder, where it would be an inline
    comprehension nobody could test on its own.
    """
    from .actions import ORIENTATION_CODES

    allowed = TEMPLATE_SPECS[family].orientations
    return tuple(code for code, full in ORIENTATION_CODES.items() if full in allowed)


def _quote_grammar(config: EnvConfig) -> list[str]:
    """How the prompt teaches the quote round, per channel.

    Three cases and no overlap between them.  Off: nothing, because a verb the
    runner will not answer must not be advertised.  ``"text"``: the full ``Q``
    line and its semantics, which is the only place they are stated.
    ``"tool"``: the *turn structure* only -- the tool's own description already
    carries the argument catalogue, the per-name cap and the fields that come
    back, and restating them here would be two strings the model reads that can
    drift apart.  What the description cannot say is the thing that belongs to
    this environment rather than to the tool: calling it spends the step's
    first turn, so nothing trades on that turn and the order goes on the next
    one.  A model that does not know this pairs a call with order lines and
    loses the orders.
    """
    if not config.quotes_enabled:
        return []
    if config.quote_channel == "tool":
        return [
            f"to price a package before ordering it, call {QUOTE_TOOL_NAME}",
            "  a turn that calls it makes no trades: order lines on that same"
            " turn are",
            "  refused. the prices come back and you order on the next turn",
            "  quoting is free, commits you to nothing, and a package you never"
            " quoted",
            "  may still be opened",
        ]
    return [
        "Q <t> <fam> <or> <tenor> <coords>  price a package without opening it",
        # The verb is worth its tokens only if the policy knows it is free,
        # that it is a separate turn, and what comes back.  A model that thinks
        # Q trades will not use it; a model that mixes Q with O loses the
        # orders.
        f"  ask for up to {config.max_quotes_per_name} per name per step."
        " quoting costs nothing and",
        "  moves nothing -- no position is opened and no cash moves",
        "  a turn containing any Q sends *only* Q. orders on that turn are",
        "  refused; you get the quotes back and then order on the next turn",
        "  each answer is: R OK <your line> q<qty> m<premium> c<cost> e<dte>"
        " d<deltas>",
        "  q is the size the resolver picked for you, and m and c are both",
        "  quoted at that size. c is the entry half of the round trip, the",
        "  same number the fill will charge. m is the mid you would pay or",
        "  receive. the ratio c/m is the hurdle that package has to clear",
        "  a quote is not a commitment and an unquoted package may still be"
        " opened",
    ]


DATASET_UNDERLYING = "underlying_market_features"
#: ``iv``/``ts``/``sk``/``bf`` are defined at delta coordinates -- ATM is the
#: 50-delta call, the wings are 25-delta -- and the chain carries ``model_delta``
#: and ``project_iv`` per contract, so the point is read off the ladder rather
#: than off a fit.  ``option_iv_surface_features`` answered the same question by
#: subsampling contracts into buckets (six delta bins on one path, seven
#: log-moneyness bins on the other), fitting a smile, then evaluating the fit at
#: the delta point.  The buckets were never the coordinate system; they only
#: chose which quotes entered the fit, and having two disagreeing bucketings is
#: what made the feature table's source mix look like a coverage ceiling when
#: the raw feeds in fact reach the same coordinates.  Field names are unchanged,
#: so this is a swap of provenance and not of meaning.
DATASET_SURFACE = "option_delta_point_surface"
DATASET_FLOW = "compact_option_flow_states"
DATASET_OI = "option_open_interest_features"
#: The ``N`` block, materialized by ``jobs/build_news_catalyst_rows.py``.
#:
#: Read as an ordinary feature dataset rather than assembled here, for the same
#: reason every other block is: the state space is not allowed to know how a
#: corpus is selected from, only how a row is rendered.  Two properties it
#: depends on and does not implement -- that an item is placed at the first
#: *session* whose decision time it precedes (a per-date selection leaks into
#: AM on 20.7% of rows) and that each item is emitted once rather than re-served
#: while still "recent" -- belong to the builder.  What survives here is the
#: check: the row carries the item's own ``available_time``, so
#: ``assert_point_in_time`` raises on a leak rather than the leak being an
#: argument between two docstrings.
DATASET_NEWS = "news_catalyst_rows"

#: Stands in for the ``N`` block on a date whose news partition is absent.
#:
#: This is the one dataset whose absence is tolerated, and it is tolerated only
#: because it is the one dataset whose *rows* are already optional: a name with
#: no item renders no row, so a missing partition is indistinguishable from a
#: quiet day unless something says otherwise.  That indistinguishability is the
#: whole problem — the system block tells the policy that an absent row means
#: "nothing new since the last step", and on a date with no corpus that is a
#: false statement the policy has no way to detect.  So the tolerance is a
#: *row*, not a shrug: the block still renders, and it opens by withdrawing the
#: claim the system block made.
#:
#: Deliberately not generalized to the numeric datasets.  ``M`` has fixed arity,
#: so their absence would render ``na`` in cells the policy reads as measured
#: quantities, and a plausible ``na`` is worse than a crash.  ``DatasetError``
#: continues to propagate for all five of them.
#:
#: The ticker cell is ``*`` because no universe label is ever ``*``, anonymized
#: or not, so the row cannot be mistaken for a name's own item; the remaining
#: cells are legal codes so the row still parses under the ``N`` grammar.
NEWS_UNAVAILABLE = "N * ot ? ? no news feed for this date - absence of N rows below means nothing"

#: ``dp`` in ``docs/state_space.md`` R2 — one unit is 10 bp of vol or return.
SCALE_DP = 1000.0
#: Percentages (``fi``, ``doi``, ``util``, ``dd``, and the %NLV greeks).
SCALE_PCT = 100.0
#: ``$k`` for the account line, ``$100s`` for its PnL columns.
SCALE_KDOLLAR = 0.001
SCALE_HDOLLAR = 0.01

#: Minimum prints in the window before ``fi`` is emitted — section 0 rule C1.
#: See :meth:`StateSpaceV1._flow_imbalance` for why the value is not delicate.
MIN_FLOW_TRADES = 25

FAMILY_CODES: Mapping[str, str] = {
    "outright": "ol",
    "debit_vertical": "dv",
    "credit_vertical": "cv",
    "defined_risk_reversal": "dg",
    "butterfly": "bf",
    "long_straddle": "ls",
    "long_strangle": "lg",
    "iron_butterfly": "ib",
    "iron_condor": "ic",
}

ORIENTATION_CODES: Mapping[str, str] = {
    "bullish": "b",
    "bearish": "r",
    "neutral": "n",
}

#: **Deliberately absent: a per-family round-trip cost table.**
#:
#: One lived here.  It gave the round-trip half-spread as a percent of net
#: premium for each family (``ol 2  dv 10  cv 9 ... dg 13``) and was rendered
#: into ``system_block``.  It is gone because it was measured on
#: ``runs/policy_zeroshot.all``, whose window *is* the evaluation window
#: 2024-09-03..2025-08-29.  There is no earlier window to measure it on: the
#: chain holds zero dates strictly before 2024-09-03.  So the table could not be
#: made point-in-time, and handing it to the policy at ``T 0`` told it, before
#: its first order, what execution would cost over the year it was about to be
#: scored on.
#:
#: The defence offered at the time -- that per-episode medians are stable, so
#: the numbers are a *family* fact rather than a *window* fact -- does not
#: survive.  Stability was itself established by reading the evaluation window;
#: it is a conclusion drawn from the held-out data, not an argument for being
#: allowed to draw it.  The correct test is provenance, not magnitude: no
#: quantity derived from dates inside the window may enter the prompt, however
#: small its variance looks once you have looked.
#:
#: What replaces it is the two things the policy can legitimately have.  The
#: **mechanism** is disclosed in ``system_block`` -- you cross half the bid-ask
#: on every leg, entering and again exiting -- which is a rule of the
#: environment, true on day one and derived from no data at all.  The
#: **magnitude** is disclosed by ``_result_block`` as ``c<dollars>`` on each
#: fill, learned at the moment it is paid, which is the only schedule on which
#: a trader learns it.
#:
#: The leakage-free way to restore an *ex ante* estimate is a liquidity field on
#: the ``M`` row computed from that date's own chain.  That is a state-space
#: change, not a prompt change.  Do not reintroduce a constant table.
#:
#: Guarded by ``test_the_system_block_quotes_no_measured_cost_numbers``.
MISSING = "na"

#: The ``cat`` enum, rendered once into ``system_block``.  Imported rather than
#: restated so the wire codes and the corpus's own taxonomy cannot drift apart.
_CATALYST_CODES = "  ".join(f"{code} {label}" for code, label in CATALYST_CODES)


def _pct(fraction: float) -> str:
    """A config fraction as the percentage the prompt states it in.

    ``%g`` and not a fixed precision, because these span three orders of
    magnitude -- 0.5% for the scenario budget, 10% for the premium fraction --
    and a fixed ``.1f`` would print the band as ``0.5%`` but a Whalley-Wilmott
    clamp as ``0.0%``.  It also drops the binary dust: ``0.1 * 100`` is
    ``10.000000000000002``, which ``%g`` renders as ``10``.
    """
    return f"{fraction * 100:g}%"


@dataclass(frozen=True, slots=True)
class _MarketFields:
    """The ten fields of one name, already scaled to wire integers."""

    ticker: str
    cells: tuple[str, ...]
    missing: tuple[str, ...]


class StateSpaceV1:
    """Serializer for ``docs/state_space.md``.

    Holds no mutable state: one instance can serve every episode of a run, and
    two instances built from the same ``EnvConfig`` produce identical bytes.
    That is required by R7 — the system and episode blocks must be stable for
    prefix caching to bill only the per-step delta.
    """

    def __init__(self, config: EnvConfig) -> None:
        self._universe = tuple(config.universe.observed)
        self._tradeable = tuple(config.universe.tradeable)
        self._admitted = tuple(config.admitted_families)
        #: The ablation switch, read once so every method agrees.  See
        #: ``FeatureFlags.suppress_textual_context``.
        self._textual = not config.flags.suppress_textual_context
        unknown = [f for f in self._admitted if f not in FAMILY_CODES]
        if unknown:
            raise ValueError(f"{STATE_SPACE_ID} has no wire code for families: {unknown}")

    @property
    def state_space_id(self) -> str:
        return STATE_SPACE_ID

    def required_datasets(self) -> tuple[str, ...]:
        numeric = (
            DATASET_UNDERLYING,
            DATASET_SURFACE,
            DATASET_FLOW,
            DATASET_OI,
        )
        # Dropped rather than fetched-and-discarded.  ``required_datasets`` is
        # what the coverage pre-flight checks and what lands in each step's
        # provenance, so an ablation arm that kept the news dataset here would
        # report gaps on rows it never renders and would claim, in its own
        # ledger, to have read a context it did not have.
        return (numeric + (DATASET_NEWS,)) if self._textual else numeric

    # -- once per episode ------------------------------------------------

    def system_block(self, config: EnvConfig) -> str:
        """R1: the schema, declared once, ~120 tokens.

        The three semantic notes below the units table are not decoration.
        ``ret`` measures a different object at each session and ``iv`` is a
        30-day tenor while the action space trades 8–30; both are stated
        because section 4 is explicit that the policy must not be left to infer
        them from the numbers.

        The units table is one line per *unit* rather than one per field, and it
        carries a decoded example on each line.  It used to be two lines of
        grouped assignments with no worked value, which left two things to
        infer: that ``0.1%`` meant the integer was in tenths of a percent rather
        than a percentage of something, and that ``$k`` meant the wire value was
        thousands of dollars.  A live trace shows the model getting the sign and
        the ranking of the fields right and the magnitudes wrong, which is what
        an unanchored scale produces.  Every field named in the three schema
        lines appears exactly once below them; ``tests/test_statespace_v1.py``
        asserts that, because a field that quietly loses its unit reads as
        dimensionless rather than as an error.

        **The objective and the cost mechanism are stated last and on purpose.**
        Everything above them is schema; without these two the block describes
        a format and never says what the format is for.  A 492-decision
        zero-shot run on the schema-only block reasoned "it seems the user is
        simulating an options portfolio", "it's likely a trading game", and
        then fell back on imitating its own transcript at 94% of steps -- the
        prompt was measuring its own omission.  That run also paid 297k of
        frictions against a 162k loss while asking, in its own reasoning, where
        the money had gone.

        **The cost lines state the rule and withhold the number**, and the split
        is the point.  "You cross half the bid-ask on every leg, entering and
        again exiting" is a rule of the environment: true on the first day,
        derived from no data, knowable before any of the evaluation window has
        happened.  *How much* that comes to per family is a measurement, and
        every date available to measure it on lies inside the window being
        scored -- so it is withheld here and revealed by ``_result_block`` as
        ``c<dollars>`` at the moment each fill pays it.  The block says which of
        the two it is doing rather than leaving the omission to look like an
        oversight, which is the failure the schema-only block already
        demonstrated.

        The ``Q`` verb moves the *number* earlier without moving the
        *measurement* earlier, and the distinction is what keeps this legal.  A
        quote is one package, priced off the decision-time chain slice, at the
        decision-time timestamp -- the same bytes the fill would read one turn
        later.  It contains nothing from a later date and nothing aggregated
        over the scored window.  What stays withheld is the statistic: "iron
        condors cost 13.9-16.5% of premium" is a fact about the evaluation
        window and never enters the prompt, before or after the trade.  The
        policy may learn that ratio by quoting, which is measurement it performs
        itself out of information it already holds; it may not be handed it.

        **A unit is not a meaning.**  The ``M`` fields have carried one-line
        definitions since the first version; the ``A`` and ``P`` fields carried
        only a scale, so ``bp``, ``util``, ``dd``, ``nD`` and ``nV`` reached the
        policy as numbers with a size and no referent.  Two of them are actively
        misleading if guessed: ``rlz`` accumulates over the whole *run* because
        the book is carried across episode boundaries, so a policy reading it as
        this month's PnL mistakes a stale loss for a fresh one; and ``nlv``
        already contains the collateral that ``bp`` nets out, so a policy
        reading ``bp`` as its account size sizes against the wrong number.
        Neither is derivable from the digits.  ``tests/test_statespace_v1.py``
        now requires every field in the three schema lines to be *defined* here
        and not merely dimensioned.

        The first three lines must stay the three schema lines: the unit tests
        read ``lines[:3]`` for the field list, and the units table by the
        ``two spaces + exactly two pipes`` shape.  New lines must avoid that
        shape or they will be parsed as unit declarations.  A definition is
        parsed as ``name: prose`` (or ``name name: prose``) within a
        two-space-delimited segment, so a new line that wants to define a field
        has to use that shape.
        """
        families = "  ".join(f"{FAMILY_CODES[f]} {f}" for f in self._admitted)
        news = self._textual
        return "\n".join(
            [
                line
                for line in [
                "M t ret rv iv dv w ts sk bf fi doi",
                "P id t fam or dte qty mark upnl dD src",
                "A nlv cash bp util rlz unrlz dd npos nD nV",
                "N t cat dir hzn clause" if news else None,
                "every row starts with its block letter, then the fields above in order",
                "all values are integers. unit | fields | one decoded example:",
                "  0.1%  | ret rv iv dv w ts sk bf | iv 388 = 38.8% vol",
                "  1%    | fi doi util dd dD nD nV | dD -36 = -36% of nlv",
                "  $1000 | nlv cash bp             | nlv 1000 = $1,000,000",
                "  $100  | rlz unrlz               | rlz +52 = +$5,200",
                "  $0.01 | mark                    | mark 88000 = $880.00 per package",
                "  $1    | upnl                    | upnl -125 = -$125",
                "  count | dte qty npos            | dte 4 = 4 days, qty 4 = 4 packages",
                (
                    "  label | id t fam or cat dir hzn src | id p07 = close it with C p07"
                    if news
                    else "  label | id t fam or src         | id p07 = close it with C p07"
                ),
                "  text  | clause                  | clause = one sentence, as published"
                if news
                else None,
                f"{MISSING} = not available at this step",
                "ret: AM = prev close to open, PM = open to close",
                "rv: 21d realized close-to-close, as of T-1 at AM, as of T at PM",
                "iv: ATM 30d, whatever tenor you trade. read it as the vol regime",
                "    of the name, not as the price of the contract being opened",
                "w: iv - rv.  ts: atm_iv_90d - atm_iv_30d.  dv: change in iv this session",
                "sk: 25d put iv - 25d call iv, 30d.  bf: mean(25d wings) - atm, 30d",
                "fi: delta-weighted signed flow imbalance.  doi: OI change vs prior, as of T-1",
                f"fam: {families}",
                "or: b bullish  r bearish  n neutral",
                # The only place a roll survives.  The R receipt that named both
                # ids scrolls out of the append-only context after a few steps,
                # so without this cell a rolled position is indistinguishable
                # from a fresh one and the policy loses the difference between
                # "I have been wrong about this name for six weeks" and "I
                # opened it on Tuesday" -- which is the whole question a roll
                # decision turns on.
                "src: absent if you opened this fresh. <p03g2 = it replaced p03 and is",
                "     the 2nd package in that chain. the earlier ones are closed and"
                " their",
                "     pnl is already in rlz, so g counts how long you have held the"
                " view,",
                "     not how long this contract has been open",
                *(
                    [
                        "cat: the kind of event.  dir: which way the source reads it",
                        "hzn: how soon it resolves.  clause: the source's own words, truncated",
                        f"cat  {_CATALYST_CODES}",
                        "dir  + bullish  - bearish  0 neutral  ? mixed or unknown",
                        "hzn  0 intraday  1 1-5d  2 6-21d  3 gt-21d  ? unknown",
                        "an N row is one item, shown on the first step after it became public",
                        "  and never repeated. a name can carry several rows, oldest first, when",
                        "  more than one item landed since the last step. no N row for a name",
                        "  means nothing new since then - not that the name has no news. it is a",
                        "  changelog, not a feed.",
                    ]
                    if news
                    else []
                ),
                (
                    "t: the name's label, see below.  id: the position's handle,"
                    " close it with C <id>"
                    if config.flags.anonymize
                    else "t: the ticker.  id: the position's handle, close it with C <id>"
                ),
                "dte: calendar days from this step to that position's expiry",
                "qty: packages held.  mark: mid value of one package.  upnl: mark - entry",
                "dD: that position's dollar delta, as a % of nlv",
                "nlv: cash + mark of everything open. the objective scores this and only this",
                "cash: settled cash, collateral included.  bp: cash not pledged as collateral",
                "util: collateral / nlv.  dd: nlv / peak nlv - 1, never positive",
                "rlz: closed-trade cash pnl, costs included, since the run began -"
                " not since this episode began",
                "unrlz: mark - entry across the whole book.  npos: how many P rows follow",
                "nD nV: net dollar delta and net dollar vega of the book, each as a % of nlv",
                "objective: maximise log(nlv at the last step / nlv at the first),"
                " net of costs",
                "cost: you cross half the bid-ask on every leg, entering and again"
                " exiting.",
                "  it is charged in cash at the fill and is already inside rlz and nlv.",
                "  it varies by family, by name and by day"
                + (
                    ", and you can see it before you trade: price"
                    if config.quotes_enabled
                    else ", and is not shown before you trade."
                ),
                # The sentence above used to end "and is not shown before you
                # trade", which the Q verb makes false.  A prompt that
                # understates what the policy can see costs more than one that
                # overstates it: the model reads the disclaimer, concludes the
                # number is unknowable, and never asks.
                *(
                    [
                        "  a package with "
                        + ("Q" if config.quote_channel == "text" else QUOTE_TOOL_NAME)
                        + " first. quoting is free and commits you to nothing."
                    ]
                    if config.quotes_enabled
                    else []
                ),
                "  a filled order echoes back as R OK <order> <id> q<n> m<n> c<n>,"
                " where",
                "  q is the size the resolver gave you, m is the mid, and c is what"
                " that",
                "  fill cost you in dollars. half at the open, half at the"
                " close.",
                "  so a package must recover its c before it breaks even, and holding"
                " one",
                "  step and closing it pays the whole round trip for nothing.",
                *self._sizing_lines(config),
                *self._hedging_lines(config),
                *self._provenance_lines(config),
                *self._anonymization_lines(config),
                ]
                # ``None`` marks a line that belongs to the textual context and
                # is dropped with it.  Filtering here rather than at each site
                # keeps the block readable as the single ordered document it is.
                if line is not None
            ]
        )

    def _provenance_lines(self, config: EnvConfig) -> list[str]:
        """Say that the step is the only admissible source. Off by default.

        The "suppress at source" remedy for teacher-trace leakage. The audit
        found teachers reciting things the step never gave them -- an earnings
        date, an election outcome -- so this names the boundary instead of
        hoping it is inferred.

        **Why it may be said here at all**, when the cost magnitudes may not:
        this block already draws the line between a rule of the environment and
        a measurement taken on the scored window, and only the second is
        withheld. These lines quote no price, no date, no outcome and no
        measured constant. They are a rule, and a rule that carries none of the
        hindsight it asks the model to set aside.

        **Why it is phrased as "not evidence" rather than "you do not know".**
        The second is false and the model can tell it is false, which invites it
        to discount the whole instruction. The first is true regardless of what
        the model knows and asks for something it can actually do: decline to
        act on the memory. A trace that says "I recall NVDA reports on the 20th,
        but that is not in this step, so I will not size around it" is a
        *success* of this flag, and the auditor will still flag it -- which is
        why the flag is evaluated on the action, not on the prose.

        No line here may name a ticker, a date or a number. Doing so would make
        the suppression itself the leak, and
        ``test_the_provenance_section_leaks_nothing_it_forbids`` fails if one
        appears.
        """
        if not config.flags.suppress_future_knowledge:
            return []
        return [
            "provenance of fact, and it binds every order you send",
            "  this step is the whole of what you are given. decide from the rows"
            " above",
            "  and from the receipts of your own earlier orders, and from nothing"
            " else.",
            "  you may recall how some name or event turned out around now. that"
            " recollection",
            "  is not evidence here and must not enter the decision, however"
            " confident it",
            "  feels. if a fact is not in this step, you do not have it. reason"
            " from the",
            "  regime the numbers describe, not from the ending you remember.",
        ]

    def _anonymization_lines(self, config: EnvConfig) -> list[str]:
        """Declare the de-identified encoding.  Off by default.

        Without this section the arm is not an ablation, it is a broken prompt.
        The block above defines ``t`` as the ticker and shows the ``T`` header
        carrying a date; under ``flags.anonymize`` neither is true, and a policy
        reading ``T 4 m11 PM`` against an undocumented schema has to guess
        whether ``m11`` is a date, a session id or a tenor.  Every substitution
        the layer makes is therefore named here.

        **It says the labels are redrawn, and that is deliberate.**  The
        alternative -- letting the policy assume a label means the same name
        next month -- would have it carry a view across a boundary where the
        mapping changed, which is a *wrong* decision rather than a cautious one.
        Telling it the truth costs nothing that was being protected: knowing
        that the map is redrawn is not knowing the map.

        **It does not say which names are in the universe**, even though the
        default universe is in a public config file.  The point of the layer is
        that a step cannot be matched against a memory of a name on a date, and
        a ten-name roster in the prompt is most of the way to re-identification
        by elimination once the ``M`` rows are read.

        No line here may name a ticker or a date, for the same reason
        ``_provenance_lines`` may not: a legend that leaks is worse than no
        legend, because the leak arrives in the block that is byte-identical
        across every step of every episode.
        """
        if not config.flags.anonymize:
            return []
        return [
            "names and dates are withheld. the encoding is stated here in full",
            "  t: a label like U03, not a ticker. the same label means the same"
            " name",
            "    for the whole of this episode and is redrawn for the next one, so"
            " do",
            "    not carry a view from a label to that label later. what you own"
            " is",
            "    restated by id at every boundary, and that restatement is what"
            " binds.",
            "  the T row reads T <step> m<n> <session>, where n is calendar days"
            " from",
            "    this step to the standard monthly expiry, the third friday of the"
            " month.",
            "    it counts down to 0 on expiry and jumps back to about a month the"
            " next",
            "    session, so it tells you where in the option cycle you are and"
            " how much",
            "    time the tradeable tenors have. it is not a date and two steps"
            " with the",
            "    same n are not the same step. step, which never repeats, is what"
            " orders",
            "    the episode.",
            "  <ent> is a company, person or place the source named. every one of"
            " them",
            "    is the same token, so <ent> in two rows may be two different"
            " things.",
            "  <d> and <y> are a date and a year that were removed.",
            "  order in the labels carries nothing: rows are sorted by label, not"
            " by",
            "    size, sector or any ranking.",
        ]

    def _sizing_lines(self, config: EnvConfig) -> list[str]:
        """How ``SizeResolver`` turns an intent into a package count.

        Size left the action space, so the policy expresses a view and the
        resolver decides the quantity.  Until now the prompt said only that
        ("size is absent on purpose", in ``action_grammar``), which left the
        policy told that something else picks the number and never told what
        that something reads.  The consequences it could not infer are the ones
        that change what to *send*: that a convex package is allowed to be
        larger than a concave one at the same budget, that a credit structure is
        bounded by its gross premium rather than by the credit it collects, and
        that an ``E_LIMIT`` naming ``cash`` calls for a different next order
        than one naming ``max_positions_per_underlying``.

        **Every magnitude is rendered from ``EnvConfig``, and the section is
        built from ``SIZE_RULE_LIMITS`` rather than written out.**  The limits
        are a per-rule tuple: ``nav_fraction`` runs two of them, ``scenario``
        three, ``full`` six.  A hand-written paragraph would describe one arm
        and be shipped to all three -- telling a ``nav_fraction`` run about a
        scenario budget that never runs, or hiding the ``delta`` limit from the
        ``full`` arm that is bound by it.  That is the same false-disclosure
        failure ``max_orders_per_step`` is rendered to avoid: the number the
        prompt prints and the number the resolver enforces have to be one value.

        **The limits are named as the refusal names them.**  ``SizeRefusal``
        reports ``binding`` as ``scenario``, ``nav_fraction``, ``name``,
        ``total``, ``delta`` or ``cash``, and ``_result_block`` passes that
        string through to the policy verbatim.  Describing them here under
        prettier names ("premium_limit") would leave the policy with a glossary
        that does not match the word it receives at the refusal, which is the
        one moment the disclosure has to pay off.

        This is a rule, not a measurement: every number below is a config field
        fixed before the window opens, so none of it is a constant fitted on the
        window being scored.  The *outcome* -- which limit bound, and at what
        quantity -- is still revealed only at the fill, in the ``E_LIMIT``
        detail, exactly as the cost magnitude is.
        """
        bounds = config.size
        active = SIZE_RULE_LIMITS[bounds.size_rule]
        # Described only if live.  The dict is keyed by the resolver's own limit
        # names so that adding a limit to ``SIZE_RULE_LIMITS`` without a
        # description raises here rather than shipping an undocumented bound.
        described = {
            "scenario": [
                f"  scenario: a one-day stress budget of {_pct(bounds.target_scenario_risk)}"
                " of nlv per package,",
                "    divided by that package's stress loss",
                "    risk = |delta|*s - 0.5*gamma*(S*s)^2 + |vega|*iv_pkg*"
                f"{bounds.vol_shock_relative:g} - theta,",
                "    all four greeks in dollars for the whole package and theta per",
                "    trading day - these are not the scaled dD you read in a P row.",
                f"    s = iv(name)/sqrt({bounds.trading_days_per_year:g}) is the"
                " one-sigma spot move, and iv_pkg is the",
                "    vega-weighted iv of your own legs, shocked"
                f" {_pct(bounds.vol_shock_relative)} relative.",
                "    the signs are the point: long gamma is a credit and short gamma a",
                "    charge, carry you receive is a credit and carry you pay a charge. a",
                "    convex package may be larger than a concave one on the same budget.",
                "    if the one-day loss is not positive this limit does not bind at all",
                "    and the others size you instead",
            ],
            "nav_fraction": [
                f"  nav_fraction: {_pct(bounds.nav_fraction)} of nlv divided by the"
                " package's gross premium,",
                "    both legs, unnetted. a credit structure is bounded by what it puts",
                "    at risk, not by the credit it collects, and on a cheap package this",
                "    is usually the limit that binds",
            ],
            "name": [
                f"  name: {_pct(bounds.max_risk_per_underlying)} of nlv of max loss"
                " already open on this name",
            ],
            "total": [
                f"  total: {_pct(bounds.max_total_open_risk)} of nlv of max loss open"
                " across the book",
            ],
            "delta": [
                f"  delta: the net dollar delta of the book is capped at"
                f" {_pct(bounds.max_net_dollar_delta)} of nlv.",
                "    only the part of a package that pushes the net further from zero is",
                "    charged, so an offsetting package has unlimited room by this measure",
            ],
            "cash": [
                "  cash: bp divided by (debit + collateral), where collateral is the",
                f"    uncovered part of max loss plus a {_pct(bounds.collateral_buffer)}"
                " buffer",
            ],
        }
        missing = [name for name in active if name not in described]
        if missing:  # pragma: no cover - guards a future limit, not a live path
            raise KeyError(f"size limits with no prompt text: {', '.join(missing)}")

        lines = [
            "sizing: you never choose a quantity. each O is sized to",
            f"  floor(min({', '.join(active)})) packages and is refused as E_LIMIT if",
            "  that floors to zero. the refusal names the limit that bound, so \"this",
            "  name is full\" and \"you are out of cash\" are different messages - read",
            "  it and send something else rather than the same order again.",
        ]
        for name in active:
            lines.extend(described[name])
        lines.extend(
            [
                "  refused before any of that: a package whose max loss is not finite",
                "    and positive, and the position counts under limits: below",
                "  so conviction is expressed in the choice of package, not in a size. a",
                "  tighter expression of the same view is sized larger, and spending the",
                "  per-name position budget is itself a cost - a third package in a name",
                "  closes that name.",
            ]
        )
        return lines

    def _hedging_lines(self, config: EnvConfig) -> list[str]:
        """What the share hedge does after the policy's orders fill.

        Emitted only when hedging is on.  An unhedged arm that read this section
        would be told about machinery that does not run, and would reason about
        a delta someone else was going to take off its hands.

        **The hedged set is rendered, not asserted.**  ``--hedge`` resolves to a
        family tuple and is a per-arm flag: ``volatility`` hedges five families,
        ``all`` hedges nine, and a comma list hedges whatever it names.  Saying
        "only the volatility families are hedged" in fixed text would be false
        on every arm but one -- and specifically it would tell an ``--hedge
        all`` run that its outright keeps its delta when the resolver is about
        to neutralise it.

        Note that this makes ``--hedge`` a prompt-visible flag for the first
        time.  ``scripts/psc/run_policy_episodes.sbatch`` relied on the opposite
        ("--hedge never enters the prompt") to justify comparing two hedge
        settings by replaying one recorded decision stream under each; that
        comparison is no longer valid for recordings made after this change,
        because the two settings now differ in the prompt the decisions were
        taken under. The comment there is updated to say so.
        """
        hedge = config.hedge
        if not (hedge.enabled and hedge.hedged_families):
            return []

        # Imported here, not at module scope, for the same reason the family
        # table imports ``contract``: the state space describes the resolvers
        # and must not pull the resolver package in at load time.
        from portfolio_monkey.env.resolvers.hedge import MIN_HEDGE_FRACTION

        hedged = "  ".join(
            f"{FAMILY_CODES[f]} {f}"
            for f in hedge.hedged_families
            if f in self._admitted
        )
        unhedged = [f for f in self._admitted if f not in hedge.hedged_families]
        lines = [
            "hedge: after your orders fill, on every decision, a share hedge runs.",
            f"  it covers {hedged}",
        ]
        if unhedged:
            lines.append(
                "  the rest keep their own delta: "
                + "  ".join(f"{FAMILY_CODES[f]} {f}" for f in unhedged)
            )
            lines.append(
                "  so directional exposure is yours to want and yours to close"
            )
        lines.append(
            "  every name you can trade is share-backed, so a package in that set"
        )
        lines.append("  always gets its hedge; it is never skipped for want of an instrument")
        if hedge.portfolio_level:
            lines.extend(
                [
                    "  per name, the option dollar deltas of all hedged positions are",
                    "  netted and the shares you already hold are added, giving one",
                    "  exposure. a second hedged package can offset the first and cost",
                    "  nothing extra to hedge",
                ]
            )
        else:
            lines.append(
                "  each hedged position is corrected on its own delta, so two offsetting"
            )
            lines.append("  packages are each hedged separately and each pays for it")
        # The band is one number under ``fixed`` and one per group under
        # ``whalley_wilmott``, where it is a function of the group's gamma and
        # the underlying's session half-spread.  Printing ``delta_band`` under
        # the second rule would name a field that rule does not read.
        if hedge.band_rule == "fixed":
            lines.append(
                f"  inside a band of {_pct(hedge.delta_band)} of nlv nothing trades."
                " outside it,"
            )
        else:
            lines.append(
                "  inside a per-name band set by that name's gamma and its share"
                " spread -"
            )
            lines.append(
                "  wider where delta runs away fastest and where correcting it costs"
            )
            lines.append("  most - nothing trades. outside it,")
        if hedge.hedge_to_band_edge:
            lines.append(
                "  shares trade back to the band edge, not to zero, keeping the sign."
            )
        else:
            lines.append("  shares trade back to flat.")
        lines.extend(
            [
                f"  an order correcting less than {_pct(MIN_HEDGE_FRACTION)} of the"
                " band is dropped",
                "  shares in a name that no longer has a hedged option position are",
                "  unwound to flat unconditionally",
                "  share fills cross the spread and pay commission. the hedge is not",
                "  free, it is charged to you, and closing an option position can itself",
                "  force a share trade.",
            ]
        )
        return lines

    def action_grammar(self, config: EnvConfig) -> str:
        """The order wire format, in the same positional style as the state.

        Size is absent on purpose: ``docs/env_contract.md`` section 4.5 moves
        it out of the action space entirely and into ``SizeResolver``, so the
        policy expresses conviction and never a contract count.

        **Tenor is required, and the leg table is printed.**  Both are answers
        to the same failure: the policy was being asked for a package while
        being told only the *names* of that package's coordinates.  It was
        never told that an iron condor is a short call, a long call further
        out, a short put and a long put further out -- that table lived in
        ``contract.py::_TOPOLOGY`` and never reached the prompt.  The lines are
        generated from that table rather than retyped, because a grammar that
        describes the resolver incorrectly costs a whole run of rejected orders
        to notice.

        **What "closest" means is stated three times**, once for each place the
        resolver approximates: the expiry is the listed one nearest the
        bucket's anchor, the strike is the listed one nearest the requested
        delta, and the request itself is snapped to a multiple of the delta
        step first.  The realised values come back on the ``R`` line, on the
        same principle as ``c<n>``: state the rule here, reveal the number at
        the fill.

        **``X`` needs four lines of prose where ``C`` needs none**, and that is
        not verbosity.  Every other verb's omitted fields resolve to a constant
        the grammar prints — the family defaults, right there in the table.  A
        roll's omitted fields resolve to *the replaced position's* coordinates,
        which no line of this block can print because they differ per position
        and per step.  So the rule has to be stated instead of shown, and it has
        to be stated in the one place the policy reads before writing an order.
        Unstated, ``X p03 31_90`` looks like a cheaper ``O`` and the policy gets
        back a package it did not describe — the failure being silent, because
        the order fills.
        """
        tenor_line = "  ".join(
            f"{b}({config.tenor.ranges[b][0]}-{config.tenor.ranges[b][1]}d"
            f"@{config.tenor.preferred_dte[b]})"
            for b in config.tenor.admitted
        )
        return "\n".join(
            [
                "O <t> <fam> <or> <tenor> <coords>  open a package",
                *_quote_grammar(config),
                "C <id>                             close an open position",
                "X <id> <tenor> <coords>            roll: close <id>, reopen the"
                " same view",
                "H                                  hold, emit nothing else",
                # Without these four lines the verb is unusable even though the
                # parser accepts it.  ``X`` is the only order whose *omitted*
                # fields are not the family defaults, so a policy that reads it
                # as shorthand for O will write ``X p03 31_90`` expecting a 55d
                # outright and get its own 35/10 vertical back at a new expiry.
                "  inherits the name, family and orientation of <id> -- X cannot"
                " change",
                "  those, only the tenor and the deltas. to change your view,"
                " C then O",
                "  omit <coords> and you keep the deltas <id> was opened at, not"
                " the",
                "  family defaults. this is the one blank in this grammar that is"
                " not a",
                "  default",
                "  the same bucket is allowed and is the usual roll: X p03 8_30 on"
                " a p03",
                "  now at 3 dte buys the bucket back from the start",
                "  one order, not two, but two half-spreads -- you pay to close and"
                " to",
                "  open, and the R line reports the sum. the replacement gets a new"
                " id",
                "  and carries src <id>g<n> in the P block from then on",
                # The id was never sourced, so the first live run answered
                # ``C 1`` and ``C 2`` against an empty book: the model was
                # numbering its own trades rather than reading the P block.
                # Saying where ids come from also says when there is nothing
                # to close, which is the more useful half.
                "id: copy it from the P block (p01, p02, ...). no P rows means"
                " nothing is open and neither C nor X is available",
                f"t: {' '.join(self._tradeable)}",
                f"tenor: {tenor_line}",
                "  required. (low-high@anchor) in days. every leg of the package",
                "  gets one expiry: the listed one closest to the anchor, inside",
                "  the bucket. the dte you actually got is on the R line and in",
                "  the P block. you may close at any later step -- the bucket is",
                "  the contract's life, not a holding period you commit to",
                "coords: deltas in percent, slash-separated, in the order below;"
                " omit any tail to take the default",
                "legs: the second line per family is what it builds. a leg is"
                " +/-<n><C|P>:",
                "  + long, - short, n the leg's delta in percent, C/P call/put,"
                " xN a ratio.",
                "  shown at the family defaults in the bullish form; your"
                " coordinates move",
                "  the deltas, and <or> r mirrors C/P for the directional"
                " families",
                # One line per family rather than a flat list of every bound.
                # The old grammar said coordinates go "in the family's declared
                # order" and then never declared it, so the model had to guess
                # both the arity and the meaning: the first live run answered
                # ``O NVDA ic n 30/30/10`` for a family that takes two.  The
                # schema *is* the thing being asked for, so it is stated.
                *self._family_lines(config),
                # In percent like everything else on these lines.  Printing the
                # raw 0.05 next to a default of "55" invites reading the default
                # as a delta and the step as a fraction of one.
                f"step: {config.coordinates.delta_step * 100:.0f}"
                " (coordinates snap to a multiple of this)",
                "nearest: you name a delta, you get the listed strike whose delta"
                " is",
                "  closest to it -- not that delta exactly. same for the expiry"
                " and its",
                "  anchor. what you actually got comes back on the R line as"
                " e<dte> and",
                "  d<deltas>, so compare them against what you asked for",
                "example: O NVDA cv b 8_30 35/10 = credit vertical, 8-30d,"
                " short 35d, 10d wide",
                "example: O AAPL ol b 31_90      = outright call, 31-90d,"
                " default 55d",
                # Deliberately the *omission* case rather than the full form:
                # the full form reads like an O with a different first token and
                # teaches nothing the O examples did not.  What has to be shown
                # is that the blank coordinates came from p07 and not from the
                # family table.
                "example: X p07 31_90            = if p07 is the NVDA cv above,"
                " that same",
                "         short 35d / 10d wide vertical again at 31-90d, new id",
                # The three ways an order can be refused for arithmetic rather
                # than for content.  Undisclosed, they are discoverable only by
                # being rejected, and an arm that varies one of them then
                # measures how fast a policy infers an unstated rule rather than
                # how it trades under a stated one.  Protocol 5 draws the line
                # at rules vs magnitudes: a cap is a rule and belongs here, the
                # cost of a fill is a magnitude and stays on the R line.
                #
                # Read off ``config`` rather than typed, because these are the
                # knobs the sweeps move -- a grammar that names 3 while the
                # resolver enforces 1 is worse than one that says nothing.
                *self._limit_lines(config),
                # Stated because it was once absent.  In the first live run the
                # model reasoned its way to a defensible trade and then narrated
                # it, ending in a fenced ``O MSFT cv b`` that the parser read as
                # prose and the ledger scored as an abstain.  The parser now
                # unwraps the fence, but a grammar that never says what a reply
                # looks like is measuring the prompt's omission, not the policy.
                "reply: order lines only, one per line, nothing else -- no prose,",
                "       no markdown, no code fences. a reply containing no order",
                "       line is an abstain, which is not the same as H",
            ]
        )

    def _limit_lines(self, config: EnvConfig) -> list[str]:
        """The counting rules, stated rather than discovered by rejection.

        Saying closes settle first is not a courtesy.  ``OptionsEnv._act`` runs
        every close before every open so that a rotation sizes against the
        buying power the close released, which means a replacement is legal in
        the same step *and in either order on the wire*.  A policy that assumed
        wire order was execution order would spend a step closing and the next
        step opening, paying an extra night of carry for nothing.

        **A close and its replacement spend two orders, and that sentence is
        shared rather than attached to the per-name cap.**  It was branch-local
        at first, stated only when the cap was 1, on the reasoning that that is
        where rotation is forced.  The reasoning was wrong twice over.  It is
        not cap-specific -- closes have always counted against the per-step
        budget -- and making it cap-specific confounds the sweep the branch
        exists to serve: the two arms then differ by the cap *and* by a warning.
        Unwarned, the P=10/3 arm spent a step writing seven closes and seven
        opens, and the cap took the first eight in wire order, so it liquidated
        seven positions and re-entered one.  Six ``E_TOO_MANY_ORDERS`` and a
        flat book, from a prompt that had told it the budget but not what a
        close costs against it.
        """
        size = config.size
        per_name = size.max_positions_per_underlying
        lines = [
            f"limits: at most {config.max_orders_per_step} orders per step,"
            f" {size.max_positions} positions open at once",
            f"  at most {per_name} position{'' if per_name == 1 else 's'} per name",
            "  closes settle before opens, so closing one position and opening its",
            "  replacement in the same step works, in either order on the wire --",
            "  but both count, so a rotation spends two of the step's orders per",
            "  name. orders past the per-step cap are refused in the order you",
            "  wrote them, and closes are orders too",
            # The budget asymmetry is the whole reason the verb pays for itself.
            # Left unstated, a policy that has read "a rotation spends two"
            # above will keep writing C+O for a move that X does in one slot,
            # and on a tight step that is the difference between rolling three
            # names and rolling one.
            "  X is one order for the pair, so if the only thing changing is the",
            "  expiry or the strikes, X does in one slot what C and O do in two",
        ]
        if per_name == 1:
            # The only genuinely cap-specific consequence: at 1 the rotation is
            # not one option among several, it is the sole way to change a view
            # on a name already held.
            lines.append(
                "  so the only way to change your view on a name you already hold"
            )
            lines.append("  is to close it and reopen")
        lines.append(
            "  an order over a limit is refused as E_LIMIT and costs nothing but"
            " the slot"
        )
        return lines

    def _family_lines(self, config: EnvConfig) -> list[str]:
        """One line per admitted family: code, coordinate names, defaults, bounds.

        Read off ``config.coordinates`` rather than written out, so a family
        whose schema changes cannot end up described one way in the prompt and
        validated another — the failure mode being a grammar that is confidently
        wrong, which costs a whole run of rejected orders to notice.

        The second line per family is the leg table, evaluated from
        ``contract.py::_TOPOLOGY`` against that family's *defaults*, in the
        bullish orientation.  It is an illustration of the shape, not of the
        order about to be sent: a policy that overrides a coordinate gets legs
        at its own deltas, and a bearish order gets the mirrored rights.  The
        alternative -- naming the coordinates and never saying what they build
        -- is what the zero-shot arm was asked to work with, and it produced
        ``O NVDA ic n 30/30/10`` for a two-coordinate family.
        """
        # Imported here rather than at module scope: the state space describes
        # the action space, but making it import the *resolver* at load time
        # would couple rendering to a package that pulls in the chain reader.
        from portfolio_monkey.env.resolvers.contract import (
            _TOPOLOGY,
            _evaluate,
            _right_for,
        )

        lines = []
        for family in self._admitted:
            names = list(config.coordinates.defaults.get(family, {}))
            if not names:
                lines.append(f"  {FAMILY_CODES[family]} {family}: no coordinates")
                continue
            schema = "/".join(names)
            defaults = "/".join(
                f"{config.coordinates.defaults[family][n] * 100:.0f}" for n in names
            )
            ranges = " ".join(
                f"{config.coordinates.bounds[n][0] * 100:.0f}-{config.coordinates.bounds[n][1] * 100:.0f}"
                for n in names
            )
            lines.append(
                f"  {FAMILY_CODES[family]} {schema}  default {defaults}  range {ranges}"
            )
            coordinates = config.coordinates.defaults[family]
            legs = []
            for _role, rule, expression, ratio in _TOPOLOGY[family]:
                delta = _evaluate(expression, coordinates)
                right = _right_for(rule, "bullish").upper()[0]
                sign = "+" if ratio > 0 else "-"
                count = "" if abs(ratio) == 1 else f"x{abs(ratio)}"
                legs.append(f"{sign}{delta * 100:.0f}{right}{count}")
            lines.append(f"    {' '.join(legs)}")
        return lines

    def episode_block(self, ctx: EpisodeContext) -> str:
        """Episode header, including the full restatement of a carried book.

        ``docs/env_contract.md`` section 1.4: under monthly episodes the book
        crosses the boundary but the context does not, so this is the only
        place the policy can be told what it already owns.  The carried
        positions are rendered with the same ``POS`` renderer used at every
        step, so the encoding the policy learns at ``t=0`` is the one it sees
        at ``t=1``.
        """
        lines = [
            f"EPI {ctx.episode_id} {ctx.start_date.isoformat()}..{ctx.end_date.isoformat()}"
            f" steps={ctx.decision_points}",
        ]
        # The ``spot`` line is **deleted** (2026-09-23), not suppressed behind
        # the anonymization flag, and the reason matters for how earlier runs
        # are read: it was fed from ``EpisodeContext.metadata["spot"]``, which
        # nothing in the package ever populated, so the branch never fired and
        # no observation has ever carried a price level.  The ruling that asked
        # to "drop spot and replace it with the returns" is therefore satisfied
        # on both halves already -- the ``M`` row's leading ``ret`` cell *is*
        # the session return per name, in 0.1% units, from
        # ``spot_return_step``.  Removing a branch that cannot fire changes no
        # arm's bytes; leaving it would have kept a price renderer sitting in
        # the one block that is byte-identical across every step.
        if ctx.carried_in is not None:
            lines.append(self._account_row(ctx.carried_in))
            if ctx.carried_in.positions:
                lines.append("carried in:")
                lines.append(self._position_rows(ctx.carried_in))
            else:
                lines.append("carried in: flat")
        return "\n".join(lines)

    # -- once per step ---------------------------------------------------

    def step_block(self, ctx: StepContext) -> Observation:
        market, missing = self._market_block(ctx)
        account = self._account_row(ctx.book)
        positions = self._position_rows(ctx.book)
        result = self._result_block(ctx.last_result)

        news = self._news_block(ctx)

        blocks: dict[str, str] = {"MKT": market, "ACCT": account}
        if positions:
            blocks["POS"] = positions
        if news:
            blocks["N"] = news
        if result:
            blocks["RES"] = result

        header = f"T {ctx.step_index} {ctx.trade_date.isoformat()} {ctx.session}"
        text = "\n".join([header, *blocks.values()])
        return Observation(
            text=text,
            blocks=blocks,
            missing=missing,
            token_estimate=estimate_tokens(text),
            provenance={
                "state_space_id": STATE_SPACE_ID,
                "decision_time": ctx.decision_time.isoformat(),
                "datasets": self.required_datasets(),
            },
        )

    def quote_block(self, results: Sequence[Mapping[str, Any]]) -> Observation:
        """The answer to a ``Q`` turn: the same ``R`` rows a fill would produce.

        Deliberately *only* the result block.  The market, account and position
        blocks were sent one turn ago and nothing since has moved them -- a
        quote opens nothing and spends nothing -- so re-sending them would pay
        for the whole state twice per decision and, worse, would make the
        second copy look like new information.  The policy is mid-decision; it
        needs the prices it asked for and nothing else.

        Rendered through ``_result_block``, the same renderer the fills use, so
        the preview and the execution are the same bytes in the same order.  A
        separate quote renderer would be a second wire format to keep in step,
        and the one thing this whole path exists to guarantee is that what was
        quoted is what gets charged.

        **One rendering, on both channels.**  A tool-calling policy answers the
        same bytes rather than a per-call ``toolResult``, which is possible
        because Bedrock accepts a ``toolUse`` turn replayed as plain assistant
        text followed by a plain user turn -- measured 2026-09-24.  That is what
        keeps the ledger honest: ``observation`` in ``decisions.jsonl`` is what
        the model actually read, on either channel, so the SFT corpus is not a
        re-rendering of a conversation that happened in JSON.
        """
        text = self._result_block({"results": results})
        return Observation(
            text=text,
            blocks={"RES": text} if text else {},
            token_estimate=estimate_tokens(text),
            provenance={"state_space_id": STATE_SPACE_ID, "turn": "quote"},
        )

    def quote_tool_schema(self, config: EnvConfig) -> Mapping[str, Any] | None:
        """``action_grammar``'s machine-readable half, for the ``Q`` verb only.

        ``None`` unless the arm asked for the tool channel, and that gate lives
        here rather than in the runner because ``action_grammar`` reads the same
        two fields: under ``"tool"`` the ``Q`` lines come out of the prompt and
        this schema goes on the wire, and the only way those two can never
        disagree is for one object to decide both.

        Every enum here is *derived*, never typed out.  The vocabulary already
        has exactly one declaration each -- ``FAMILY_CODES`` and
        ``ORIENTATION_CODES`` in ``actions.py``, ``COORDINATE_ORDER`` in
        ``strategy_templates.py``, the tenor buckets in the config -- and the
        cost of a second copy is recorded at ``COORDINATE_ORDER`` itself: *"it
        used to be a second copy, and a second copy is what let
        ``long_straddle`` end up with one prefix here and two there."*  A
        hand-written schema would be that second copy with a new failure mode
        attached, because the provider would validate the policy's call against
        a vocabulary ``parse_action`` then rejects -- the model would be told
        its call was well-formed and the environment would refuse it.

        **Shape is validated here, semantics stay in the parser.**  The enums
        are flat: any admitted family, any orientation, any admitted bucket.
        Which orientations a given family admits lives in
        ``TEMPLATE_SPECS[family].orientations`` and is expressed in the
        *description* rather than as a per-family ``oneOf``.  Two reasons, and
        neither is brevity for its own sake.  A ``oneOf`` over nine families
        multiplies the schema, and the schema is paid on every request of the
        episode.  More importantly the parser has to check it anyway --
        ``E_ORIENTATION`` fired once in job 46911489 and has to keep firing --
        so a ``oneOf`` would buy a second enforcement point for a rule that
        already has one, at the price of the two disagreeing.

        Returns the Bedrock ``toolSpec`` body.  The provider-specific envelope
        (``toolConfig`` for Converse, ``tools`` for the OpenAI shape) belongs to
        the policy that speaks that wire, not here.
        """
        if not (config.quotes_enabled and config.quote_channel == "tool"):
            return None

        from .actions import FAMILY_CODES, ORIENTATION_CODES

        codes = {full: code for code, full in FAMILY_CODES.items()}
        families = tuple(
            codes[f] for f in config.admitted_families if f in codes
        )
        # Per family: its code, the orientations it admits, and the positional
        # coordinate names in declaration order.  This is the leg table the
        # text grammar prints, in the one form a schema can carry.
        catalogue = "; ".join(
            f"{codes[f]}={f}"
            f"[{'/'.join(sorted(ORIENTATION_CODES[o] for o in _orientation_codes(f)))}]"
            f"({','.join(name for name, _ in coordinate_order(f))})"
            for f in config.admitted_families
            if f in codes
        )
        tenors = tuple(config.tenor.admitted)
        # The roster is an enum on a plain arm and a free string on an
        # anonymized one, and this is the same rule ``_anonymization_lines``
        # states for itself: *"it does not say which names are in the universe
        # ... a ten-name roster in the prompt is most of the way to
        # re-identification by elimination once the M rows are read."*  A
        # schema is prompt, so an enum of tickers here would leak the roster in
        # the one block that is byte-identical across every step of every
        # episode.  Enumerating the *labels* instead is not available either:
        # they are redrawn per episode and this schema is built once per run.
        underlying: dict[str, Any] = {
            "type": "string",
            "description": "which name to price, as it appears in the t column",
        }
        if not config.flags.anonymize:
            underlying["enum"] = list(config.universe.tradeable)
            underlying["description"] = "which name to price"
        return {
            "name": QUOTE_TOOL_NAME,
            "description": (
                "Price one option package at the current chain without opening "
                "it. Quoting moves no cash and opens no position. The answer "
                "comes back on the next turn as a line: R OK <the package> "
                "q<size> m<premium> c<cost> e<dte> d<per-leg deltas>. q is the "
                "size the resolver picked for you, and m and c are both quoted "
                "at that size. c is the entry half of the round trip, the same "
                "number the fill will charge. m is the mid you would pay or "
                "receive. The ratio c/m is the hurdle that package has to "
                f"clear. At most {config.max_quotes_per_name} calls per "
                "underlying per step. Families, the orientations each admits, "
                f"and its positional coordinates: {catalogue}."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "underlying": underlying,
                        "family": {
                            "type": "string",
                            "enum": list(families),
                            "description": "strategy family code",
                        },
                        "orientation": {
                            "type": "string",
                            "enum": list(ORIENTATION_CODES),
                            "description": (
                                "b bullish, r bearish, n neutral. Not every "
                                "family admits every orientation -- see the "
                                "catalogue in this tool's description"
                            ),
                        },
                        "tenor": {
                            "type": "string",
                            "enum": list(tenors),
                            "description": "days-to-expiry bucket",
                        },
                        "coordinates": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "delta coordinates in percent, positional, in "
                                "the order this family declares. Omit to take "
                                "the family defaults; a partial list fills from "
                                "the left. Snapped to the delta step"
                            ),
                        },
                    },
                    "required": ["underlying", "family", "orientation", "tenor"],
                }
            },
        }

    # -- blocks ----------------------------------------------------------

    def _news_block(self, ctx: StepContext) -> str:
        """One ``N`` row per name per session with news that broke since the last step.

        Absent rather than ``na`` when there is nothing, which is the one place
        this state space departs from the fixed-arity rule the ``M`` rows
        follow.  The rule exists because a positional encoding needs a stable
        column count; an ``N`` row has no columns to misalign, and a per-name
        ``N AAPL na na na na`` on the 61.7% of slots with no news would cost
        more tokens across an episode than every real row put together.  The
        ``system_block`` therefore says what an absent row means, because
        silence and "no news" are different claims and only the first is true.

        Ordered by the universe, not by arrival, so that two steps with the same
        set of names render identically and the ordering carries no accidental
        signal about which item the selector liked best.  Within a name the
        sessions run earliest-first, which is the one ordering that *is* meant
        to carry information.

        **The step backfills the sessions its grid skipped** (user ruling,
        2026-09-22: *"for PM session, include the AM session's news along with
        the PM news"*).  The builder is a changelog: it emits each item once, at
        the first session whose decision time it precedes.  That is addressed to
        a grid that visits every session, and the shipped grid does not -- it
        visits PM only -- so an item stamped AM was being dropped rather than
        deferred.  The stamp is a property of when the item published, so the
        channel going missing was the overnight one.

        Measured over the 245-date corpus: 1,250 rows are stamped AM against
        634 PM, so the PM step goes from 634 rows to 1,884 (2.97x) and from
        95.9% to 100% of steps carrying at least one row.  It is not free --
        the block goes from 103 to 306 estimated tokens per step, about +4.3k
        over a 21-step month -- and that cost is the reason the backfill is
        scoped to skipped sessions rather than applied unconditionally.

        This is backfill, not look-ahead, and the guard proves it rather than
        the docstring: the AM rows carry their own ``available_time``, ``fetch``
        asserts against the *PM* decision time, and an AM item is public before
        the open.  See ``StepContext.skipped_sessions`` for why a session the
        grid *does* visit is not re-served.

        **A date with no news partition renders ``NEWS_UNAVAILABLE`` and keeps
        going** rather than raising.  One date in 246 has no partition
        (``2025-03-28``), and the alternatives were to backfill it or to drop it
        from the window; both change the window, and the window is in the paper.
        Tolerating it costs one row of tokens on one step.  Tolerating it
        *silently* would cost a false claim, which is why the row exists.
        """
        # The ablation arm returns before touching ``ctx.fetch``, so it does not
        # merely hide the rows -- it never reads the dataset.  That matters for
        # the leakage question the ablation is meant to answer: a run that
        # fetched the news and dropped it would still have had the corpus in
        # scope at decision time, and nothing in the artifact would say which.
        if not self._textual:
            return ""
        sessions = (*ctx.skipped_sessions, ctx.session)
        per_session: list[tuple[str, Mapping[str, Any]]] = []
        absent = False
        for s in sessions:
            try:
                per_session.append(
                    (s, ctx.fetch(DATASET_NEWS, self._universe) if s == ctx.session
                        else ctx.fetch_session(DATASET_NEWS, self._universe, session=s))
                )
            except DatasetError:
                # Tolerated here and nowhere else (user ruling, 2026-09-24:
                # *"tolerate-absent-news in code"*), for the reason given at
                # ``NEWS_UNAVAILABLE``.  Caught per session rather than around
                # the loop because the partition is per *date*: if it is gone it
                # is gone for every session, and a loop that stopped at the
                # first failure would silently serve the sessions it had already
                # read as though they were the whole changelog.
                absent = True
        rows = [NEWS_UNAVAILABLE] if absent else []
        for ticker in self._universe:
            for _session, records in per_session:
                record = records.get(ticker)
                if record is None:
                    continue
                clause = str(record.get("clause") or "").strip()
                if not clause:
                    continue
                rows.append(
                    f"N {ticker} {record.get('catalyst', 'ot')} "
                    f"{record.get('direction', '?')} {record.get('horizon', '?')} "
                    f"{clause}"
                )
        return "\n".join(rows)

    def _market_block(self, ctx: StepContext) -> tuple[str, tuple[str, ...]]:
        keys: Sequence[str] = self._universe
        underlying = ctx.fetch(DATASET_UNDERLYING, keys)
        surface = ctx.fetch(DATASET_SURFACE, keys)
        flow = ctx.fetch(DATASET_FLOW, keys)
        open_interest = ctx.fetch(DATASET_OI, keys)
        # Only the AM row needs it, and the partition is cached, so the PM step
        # does not pay for a read it will not use.
        previous_close = (
            ctx.fetch_previous_close(DATASET_SURFACE, keys)
            if ctx.session == "AM"
            else {}
        )

        rows: list[tuple[str, ...]] = []
        missing: list[str] = []
        for ticker in self._universe:
            fields = self._market_fields(
                ticker,
                session=ctx.session,
                underlying=underlying,
                surface=surface.get(ticker),
                flow=flow.get(ticker),
                open_interest=open_interest.get(ticker),
                previous_close=previous_close.get(ticker),
            )
            # The leading ``M`` costs one token per name per step and buys the
            # only cue that says which schema line to read the row against.
            # Without it the account row was the sole self-labelling row in the
            # observation, so a reader who learned "first token is the block
            # letter" from ``A`` applied it to ``AAPL`` and decoded every market
            # row one column out of phase.
            rows.append(("M", fields.ticker, *fields.cells))
            missing.extend(fields.missing)
        return render_rows(rows), tuple(missing)

    def _market_fields(
        self,
        ticker: str,
        *,
        session: str,
        underlying: Mapping[str, Any],
        surface: Any,
        flow: Any,
        open_interest: Any,
        previous_close: Any = None,
    ) -> _MarketFields:
        missing: list[str] = []

        ret_raw, rv_raw = self._equity_fields(ticker, underlying)
        iv_raw = _value(surface, "atm_iv_30d")
        ts_raw = _value(surface, "term_slope")
        sk_raw = _value(surface, "skew_25d")
        bf_raw = _value(surface, "butterfly_25d")

        # ``dv`` is the change since the previous mark, which is the same day's
        # open at PM and the previous day's close at AM (section 4.1).  The two
        # span different amounts of clock time — six hours against seventeen,
        # or sixty-five over a weekend — but the row declares its own session,
        # and the overnight move is the one that carries earnings and news.
        dv_raw = (
            _value(surface, "surface_change_step.atm_iv_30d")
            if session == "PM"
            else _overnight_change(iv_raw, _value(previous_close, "atm_iv_30d"))
        )

        # R5: the wedge is precomputed so the policy never subtracts in context.
        w_raw = None if (iv_raw is None or rv_raw is None) else iv_raw - rv_raw

        fi_raw = self._flow_imbalance(flow)
        doi_raw = self._oi_change(open_interest)

        cells = (
            scaled(ret_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(rv_raw, SCALE_DP, missing=MISSING),
            scaled(iv_raw, SCALE_DP, missing=MISSING),
            scaled(dv_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(w_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(ts_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(sk_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(bf_raw, SCALE_DP, signed=True, missing=MISSING),
            scaled(fi_raw, SCALE_PCT, signed=True, missing=MISSING),
            scaled(doi_raw, SCALE_PCT, signed=True, missing=MISSING),
        )
        for name, cell in zip(
            ("ret", "rv", "iv", "dv", "w", "ts", "sk", "bf", "fi", "doi"), cells, strict=True
        ):
            if cell == MISSING:
                missing.append(f"{ticker}.{name}")
        return _MarketFields(ticker, cells, tuple(missing))

    def _equity_fields(
        self, ticker: str, underlying: Mapping[str, Any]
    ) -> tuple[float | None, float | None]:
        """``ret`` and ``rv``, read only from the name's own row.

        There used to be a fallback here, and section 7 item 1 was right to want
        it: SPX held the tenth slot and had no row in
        ``underlying_market_features`` at all — it is an index, there is no
        security to pull — so it borrowed the ``market_return_step`` /
        ``market_realized_vol_21d`` columns that every row carries.  For SPX that
        substitution was not an approximation of the value, it *was* the value.

        SPY took the slot and has a row of its own, so no name is structurally
        absent any more and the fallback lost its only legitimate trigger.  What
        it kept was an illegitimate one: a name whose partition simply failed to
        land would borrow the market columns off whichever *other* name did land,
        and emit the index's return and the index's realized vol under its own
        ticker.  ``w = iv - rv`` is then computed from one name's implied vol
        against the index's realized vol, which does not merely blur the row — it
        manufactures a vol-premium edge out of a missing file, and manufactures it
        largest exactly when the name is most unlike the index.

        So the gap is reported as a gap.  ``ret`` and ``rv`` go to ``na``, ``w``
        inherits that (section 0 C4), and the field names land in
        ``observation.missing`` where a consumer can see them.  Note the shape of
        what was fixed: the old path was safe when the *whole date* was missing
        (nothing to borrow from) and unsafe when a *single name* was, which is
        the opposite of the order in which those two failures should be trusted.

        ``⟨Q7b⟩`` is closed by this — it asked whether to source SPX's equity
        fields from SPY's ``market_*`` or ingest the index level directly, and
        the answer turned out to be neither.
        """
        record = underlying.get(ticker)
        if record is None:
            return None, None
        return _value(record, "spot_return_step"), _value(record, "realized_vol_21d")

    @staticmethod
    def _flow_imbalance(flow: Any) -> float | None:
        """``fi``, gated on ``trade_count`` — section 0 rule C1.

        The earlier guard tested ``contract_count > 0``, which section 0 calls
        **inverted**: it admits the least informative cells and marks the rest
        missing.  ``fi`` is a ratio of signed to gross delta traded, so a window
        holding one print forces it to ±1.0.  Measured over the window, the
        ``contract_count`` gate admitted 140 AM cells of which **58 (41%) were
        saturated at |fi| ≥ 0.999**; nine single names therefore read as
        maximum directional conviction off one lot each, under the same field
        name and scale as real flow.

        The threshold is not delicate.  The trade-count distribution is
        bimodal — SPX prints in the thousands at the open, the nine singles in
        single digits — so every value from 10 to 1,000 admits *exactly* the
        same cells: all 378 PM cells (zero saturated at any threshold, median
        110k trades) and SPX alone at AM.  25 is chosen inside that plateau
        rather than at its edge.  What the gate costs is therefore nothing at
        PM and the whole AM single-name column, which is the point: section 4
        already concludes ``fi`` is a PM-only field for single names.
        """
        if flow is None:
            return None
        trades = _value(flow, "trade_count")
        if trades is None or trades < MIN_FLOW_TRADES:
            return None
        return _value(flow, "delta_weighted_flow_imbalance")

    @staticmethod
    def _oi_change(open_interest: Any) -> float | None:
        if open_interest is None:
            return None
        prior = _value(open_interest, "prior_option_open_interest")
        change = _value(open_interest, "open_interest_change")
        if prior is None or change is None or prior == 0:
            return None
        return change / prior

    def _account_row(self, book: BookView) -> str:
        """``A nlv cash bp util rlz unrlz dd npos nD nV`` — 37 tokens."""
        nav = book.nav
        cells = (
            scaled(nav, SCALE_KDOLLAR, missing=MISSING),
            scaled(book.cash, SCALE_KDOLLAR, missing=MISSING),
            scaled(book.buying_power, SCALE_KDOLLAR, missing=MISSING),
            scaled(book.collateral_utilization, SCALE_PCT, missing=MISSING),
            scaled(book.realized_pnl, SCALE_HDOLLAR, signed=True, missing=MISSING),
            scaled(book.unrealized_pnl, SCALE_HDOLLAR, signed=True, missing=MISSING),
            scaled(book.drawdown, SCALE_PCT, signed=True, missing=MISSING),
            str(book.n_positions),
            _pct_of_nav(book.net_dollar_delta, nav),
            _pct_of_nav(book.net_dollar_vega, nav),
        )
        return "A " + " ".join(cells)

    def _position_rows(self, book: BookView) -> str:
        """``P id t fam or dte qty mark upnl dD [src]``, one line per package.

        Section 6.3's amendment to ``docs/env_contract.md`` section 3: only
        ``dollar_delta`` is serialized per position — it is what says which
        position to close — while gamma, vega and theta appear once on the
        account line.  Measured saving 9 tok/position, ~4,500 tokens/episode at
        P=12.  ``MarketResolver`` still computes all four; this is about what
        reaches the wire.  Recorded as ``⟨Q7d⟩``.

        **``src`` is the one optional cell, and it is last and self-describing
        for that reason.**  Every other field is positional, so an optional cell
        in the middle would shift the meaning of everything after it on rolled
        rows only — the worst kind of encoding bug, because the common row still
        reads correctly.  Put last and marked with a leading ``<``, a reader that
        ignores it loses only the lineage, and a reader that counts cells is not
        misled about ``dD``.
        """
        rows = [self._position_cells(p, book.nav) for p in book.positions]
        return render_rows(rows)

    def _position_cells(self, position: PositionView, nav: float) -> tuple[str, ...]:
        cells = (
            "P",
            position.position_id,
            position.underlying,
            FAMILY_CODES.get(position.family, position.family),
            ORIENTATION_CODES.get(position.orientation, position.orientation),
            str(position.dte),
            str(position.quantity),
            scaled(position.mark, SCALE_PCT, missing=MISSING),
            scaled(position.unrealized_pnl, 1.0, signed=True, missing=MISSING),
            _pct_of_nav(position.dollar_delta, nav),
        )
        if not position.rolled_from:
            return cells
        # The generation is carried here and not only in the id because the
        # chain cannot be walked from the prompt: p03's own ``src`` named p01,
        # but p03 is closed and gone from the P block, so a policy reading only
        # ``<p03`` sees a first roll where there have been four.
        return (*cells, f"<{position.rolled_from}g{position.roll_generation}")

    @staticmethod
    def _result_block(result: Mapping[str, Any] | None) -> str:
        """What the resolvers did with the previous action.

        Rejections are the highest-value tokens in the whole context: a policy
        that cannot see why an order was refused will re-emit it every step for
        the rest of the episode.  One line per order, verdict first.

        The key is ``results`` because that is what :class:`OptionEnv` writes.
        It was read as ``orders`` here, and the mismatch was silent — the loop
        just found nothing, ``RES`` was never emitted, and the first live run
        spent an entire episode believing it held positions the chain had
        refused to open.  ``.get`` on the wrong key cannot fail loudly, so the
        test below asserts the verdict reaches the rendered text.

        **``c<dollars>`` is what the fill cost.**  It used to be absent, and a
        filled order reported no price and no friction at all -- the account
        line moved and nothing said why.  A live trace shows the model working
        this out and giving up: "rlz -5.  Where did -5 come from?  Maybe from
        transaction costs?  ...  Not specified."  It never found out, and over
        492 decisions it traded 7.6x its capital at a cost it could not see.
        One integer per fill closes that loop; the family table in
        ``system_block`` is the ex-ante half of the same disclosure.
        """
        if not result:
            return ""
        lines: list[str] = []
        for entry in result.get("results", ()):
            verdict = entry.get("status", "?")
            detail = entry.get("detail") or entry.get("error") or ""
            token = entry.get("order", "")
            position_id = entry.get("position_id")
            cost = entry.get("cost")
            parts = ["R", verdict, str(token)]
            # A roll is two fills reported as one row, so the id cell has to
            # carry the pair.  ``p07<p03`` is the new package and the one it
            # replaced; a bare ``<p03`` is the case that matters most -- the
            # close went through and the replacement was refused, so the book
            # is now flat on that name.  Without the second form a policy reads
            # only the rejection code and keeps reasoning about a position it
            # no longer holds.  Same ``<`` as the P block's ``src``.
            closed = entry.get("closed")
            if closed:
                parts.append(f"{position_id or ''}<{closed}")
            elif position_id:
                parts.append(str(position_id))
            # ``q`` is the size the resolver picked.  The grammar has no
            # quantity field, so this is the only place the policy learns how
            # many packages its line bought -- and without it ``p`` and ``c``
            # are unreadable, because both are quoted at that size and neither
            # is per-package.
            quantity = entry.get("quantity")
            if quantity is not None:
                parts.append(f"q{int(quantity):d}")
            # ``m`` is the mid, ``c`` the friction on top.  Cost alone states a
            # level; the pair states a *ratio*, and the ratio is the decision.
            # Measured round-trip cost is 16.6-18.3% of premium overall but
            # spans an order of magnitude by family (outright 0.8-2.3%,
            # iron_condor 13.9-16.5%), so a policy reading ``c`` without the
            # premium cannot tell a cheap structure from an expensive one.
            #
            # ``m`` and not ``p``, which is what this was first written as: the
            # id cell is already ``p07``, so ``p612`` on the same line would be
            # two meanings for one prefix, distinguishable only by position.
            # The policy has to parse these rows every step, and a prefix
            # alphabet that needs positional disambiguation is one it will get
            # wrong -- ``p07 q4 p612`` invites reading a premium as a position.
            premium = entry.get("premium")
            if premium:
                parts.append(f"m{round(float(premium)):d}")
            # Rounded to the dollar.  Fills ran $18-$2,400 on the measured run,
            # so the cent precision the ledger keeps would spend tokens on
            # digits no decision turns on.
            if cost:
                parts.append(f"c{round(float(cost)):d}")
            # ``e<dte>`` and ``d<deltas>`` are the realised half of the
            # "nearest" rule the grammar states.  The policy names a tenor
            # bucket and a set of deltas and gets the closest *listed* expiry
            # and strikes, which can be several points away on a sparse ladder
            # -- a 31_90 request anchored at 45 can fill at 38 or 52.  Without
            # the echo the approximation is invisible, so the policy cannot
            # learn which buckets resolve tightly for which names, and a
            # post-hoc reader cannot tell a deliberate 38-DTE package from an
            # anchor that missed.  Same contract as ``c<n>``: the rule is in
            # the prompt, the number arrives with the fill.
            dte = entry.get("dte")
            if dte is not None:
                parts.append(f"e{int(dte):d}")
            deltas = entry.get("leg_deltas")
            if deltas:
                parts.append(
                    "d" + "/".join(f"{abs(float(d)) * 100:.0f}" for d in deltas)
                )
            if detail:
                parts.append(str(detail))
            lines.append(" ".join(p for p in parts if p))
        for note in result.get("notes", ()):
            lines.append(f"R note {note}")
        return "\n".join(lines)


def _value(record: Any, name: str) -> float | None:
    """Read one field, treating NaN as absent.

    Several surface columns carry NaN rather than null for an unfitted value.
    NaN survives arithmetic silently and compares False against everything, so
    it has to be normalized to ``None`` at the boundary rather than downstream.
    """
    if record is None:
        return None
    value = record.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value


def _overnight_change(opened: float | None, previous_close: float | None) -> float | None:
    """``dv`` for an AM row: the move from the previous close to this open.

    ``None`` when either end is absent, which is the first date in coverage,
    any date following a gap, and any name whose surface did not fit at one of
    the two marks.  Deliberately not zero: a flat overnight session and an
    unfitted one must not render as the same bytes.
    """
    if opened is None or previous_close is None:
        return None
    return opened - previous_close


def _pct_of_nav(value: float | None, nav: float) -> str:
    if value is None or nav == 0:
        return MISSING
    return scaled(value / nav, SCALE_PCT, signed=True, missing=MISSING)


register_state_space(STATE_SPACE_ID, StateSpaceV1)
