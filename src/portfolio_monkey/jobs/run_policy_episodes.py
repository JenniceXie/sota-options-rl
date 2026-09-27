"""Run one arm of the evaluation over the window and write its ledger.

    python -m portfolio_monkey.jobs.run_policy_episodes \
        --arm deepseek --policy deepseek \
        --start 2025-03-01 --end 2025-08-29 \
        --risk-free-rate 0.043 \
        --out runs/deepseek

One invocation is one arm.  The arms of ``docs/evaluation_protocol.md`` section
6 differ only in ``--policy``, and running them from one entry point is what
makes that claim checkable: the config, the data, the resolvers, the cost model
and the ledger writer are the same objects in every case.

Two guards run before the first API call, because both failures are cheap to
detect now and expensive to discover on step 40 of 42:

* **Coverage.**  The trading dates come from the chain's own partitions, not
  from a calendar, and the datasets the state space declares are checked for the
  same dates.  A run that would step onto a date with no chain stops here.
* **The risk-free rate.**  Section 5 requires cash to accrue ``r_f``.  With none
  supplied the book carries a 4-5% annual headwind against every index it is
  compared to, so the job refuses rather than quietly producing a number that
  looks like underperformance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from portfolio_monkey.env.book import BookState
from portfolio_monkey.env.chain import OptionChain
from portfolio_monkey.env.datasets import DatasetFeatureSource, data_root
from portfolio_monkey.env.environment import OptionsEnv, build_grid, monthly_episodes
from portfolio_monkey.env.ledger import LedgerWriter, reconcile
from portfolio_monkey.env.markquotes import (
    MARK_QUOTE_SOURCE_VERSION,
    MassiveMarkQuotes,
)
from portfolio_monkey.env.policy import HoldPolicy, Policy
from portfolio_monkey.env.policy.baselines import (
    EconomicRulePolicy,
    GarchVolPremiumPolicy,
    load_params,
)
from portfolio_monkey.env.policy.replay import EXACT, TOLERANCES, ReplayPolicy, STATE_TOLERANT
from portfolio_monkey.env.runner import run_arm
from portfolio_monkey.env.spec import (
    BAND_RULES,
    EnvConfig,
    FeatureFlags,
    GridSpec,
    HedgeSpec,
    SIZE_RULE_LIMITS,
    SIZE_RULES,
    SizeBounds,
    VOLATILITY_FAMILIES,
)
from portfolio_monkey.env.spreads import SpreadTable, SpreadTableError
from portfolio_monkey.env.tokens import HEURISTIC_SPEC, build_token_counter
from portfolio_monkey.eval.builders.from_ledger import decision_quality

POLICIES = (
    "hold", "deepseek", "bedrock", "replay", "garch_vrp", "econ_rules",
    # Band D.  Scores every (name, head, tenor) with a frozen per-head
    # classifier and opens the best few the caps allow.  Unlike the two rule
    # baselines it looks no ticker up -- a package is scored from the market row
    # in front of it -- so it is safe under ``--anonymize`` and is not refused.
    "package_model",
)

#: Where the two conventional baselines read their frozen parameters from.  A
#: default *path* rather than a default parameter set: the thresholds live on
#: disk beside a provenance block, so "were these fitted on the test window?" is
#: answered by reading one file instead of by trusting a docstring.
DEFAULT_BASELINE_PARAMS = "configs/baselines/params.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, help="label for this run in the ledger")
    parser.add_argument("--policy", default="hold", choices=POLICIES)
    parser.add_argument(
        "--replay-from",
        default=None,
        type=Path,
        help=(
            "decisions.jsonl to re-emit, for --policy replay. Varying a config "
            "flag across two live runs measures the flag plus the sampler; "
            "replaying one recording under each measures the flag."
        ),
    )
    parser.add_argument(
        "--package-models",
        default=None,
        help=(
            "directory of frozen per-head classifiers for --policy "
            "package_model, written by scripts/analysis/fit_package_models.py. "
            "Its meta.json records the fit window, the validation window the "
            "abstain threshold was chosen on, and that the test window was not "
            "touched; nothing is fitted at run time."
        ),
    )
    parser.add_argument(
        "--baseline-params",
        default=DEFAULT_BASELINE_PARAMS,
        help=(
            "frozen parameters for --policy garch_vrp / econ_rules: GARCH "
            "coefficients, the per-date volatility forecast and the signal "
            "thresholds. Built by scripts/analysis/fit_baseline_params.py on "
            "train+validation dates only; the artifact carries its fit window "
            "and is refused if it cannot state one."
        ),
    )
    parser.add_argument(
        "--replay-tolerance",
        default=EXACT,
        choices=TOLERANCES,
        help=(
            f"how a replayed observation is compared to the recording. "
            f"{EXACT!r} (default) refuses any difference, so a completed replay "
            f"is itself proof the environment reproduced the recorded run. "
            f"{STATE_TOLERANT!r} keeps that check on the header, market and news "
            "blocks and permits the account, position and result blocks to move "
            "-- which is what an arm that changes a fill (sizing, hedging) "
            "requires, and is the only way to run those arms without re-"
            "generating trajectories on a GPU. It is recorded in the manifest "
            "and dates its first divergence; it is never selected automatically."
        ),
    )
    parser.add_argument("--track", default="full", help="cost/state track label")
    parser.add_argument("--out", required=True, type=Path, help="ledger directory")
    parser.add_argument("--start", required=True, type=_as_date)
    parser.add_argument("--end", required=True, type=_as_date)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--feature-cache",
        default=None,
        help=(
            "where to keep universe-filtered feature extracts "
            "(default <data-root>/.feature_extracts; 'none' to read the full "
            "partitions every step)"
        ),
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=None,
        help="annualized, decimal (0.043 for 4.3%%). Required unless --no-rf is given.",
    )
    parser.add_argument(
        "--no-rf",
        action="store_true",
        help="run with zero cash accrual, accepting the 4-5%% headwind of protocol 5",
    )
    parser.add_argument(
        "--decision-sessions",
        default="PM",
        help=(
            "comma-separated sessions the policy may act on: 'PM' (default) or "
            "'AM,PM'. AM orders fill against that date's close, because the AM "
            "chain is a copy of the previous close while AM features already "
            "carry today's open; see GridSpec.am_fills_at_close."
        ),
    )
    parser.add_argument(
        "--hedge",
        default="none",
        help=(
            "which strategy families are delta-hedged: 'none' (default), "
            "'volatility' for the families whose thesis is vol rather than "
            "direction (see spec.VOLATILITY_FAMILIES), 'all', or an explicit "
            "comma-separated list of family names"
        ),
    )
    parser.add_argument(
        "--delta-band",
        type=float,
        # ``HedgeSpec()`` and not ``HedgeSpec.delta_band``: the dataclass is
        # ``slots=True``, so the class attribute is a ``member_descriptor``
        # rather than the default value.  ``type=float`` is not applied to a
        # default, so the descriptor reached ``HedgeSpec(delta_band=...)``
        # intact and ``band = spec.delta_band * nav`` raised ``TypeError`` --
        # which happens before the ``enabled`` check, so it fired even with
        # hedging off.  Every run so far passed ``--delta-band`` from the
        # sbatch, which is the only reason this never surfaced.
        default=HedgeSpec().delta_band,
        help="hedge when |net dollar delta| exceeds this fraction of NLV",
    )
    parser.add_argument(
        "--band-rule",
        choices=BAND_RULES,
        default=HedgeSpec().band_rule,
        help=(
            "how the no-trade band is set. 'fixed' (default) is --delta-band * "
            "NLV for every group. 'whalley_wilmott' sets it per group and per "
            "session from the measured half-spread and the group's gamma: "
            "H = (3/2 * k * S * G^2 / gamma)^(1/3) shares, so it widens where "
            "correcting is expensive and where the delta moves fastest"
        ),
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=HedgeSpec().risk_aversion,
        help=(
            "relative risk aversion in the Whalley-Wilmott band (gamma_abs = "
            "this / NLV, so the band is scale-free in the book). Larger means "
            "less tolerance for delta and a narrower band, as the inverse cube "
            "root: 8x the aversion halves the band"
        ),
    )
    parser.add_argument(
        "--band-multiple",
        type=float,
        default=HedgeSpec().band_multiple,
        help=(
            "linear scale on the Whalley-Wilmott band. Not a cube root: this "
            "corrects for the decision grid being twice-daily rather than "
            "continuous, which the closed form assumes, so it is the knob to "
            "sweep for grid coarseness rather than for risk appetite"
        ),
    )
    parser.add_argument(
        "--min-band-fraction",
        type=float,
        default=HedgeSpec().min_band_fraction,
        help=(
            "floor on the band as a fraction of NLV. A decayed position has "
            "zero gamma and therefore a zero-width band, which re-hedges every "
            "dollar of residual delta at every grid point forever"
        ),
    )
    parser.add_argument(
        "--max-band-fraction",
        type=float,
        default=HedgeSpec().max_band_fraction,
        help=(
            "ceiling on the band as a fraction of NLV, so a high-gamma group "
            "cannot use the band rule to opt out of the delta limit entirely"
        ),
    )
    parser.add_argument(
        "--spread-table",
        type=Path,
        default=None,
        help=(
            "CSV of per-(date, ticker) AM/PM half-spreads k = 0.5*(ask-bid)/mid "
            "(columns date,ticker,k_am,k_pm), from "
            "scripts/analysis/build_underlying_spreads.py. Unset, the band rule "
            "uses a flat --fallback-half-spread for every name, which is the "
            "control that isolates the band's gamma scaling from its spread "
            "scaling. Missing rows fall back rather than fail, and the manifest "
            "records the fallback count per name"
        ),
    )
    parser.add_argument(
        "--fallback-half-spread",
        type=float,
        default=HedgeSpec().fallback_half_spread,
        help=(
            "relative half-spread used where --spread-table has no row (default "
            "1e-4 = 1bp). Also the flat value when no table is given"
        ),
    )
    parser.add_argument(
        "--measured-spread-costs",
        action="store_true",
        help=(
            "charge the hedge's share fills the measured half-spread rather "
            "than the flat cost.stock_half_spread_bps. Independent of "
            "--band-rule on purpose: an arm that both widens the band and "
            "reprices the fill cannot say which half moved the PnL"
        ),
    )
    parser.add_argument(
        "--no-textual-context",
        action="store_true",
        help=(
            "the ablation arm: no N rows and no cat/dir/hzn/clause legend, so "
            "the policy sees only the numeric state. Everything else -- "
            "universe, grid, sizing, hedging, costs -- is unchanged. Also drops "
            "news_catalyst_rows from the required datasets, so the arm does not "
            "fetch a corpus it never renders"
        ),
    )
    parser.add_argument(
        "--suppress-future-knowledge",
        action="store_true",
        help=(
            "suppress at source: tell the policy in the system block to decide "
            "only from the step and not from what it remembers about these "
            "dates. This is the remedy arm for teacher-trace leakage. It is an "
            "instruction, not a capability bound -- a model that knows how the "
            "window ended still knows it -- so judge it on the audited leak "
            "rate and on whether the action moved, never as a guarantee"
        ),
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help=(
            "de-identify the observation: tickers become U01..U10, redrawn "
            "every episode, and the date becomes m<n>, calendar days to the "
            "standard third-Friday monthly expiry. Non-universe proper nouns "
            "in a news clause become <ent>. The book, the ledger and the "
            "resolvers keep real tickers and real dates, and the order is "
            "unmasked before it is parsed -- so the resolver still finds a real "
            "expiry from a tenor bucket the policy named without knowing the "
            "date. This is the leakage remedy that does not depend on the model "
            "cooperating, but its guard counts proper nouns only: sector "
            "language survives and still dates a step"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model id. OpenRouter id for --policy deepseek; Bedrock inference "
            "profile for --policy bedrock (default us.openai.gpt-6-astra -- the "
            "bare id without the 'us.' prefix is refused as on-demand "
            "unsupported)"
        ),
    )
    parser.add_argument(
        "--bedrock-region",
        default=None,
        help=(
            "region to call for --policy bedrock (default us-west-2). The 'us.' "
            "profile may still serve from us-east-1 or us-east-2, so the arm is "
            "recorded as region-unpinned whatever this is set to"
        ),
    )
    parser.add_argument(
        "--provider",
        default="",
        help=(
            "comma-separated OpenRouter provider names, most preferred first "
            "(e.g. 'SiliconFlow,GMICloud,Novita'). Unset, OpenRouter picks a "
            "host per request, and the hosts behind one model id differ in "
            "quantization and in decode rate, so the arm becomes a mixture "
            "over hosts in unrecorded proportions"
        ),
    )
    parser.add_argument(
        "--no-provider-fallbacks",
        action="store_true",
        help=(
            "fail the step rather than let OpenRouter route outside --provider. "
            "Buys exact host provenance at the cost of turning one host's "
            "outage into held steps"
        ),
    )
    # Both caps reach the prompt as well as the resolver, so a sweep over them
    # varies a rule the policy was told rather than one it has to infer from
    # being refused.  Defaults restate ``SizeBounds`` so an unswept run is
    # byte-identical to one that never passed the flag.
    parser.add_argument(
        "--max-positions",
        type=int,
        default=SizeBounds().max_positions,
        help="how many positions may be open at once (P). Bounds the POS block",
    )
    parser.add_argument(
        "--max-positions-per-underlying",
        type=int,
        default=SizeBounds().max_positions_per_underlying,
        help=(
            "how many positions may be open on one name. At 1, changing a view "
            "means closing and reopening, which costs two of the per-step order "
            "budget rather than one"
        ),
    )
    # The sizing rule and its three scale knobs.  Grouped here because a sweep
    # over them is the point: ``--size-rule`` chooses *which* limits bind and
    # the other three choose *where*, and an arm is only interpretable if the
    # manifest carries all four.
    parser.add_argument(
        "--size-rule",
        choices=SIZE_RULES,
        default=SizeBounds().size_rule,
        help=(
            "which limits the size is a minimum over. 'nav_fraction' is the "
            "default and the toggle-OFF arm: gross premium capped at a fraction "
            "of NAV, reading no greek. 'scenario' is the toggle-ON arm, the same "
            "rule plus the one-day Taylor-expansion risk budget, so the two "
            "differ by exactly one limit. 'full' is a diagnostic, not an arm. "
            "'max_loss' and 'scenario_max_loss' were removed 2026-09-22 with "
            "the ceiling they were built on and will be rejected here"
        ),
    )
    parser.add_argument(
        "--vol-shock",
        type=float,
        default=SizeBounds().vol_shock_relative,
        help=(
            "relative shock to the package's own IV in the scenario, charged as "
            "adverse in both spot directions. The only term that reaches vega, so "
            "it is what prices a long straddle against a debit vertical"
        ),
    )
    parser.add_argument(
        "--target-scenario-risk",
        type=float,
        default=SizeBounds().target_scenario_risk,
        help=(
            "budgeted cost of a one-day one-sigma adverse move, as a fraction "
            "of NAV, per position. Live only under 'scenario' and 'full'; "
            "inert under the default 'nav_fraction' rule"
        ),
    )
    parser.add_argument(
        "--nav-fraction",
        type=float,
        default=SizeBounds().nav_fraction,
        help=(
            "gross premium one position may commit, as a fraction of NAV. Under "
            "the default 'nav_fraction' rule this IS the sizing unit -- the "
            "number that sets every size, so the book's deployment is roughly "
            "P x this. Under 'scenario' it is a live ceiling instead. Since "
            "the max-loss ceiling was removed (2026-09-22) this also sets the "
            "ultimate risk, at f x (max_loss / gross_premium) x NAV per "
            "position -- unbounded above, because that ratio is"
        ),
    )
    # ``--max-loss-cap`` was removed 2026-09-22 with the ceiling itself.  It is
    # deliberately not accepted-and-ignored: argparse will reject it, so a stale
    # sbatch fails loudly instead of running at a risk setting the caller
    # believes is in force and is not.
    parser.add_argument("--reasoning-effort", default="high")
    # How many recent assistant turns carry their reasoning back into the
    # prompt.  0 reproduces every arm run before 2026-09-23.  It is an arm-level
    # setting rather than a new global default because only some models need it:
    # measured that day, ``moonshotai/kimi-k3`` returns zero reasoning
    # characters at conversation depth 19 on Bedrock and on three OpenRouter
    # serves alike, while ``deepseek`` models carry a trace on 491 of 492 steps
    # under the same text-only history.  Replaying the *whole* trace is not an
    # option -- at ~1,522 tokens a step it puts a 20-decision episode at 134% of
    # the 32k budget -- so the window is the knob, and it goes in the manifest
    # because an arm that claims a window it did not run is not reproducible.
    parser.add_argument("--reasoning-replay-turns", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=32_768)
    # How --max-context-tokens is counted.  Defaults to the len//4 estimate for
    # backwards compatibility only: it under-reads Qwen3 on this state space by
    # 1.85x (measured over 378 episodes, scratch/qwen3_seq_len.py), so a run
    # left on the default is a run whose budget was not really checked.  The
    # counter's identity is written to the manifest either way, so which ruler
    # a trajectory was measured with is always recoverable.
    parser.add_argument(
        "--token-counter",
        default=HEURISTIC_SPEC,
        metavar="PATH|heuristic",
        help=(
            "path to a HuggingFace tokenizer.json (or a directory holding one) "
            "to count the context budget exactly, or 'heuristic' for len//4. "
            "An unreadable path is an error, never a fallback."
        ),
    )
    # The quote round and the wire it runs on, as one flag, because they are one
    # decision: the state space prints the ``Q`` verb under ``text`` and
    # declares the ``quote_package`` tool under ``tool``, and it must not do
    # both.  ``off`` reproduces every arm before the verb existed.
    #
    # ``tool`` is the default because it is the 2026-09-24 ruling and a
    # forgotten flag should run what was ruled.  Only ``--policy bedrock``
    # speaks it; the OpenRouter arm refuses it at ``reset`` rather than dropping
    # the schema, so a deepseek arm now needs ``--quote-channel text`` (its
    # previous behaviour) explicitly.  Refusing at startup costs one relaunch;
    # accepting it would have cost a month of steps with the ``Q`` lines gone
    # from the prompt and no tool to replace them.
    parser.add_argument(
        "--quote-channel", choices=("off", "text", "tool"), default="tool"
    )
    # ``None``, not a number.  A literal default here shadows the policy's own,
    # and the two drifted: the client was raised to clear r1's measured
    # reasoning appetite while this stayed at 4,000, so every live step kept
    # truncating even though the unit tests passed.
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    # A seed sweep is only a sweep if the sampler is actually sampling.  At the
    # default ``--temperature 0`` decoding is greedy and the seed is inert, so
    # N seeds cost N times the API budget and measure only the host-to-host
    # nondeterminism already documented on ``DeepSeekPolicy.providers`` -- not
    # policy variance.  Raise the temperature to get error bars; leave it at 0
    # to reproduce a single arm.
    parser.add_argument("--temperature", type=float, default=0.0)
    # For endpoints that do not accept a temperature at all.  Every OpenAI
    # reasoning model is one: ``openai/gpt-5``'s OpenRouter endpoints list
    # ``seed`` among their supported parameters and do not list ``temperature``,
    # and OpenRouter drops an unsupported parameter instead of rejecting it.
    # Sending one anyway is therefore silent -- the run finishes, records
    # ``temperature 0.0``, and was in fact sampled at the provider's own fixed
    # temperature, which is why ten seeds under it return ten different
    # trajectories.  Omitting the field makes the arm record ``null``, which is
    # the true statement: this arm did not set a temperature.
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="omit temperature from the request; for endpoints that ignore it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="BookState json from a prior run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report coverage and the episode plan, then exit without stepping",
    )
    parser.add_argument(
        "--mark-quote-cache",
        type=Path,
        default=None,
        help=(
            "JSONL cache of per-contract NBBOs used to mark positions the chain "
            "cannot price. Without it such legs are carried at entry price, so "
            "their NAV stops moving. Marking only; never an execution price."
        ),
    )
    parser.add_argument(
        "--mark-quotes-offline",
        action="store_true",
        help=(
            "replay --mark-quote-cache without network access, so a cached run "
            "reproduces on a compute node with no egress"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _hedged_families(text: str) -> tuple[str, ...]:
    """Resolve ``--hedge`` into the family tuple ``HedgeSpec`` wants.

    An unknown family name is refused rather than ignored.  Silently dropping
    one produces a run that reports hedging and does not hedge that family, and
    the difference only shows up as an unexplained delta in the NAV series.
    """
    choice = text.strip().lower()
    if choice in ("none", ""):
        return ()
    if choice == "volatility":
        return VOLATILITY_FAMILIES
    if choice == "all":
        return EnvConfig().admitted_families
    families = tuple(f.strip() for f in text.split(",") if f.strip())
    unknown = [f for f in families if f not in EnvConfig().admitted_families]
    if unknown:
        raise ValueError(
            f"--hedge names families that are not admitted: {', '.join(unknown)}"
        )
    return families


def build_config(args: argparse.Namespace) -> EnvConfig:
    """Turn the parsed flags into the config the whole run is defined by.

    Separate from :func:`main` so it can be exercised without a data root: this
    is where ``--hedge`` becomes the thing the resolver reads, and a flag that
    parses but does not reach ``HedgeSpec`` is invisible until a run finishes.
    """
    sessions = tuple(
        s.strip().upper() for s in args.decision_sessions.split(",") if s.strip()
    )
    hedged = _hedged_families(args.hedge)
    return EnvConfig(
        grid=GridSpec(decision_sessions=sessions),
        risk_free_rate=args.risk_free_rate,
        max_context_tokens=args.max_context_tokens,
        # One flag, two fields, for the reason ``hedge.enabled`` is derived
        # below: two switches can disagree and the disagreement is silent.
        # ``off`` has no channel to name, so it keeps the ruled one and is
        # distinguished by ``quotes_enabled``.
        quotes_enabled=args.quote_channel != "off",
        quote_channel="text" if args.quote_channel == "text" else "tool",
        # Read off the same condition that decides whether a mark source is
        # built at all (below), so the fingerprint cannot claim a source the
        # run does not have.  The version, not ``args.mark_quote_cache``: the
        # path is per-arm scratch and would make every draw a distinct arm.
        mark_quote_source=(
            MARK_QUOTE_SOURCE_VERSION if args.mark_quote_cache is not None else None
        ),
        size=SizeBounds(
            size_rule=args.size_rule,
            max_positions=args.max_positions,
            max_positions_per_underlying=args.max_positions_per_underlying,
            target_scenario_risk=args.target_scenario_risk,
            vol_shock_relative=args.vol_shock,
            nav_fraction=args.nav_fraction,
        ),
        # ``enabled`` follows the family list rather than being a second switch.
        # Two independent flags can disagree, and the disagreement is silent:
        # an arm reading ``--hedge volatility`` in its manifest while hedging
        # nothing is worse than an arm that never hedged at all.
        hedge=HedgeSpec(
            enabled=bool(hedged),
            hedged_families=hedged,
            delta_band=args.delta_band,
            band_rule=args.band_rule,
            risk_aversion=args.risk_aversion,
            band_multiple=args.band_multiple,
            min_band_fraction=args.min_band_fraction,
            max_band_fraction=args.max_band_fraction,
            fallback_half_spread=args.fallback_half_spread,
        ),
        # Separate from ``band_rule`` because they are separate claims: the band
        # rule changes *when* the hedge trades, the flag changes *what it pays*.
        # Bundling them would make the two-arm comparison uninterpretable.
        flags=FeatureFlags(
            measured_spread_costs=args.measured_spread_costs,
            suppress_textual_context=args.no_textual_context,
            suppress_future_knowledge=args.suppress_future_knowledge,
            anonymize=args.anonymize,
        ),
    )


def build_policy(args: argparse.Namespace, config: EnvConfig) -> Policy:
    # Refused rather than ignored.  The flag is set from an environment variable
    # in the sbatch, and an arm that exported it against a non-replay policy
    # would run to completion believing it had loosened a comparison it never
    # made -- the same shape of failure as a tokenizer path that falls back.
    if args.replay_tolerance != EXACT and args.policy != "replay":
        raise ValueError(
            f"--replay-tolerance {args.replay_tolerance} only means something "
            f"for --policy replay, and this run is --policy {args.policy}"
        )

    if args.policy == "hold":
        return HoldPolicy()

    if args.policy == "replay":
        if args.replay_from is None:
            raise ValueError("--policy replay requires --replay-from <decisions.jsonl>")
        if not args.replay_from.exists():
            raise ValueError(f"--replay-from {args.replay_from} does not exist")
        return ReplayPolicy.from_path(
            args.replay_from, tolerance=args.replay_tolerance
        )

    if args.policy == "package_model":
        from portfolio_monkey.baselines.policy import (
            FittedPackageModels,
            PackageModelPolicy,
        )

        if not args.package_models:
            raise ValueError(
                "--policy package_model requires --package-models <dir>. Build it "
                "with scripts/analysis/fit_package_models.py; the classifiers and "
                "the abstain threshold are frozen on disk with their fit window "
                "recorded, and this policy will not fit either at run time."
            )
        return PackageModelPolicy(
            FittedPackageModels.load(Path(args.package_models)), config
        )

    if args.policy in ("garch_vrp", "econ_rules"):
        # Refused rather than ignored, for the same reason ``--replay-tolerance``
        # is above: these flags reach the job from an sbatch environment, and a
        # baseline that silently accepted a sampling setting would record one in
        # its manifest that no code path ever read.  There is no sampling here;
        # the mapping is a chain of comparisons.
        ignored = [
            flag
            for flag, given in (("--reasoning-effort", args.reasoning_effort != "high"),
                                ("--reasoning-replay-turns", args.reasoning_replay_turns),
                                ("--provider", args.provider),
                                ("--model", args.model))
            if given
        ]
        if ignored:
            raise ValueError(
                f"--policy {args.policy} is deterministic and calls no model, so "
                f"{', '.join(ignored)} would be recorded and never applied. Drop "
                "them rather than letting the manifest claim them."
            )
        if args.anonymize:
            # The forecast table is keyed by real ticker, and under
            # ``--anonymize`` the M rows arrive as U01..U10, so every lookup
            # would miss and the policy would report "no signal" on all ten
            # names for all 124 dates -- a run that completes, holds throughout,
            # and looks like a baseline that chose not to trade.  Anonymisation
            # exists to stop a language model using memorised knowledge of a
            # ticker; a rule has none to use, so there is nothing to defend here
            # and no reason to accept the flag.
            raise ValueError(
                f"--policy {args.policy} reads a per-ticker table and cannot run "
                "under --anonymize: every lookup would miss silently and the arm "
                "would hold for the whole window. Drop --anonymize."
            )
        path = Path(args.baseline_params)
        if not path.is_file():
            raise ValueError(
                f"--baseline-params {path} does not exist. Build it first with "
                "scripts/analysis/fit_baseline_params.py; these policies will not "
                "invent a threshold at run time."
            )
        params = load_params(path)
        factory = (GarchVolPremiumPolicy if args.policy == "garch_vrp"
                   else EconomicRulePolicy)
        return factory(params, config)

    optional = (
        {"max_completion_tokens": args.max_completion_tokens}
        if args.max_completion_tokens is not None
        else {}
    )

    if args.policy == "bedrock":
        # Refused rather than ignored.  Bedrock rejects ``temperature``,
        # ``topP`` and ``seed`` for this model by name, so an arm that accepted
        # them would record a sampling setting in its manifest that the endpoint
        # never applied -- and four identical calls to it return four different
        # outputs, so the recorded setting would be actively misleading rather
        # than merely inert.
        unsupported = [
            flag
            for flag, given in (("--temperature", args.temperature), ("--seed", args.seed))
            if given
        ]
        if unsupported:
            raise ValueError(
                f"--policy bedrock does not support {', '.join(unsupported)}: the "
                "endpoint rejects temperature, topP and seed for this model"
            )
        if args.provider or args.no_provider_fallbacks:
            raise ValueError(
                "--provider and --no-provider-fallbacks are OpenRouter routing "
                "controls; --policy bedrock has no host to pin"
            )
        from portfolio_monkey.env.policy.bedrock import BedrockPolicy

        # ``--reasoning-replay-turns`` was refused here until 2026-09-23, when
        # ``BedrockPolicy`` grew the replay.  It is passed rather than dropped
        # because the whole reason the flag exists is that a run can look
        # completely healthy while producing no reasoning at all, so a window
        # written into the manifest and never applied is the failure it was
        # added to make visible.
        return BedrockPolicy(
            model=args.model,
            region=args.bedrock_region,
            name=args.arm,
            reasoning_effort=args.reasoning_effort,
            max_context_tokens=args.max_context_tokens,
            token_counter=build_token_counter(args.token_counter),
            reasoning_replay_turns=args.reasoning_replay_turns,
            **optional,
        )

    # Refused rather than silently resolved: the two flags state opposite things
    # about the arm, and picking a winner would record a sampling setting the
    # operator did not choose.
    if args.no_temperature and args.temperature:
        raise ValueError(
            "--no-temperature and --temperature are mutually exclusive: the "
            "first says this endpoint takes no temperature, the second sets one"
        )

    # Imported here so that a hold run needs no API key.
    from portfolio_monkey.env.policy.deepseek import DeepSeekPolicy

    return DeepSeekPolicy(
        model=args.model,
        name=args.arm,
        reasoning_effort=args.reasoning_effort,
        max_context_tokens=args.max_context_tokens,
        token_counter=build_token_counter(args.token_counter),
        providers=tuple(p.strip() for p in args.provider.split(",") if p.strip()),
        allow_provider_fallbacks=not args.no_provider_fallbacks,
        temperature=None if args.no_temperature else args.temperature,
        seed=args.seed,
        reasoning_replay_turns=args.reasoning_replay_turns,
        **optional,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.risk_free_rate is None and not args.no_rf:
        print(
            "refusing to run without --risk-free-rate: protocol section 5 requires "
            "cash to accrue r_f, and omitting it puts a 4-5% annual headwind on the "
            "book that no benchmark pays. Pass --no-rf to accept that deliberately.",
            file=sys.stderr,
        )
        return 2

    if args.measured_spread_costs and args.spread_table is None:
        # Refused rather than silently flattened.  Without a table the resolver
        # has nothing to look up, so the fill would pay the same flat cost it
        # always did while the manifest recorded ``measured_spread_costs: true``
        # -- an arm that reads as the treatment and ran as the control.
        print(
            "--measured-spread-costs requires --spread-table: with no table the "
            "share fill pays the flat cost.stock_half_spread_bps and the flag is "
            "inert. Build one with scripts/analysis/build_underlying_spreads.py.",
            file=sys.stderr,
        )
        return 2

    spreads: SpreadTable | None = None
    if args.spread_table is not None:
        # Loaded before the chain, for the same reason the policy is: a bad path
        # should cost a second on the login node, not the minutes it takes to
        # open the chain on a compute node.
        try:
            spreads = SpreadTable.from_csv(
                args.spread_table, fallback=args.fallback_half_spread
            )
        except SpreadTableError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        config = build_config(args)
        # Built before the data root is touched.  A missing recording is a typo
        # in the submit command, and finding it after the chain has been opened
        # means finding it minutes later on a compute node.
        policy = build_policy(args, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    root = data_root(args.data_root)
    chain = OptionChain(config.resolver, root=root)
    features = DatasetFeatureSource(root, extract_dir=_extract_dir(args, root))

    dates = [d for d in chain.coverage if args.start <= d <= args.end]
    if len(dates) < 2:
        print(
            f"option_chain_snapshots covers {len(dates)} date(s) in "
            f"[{args.start}, {args.end}] under {root}; nothing to run",
            file=sys.stderr,
        )
        return 1

    points = build_grid(dates, config)
    episodes = monthly_episodes(points)
    gaps = _coverage_gaps(features, config, dates)

    tradeable = config.universe.tradeable
    chain_by_name = chain.name_coverage(tradeable, dates)
    tradeable_dates = sum(len(v) for v in chain_by_name.values())

    if not args.quiet:
        print(f"window      {dates[0]} .. {dates[-1]}  ({len(dates)} trading dates)")
        print(f"grid        {len(points)} marks, {sum(p.is_decision for p in points)} decisions")
        print(f"episodes    {len(episodes)}: {', '.join(e.episode_id for e in episodes)}")
        for dataset, missing in sorted(gaps.items()):
            print(f"coverage    {dataset}: missing {len(missing)} of {len(dates)} dates")
        print(
            "chain       "
            + "  ".join(f"{n} {len(chain_by_name[n])}/{len(dates)}" for n in tradeable)
        )
        print(
            f"band        {config.hedge.band_rule}"
            + (
                f"  rra {config.hedge.risk_aversion:g}"
                f"  x{config.hedge.band_multiple:g}"
                f"  clamp [{config.hedge.min_band_fraction:g}, "
                f"{config.hedge.max_band_fraction:g}] of NLV"
                if config.hedge.band_rule == "whalley_wilmott"
                else f"  {config.hedge.delta_band:g} of NLV"
            )
        )
        if spreads is not None:
            # Keyed by the ticker whose *shares* trade, not by the underlying:
            # the index slot hedges with a proxy, and a table checked against
            # the option's underlying would report a hole that is not there.
            hedge_tickers = sorted(
                {
                    t
                    for name in tradeable
                    if (t := config.hedge.hedge_ticker_for(name)) is not None
                }
            )
            for label, session in (("AM", "market_open"), ("PM", "market_close")):
                present = spreads.present(hedge_tickers, dates, session=session)
                print(
                    f"spreads {label}  "
                    + "  ".join(f"{n} {present[n]}/{len(dates)}" for n in hedge_tickers)
                )

    # The date check above only asks whether *someone* was resolved that day.
    # Asking whether *these* names were is the difference between an arm that
    # declined to trade and an arm that could not: both write a ledger that
    # reconciles, and only this line tells them apart.
    if tradeable_dates == 0:
        print(
            f"no tradeable name has chain rows on any of the {len(dates)} dates in "
            f"[{args.start}, {args.end}]; every order would fail E_NO_CHAIN. Rebuild "
            "option_chain_snapshots for the universe before running an arm.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        # Print what the model would actually be sent.  The shape of the run is
        # the cheap half of the check; the expensive mistake is a prompt that
        # renders every field unavailable, and that is invisible until the first
        # observation is built.  Building one costs a chain read and no tokens.
        _print_first_prompt(config, chain, features, episodes[0])
        return 0

    carried = None
    if args.resume_from is not None:
        carried = BookState.from_dict(
            json.loads(args.resume_from.read_text(encoding="utf-8"))
        )
        if not args.quiet:
            print(f"resuming    nav {carried.nav:,.2f} as of {carried.as_of}")

    args.out.mkdir(parents=True, exist_ok=True)
    ledger = LedgerWriter(args.out, arm=args.arm, track=args.track)
    marks = (
        MassiveMarkQuotes(
            cache_path=args.mark_quote_cache, offline=args.mark_quotes_offline
        )
        if args.mark_quote_cache is not None
        else None
    )
    env = OptionsEnv(
        config,
        chain=chain,
        features=features,
        ledger=ledger,
        marks=marks,
        spreads=spreads,
    )

    if not args.quiet:
        # Printed because it was once wrong: an ambient ``OPENROUTER_MODEL``
        # replaced the arm's model, and the only place that showed was the
        # per-decision error rows after the run had finished paying for itself.
        print(f"policy      {args.policy}  model {getattr(policy, 'model', '-')}")

    try:
        result = run_arm(
            env,
            policy,
            episodes,
            arm=args.arm,
            track=args.track,
            ledger=ledger,
            state_dir=args.out / "state",
            carried_in=carried,
            manifest_extra={
                "window": [dates[0].isoformat(), dates[-1].isoformat()],
                "feature_coverage_gaps": {k: len(v) for k, v in gaps.items()},
                "chain_coverage": {k: len(v) for k, v in chain_by_name.items()},
                # Which sample this is.  Recorded next to the window rather than
                # inferred from the arm name, because the main and ablation
                # samples are meant to differ in exactly this one field and an
                # arm named "ablation" that ran with news is otherwise
                # indistinguishable from one that did not.  The rendered rows
                # are not proof either: a month with no news would produce an
                # N-free prompt under both settings.
                "textual_context": not config.flags.suppress_textual_context,
                # Recorded for the same reason: the prompt is the arm. A
                # trajectory generated under the suppression line is not
                # comparable to one generated without it, and nothing else in
                # the ledger would say which it was.
                "suppress_future_knowledge": config.flags.suppress_future_knowledge,
                # Same reason again, with one addition: this flag is the only
                # one whose effect is *invisible in the ledger by design*. The
                # ledger keeps real tickers and real dates whether or not the
                # policy ever saw them, so a de-identified run and a plain one
                # produce the same columns. Without this field the two are
                # indistinguishable after the fact, and ``decisions.jsonl``
                # would be read as though the model had been shown the names.
                #
                # ``env_fingerprint`` above covers ``flags``, so this claim is
                # checkable rather than merely asserted:
                # ``scripts/analysis/prove_prompt_applied.py`` rebuilds the
                # config both ways and requires exactly one to reproduce it.
                "anonymize": config.flags.anonymize,
                # The sweep's independent variables, recorded where the result
                # is.  ``env.hedge_stats`` says what the bands came out to; this
                # says what was asked for, and the two only agree when the
                # clamps did not eat the arm.
                "hedge_band": {
                    "rule": config.hedge.band_rule,
                    "delta_band": config.hedge.delta_band,
                    "risk_aversion": config.hedge.risk_aversion,
                    "band_multiple": config.hedge.band_multiple,
                    "min_band_fraction": config.hedge.min_band_fraction,
                    "max_band_fraction": config.hedge.max_band_fraction,
                    "fallback_half_spread": config.hedge.fallback_half_spread,
                    "spread_table": str(args.spread_table) if args.spread_table else None,
                    "measured_spread_costs": config.flags.measured_spread_costs,
                },
                # ``limits`` is derived, not another input.  It is recorded
                # anyway because it is the thing a reader of two manifests
                # actually wants to diff, and deriving it requires having this
                # version of ``SIZE_RULE_LIMITS`` to hand -- which a manifest
                # read a year from now will not.
                "sizing": {
                    "rule": config.size.size_rule,
                    "limits": list(SIZE_RULE_LIMITS[config.size.size_rule]),
                    "target_scenario_risk": config.size.target_scenario_risk,
                    "vol_shock_relative": config.size.vol_shock_relative,
                    # ``max_loss_cap`` was recorded here until 2026-09-22 and is
                    # gone with the ceiling.  Its absence is the version marker:
                    # a manifest carrying it was written under a rule that
                    # bounded ultimate risk at a flat fraction of NAV.
                    "nav_fraction": config.size.nav_fraction,
                    "max_positions": config.size.max_positions,
                },
                # Arm identity again, and the field a reader most needs spelled
                # out: ``env_fingerprint`` moves when ``quotes_enabled`` does,
                # but a hash says only *that* something changed.  A quote arm
                # spends two turns per step and carries a RES block forward on
                # every later step of the month, so its prompt curve is not
                # comparable to a one-turn arm's -- and ``decisions.jsonl``
                # would otherwise be read as one row per step.  The cap goes
                # next to the switch because it is what the token budget was
                # sized against; it was 8 until 2026-09-24, then 4, then 6.
                #
                # ``channel`` is the third field because the two wires produce
                # the same ``decisions.jsonl`` by design -- the tool call is
                # normalized to its ``Q`` line before it is written -- so
                # nothing in the panel distinguishes them.  What differs is the
                # prompt: under ``tool`` the ``Q`` grammar is gone and a ~468
                # token schema sits at the head of the prefix instead, which is
                # enough to make two arms' token curves non-comparable.
                "quotes": {
                    "enabled": config.quotes_enabled,
                    "max_per_name": config.max_quotes_per_name,
                    "channel": config.quote_channel,
                },
            },
        )
    finally:
        ledger.close()

    check = reconcile(args.out)
    quality = decision_quality(args.out)

    summary = {
        **result.as_dict(),
        "reconciliation": {
            "ok": check.ok,
            "steps": check.steps,
            "max_error": check.max_error,
            "worst_step": check.worst_step,
            "by_identity": dict(check.by_identity),
        },
        "decision_quality": quality.as_dict(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print(
            f"result      nav {result.start_nav:,.2f} -> {result.end_nav:,.2f}  "
            f"log return {result.log_return:+.4f}"
        )
        if result.terminated:
            print(f"terminated  {result.terminated}")
        if result.failure:
            print(f"failure     {result.failure}")
        print(
            f"decisions   {quality.decisions}, abstain {quality.abstain_rate:.0%}, "
            f"fill {quality.fill_rate:.0%}, provider errors "
            f"{quality.provider_error_rate:.0%}"
        )
        if quality.prompt_tokens:
            print(
                f"tokens      {quality.prompt_tokens:,} prompt "
                f"({quality.cache_hit_rate:.0%} cached), "
                f"{quality.completion_tokens:,} completion"
            )
        print(f"reconcile   {'ok' if check.ok else 'FAILED'} over {check.steps} steps")
        hedged = env.hedge_stats
        if hedged["examined"]:
            # The sweep's read-out.  A ``whalley_wilmott`` arm whose min and max
            # are both a clamp ran the fixed rule under another name, and the
            # fallback count says whether the spreads it used were measured.
            span = (
                f"{hedged['band_min']:,.0f} .. {hedged['band_max']:,.0f}"
                if hedged["band_min"] is not None
                else "-"
            )
            fallbacks = sum(hedged.get("spreads", {}).get("fallbacks", {}).values())
            print(
                f"hedge       {hedged['orders']} orders over {hedged['examined']} "
                f"groups, band ${span}, peak exposure "
                f"{hedged['peak_exposure_ratio']:.2f}x band"
                + (f", {fallbacks} spread fallbacks" if spreads is not None else "")
            )

    # A ledger that does not reconcile is not a result.  Every metric downstream
    # reads it, so shipping a non-zero exit here is the difference between a
    # caught bug and a reported number that is wrong.
    if not check.ok:
        for line in check.detail[:5]:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0 if result.failure is None else 1


def _extract_dir(args: argparse.Namespace, root: Path) -> Path | None:
    if args.feature_cache is None:
        return root / ".feature_extracts"
    if args.feature_cache.lower() == "none":
        return None
    return Path(args.feature_cache)


def _print_first_prompt(config, chain, features, episode) -> None:
    """Render the system block, the grammar and the first observation.

    Uses a throwaway env with no ledger: ``--dry-run`` must not leave a
    half-written run directory that a later build would read as a real one.
    """
    # ``mark_quote_source`` cleared because this env is handed no mark source
    # and the environment refuses a config that claims one it was not given.
    # Safe here and nowhere else: nothing below values a position, and the
    # prompt does not depend on the field -- it is fingerprint material only.
    env = OptionsEnv(
        config.with_(mark_quote_source=None), chain=chain, features=features
    )
    print()
    print(env.state_space.system_block(config))
    print()
    print(env.state_space.action_grammar(config))
    print()
    print(env.episode_header(episode, carried_in=False))
    view = env.reset(episode)
    while view.observation is None and not view.done:
        view = env.step(None)
    if view.observation is not None:
        print()
        print(view.observation.text)


def _coverage_gaps(
    features: DatasetFeatureSource, config: EnvConfig, dates: Sequence[date]
) -> dict[str, list[date]]:
    """Which of the state space's datasets are missing which dates.

    Reported rather than fatal: ``docs/state_space.md`` section 7 already
    expects fields to go missing, and the state space renders them unavailable
    rather than guessing.  A run on partial features is a real run with a
    recorded caveat; a run on a missing *chain* is not, which is why the chain
    is checked separately and stops the job.
    """
    from portfolio_monkey.env.statespace import build_state_space

    wanted = set(dates)
    gaps: dict[str, list[date]] = {}
    for dataset in build_state_space(config).required_datasets():
        try:
            covered = set(features.coverage(dataset))
        except Exception:  # noqa: BLE001 - a missing dataset dir is a full gap
            covered = set()
        missing = sorted(wanted - covered)
        if missing:
            gaps[dataset] = missing
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
