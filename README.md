# SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions

Code release accompanying the paper.

Agentic option trading requires an autonomous policy to select complex option
strategies from nuanced market conditions. SOTA is a language-model
post-training framework for dynamic option strategy selection. It represents
market conditions using features of the option-implied return distribution and
selects among nine structured option-strategy families. The policy is first
supervised on frontier-model trading trajectories, then optimized with
reinforcement learning on realized portfolio performance net of transaction
costs. Evaluation covers options on nine large-cap U.S. equities and SPY against
non-LLM strategy-selection baselines.

---

## What this repository implements

The pieces that produce the paper's numbers, and nothing else:

* **The trading environment** — a point-in-time replay over historical option
  chains: contract resolution, position sizing, delta hedging, fills against
  NBBO, American-style marking, transaction costs, and the portfolio ledger.
* **The observation encoder** — the compact per-step state the policy reads,
  including the news changelog.
* **The action space** — strategy *intent* in a small DSL over nine families,
  plus the deterministic resolvers that turn intent into contracts and hedge
  trades. The policy never names a contract.
* **Supervised fine-tuning** — corpus construction from teacher trajectories,
  the outcome gate that filters them, and the loss mask.
* **Reinforcement learning** — GRPO over the live environment, as an agent loop
  driving a veRL trainer.
* **Evaluation** — the metric stack and the block bootstrap.
* **Baselines** — the rule-based and machine-learning selectors reported in the
  paper, and the ex-post oracle.

A deliberate omission: the vendor ingestion pipeline that *builds* the datasets
is not included. It is large, it is specific to licensed feeds, and it is not
what the paper is about. What is included is the reader interface plus the
schemas in [`examples/`](examples), so the expected inputs are fully specified.

## Repository structure

```
src/portfolio_monkey/       the importable package (66 modules)
    env/                    environment, resolvers, costs, ledger, state encoding
        policy/             rule baselines, replay, LLM clients
    baselines/              the supervised per-package selectors (GBDT, Logistic)
    training/               corpus construction, reward, agent loop, veRL backend
    eval/                   NAV panel builders and the metric stack
    jobs/                   the episode runner
src/pm_train/               RL integration layer: agent loop, rollout session,
                            evaluation driver, campaign scripts (10 modules)
scripts/
    rl/                     RL campaign entry points
    analysis/               the scripts behind each reported table
configs/
    rl/                     the RL and SFT command lines and agent-loop configs
    baselines/              rule constants, fitted GARCH parameters, classifier metadata
    evaluation/             the resolved environment configuration
    dates_{sft,rl,eval}.txt the three date windows, where the code reads them
examples/                   input schemas + one synthetic trajectory
results/                    the main results table, per-checkpoint training and
                            evaluation metadata
docs/                       data requirements and per-result provenance
```

The package keeps its development name `portfolio_monkey`. The paper calls the
system SOTA; the two refer to the same thing. Renaming it across 66 modules
would have been a mechanical change with no benefit to a reader and some risk to
the computation, so it was left alone.

## Installation

```bash
git clone https://github.com/JenniceXie/sota-options-rl
cd sota-options-rl
python -m pip install -e '.[chain]'      # minimum to step the environment
python -m pip install -e '.[baselines]'  # the GBDT and Logistic baselines
python -m pip install -e '.[dev]'        # plus tokenizer, pandas, transformers, pytest
```

`dependencies` is deliberately empty so that installing this cannot argue with a
pinned `torch`/`vllm` stack during the RL stage. That is not the same as being
pure standard library — see the extras in `pyproject.toml`. **`chain` is not
optional if you intend to run anything**: the option chain is stored as parquet
and only as parquet, so the first chain read of the first step needs `pyarrow`.

The RL stage additionally needs veRL and vLLM. The reported runs used veRL 0.7.1,
vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.1 and flash-attn 2.8.3 on a
single 8-GPU node. Those are not declared here, because pinning them would break
the install for anyone who only wants the environment.

## Data

### What is in this repository

Schemas, configurations, the main results table, per-checkpoint metrics, and one
synthetic trajectory. No market data, no news, no corpora, no model weights, and
no trained baseline classifiers (their metadata is in `configs/baselines/`).

### What you must obtain yourself

The environment reads six market datasets and the news rows, over 10 underlyings
and 246 trading dates:

| dataset | content |
| --- | --- |
| `option_chain_snapshots` | per-contract NBBO and greeks (parquet) |
| `option_iv_surface_features` | implied-volatility level, term slope, skew, curvature |
| `option_delta_point_surface` | per-delta-point implied volatility |
| `option_open_interest_features` | open interest and its change |
| `compact_option_flow_states` | option flow imbalance |
| `underlying_market_features` | underlying returns and realized volatility |
| `news_catalyst_rows` | the news changelog |

These derive from commercial feeds (an options market-data vendor, CRSP/WRDS,
SpiderRock, TAQ) under licences that do not permit redistribution. **They are not
included and cannot be supplied by us.** Field-level layouts are in
[`examples/market_state_schema.json`](examples/market_state_schema.json) and
[`examples/news_schema.json`](examples/news_schema.json); point `PM_DATA_ROOT` at
a tree matching those schemas.

### What is private and will not be released

* **The supervised trajectories.** They are frontier-teacher outputs computed
  over licensed market and news data. The record format is fully specified in
  [`examples/sft_trajectory_schema.json`](examples/sft_trajectory_schema.json),
  with a hand-written synthetic record in
  [`examples/synthetic_sft_example.jsonl`](examples/synthetic_sft_example.jsonl)
  — clearly labelled synthetic, describing no real market state.
* **The news corpus and its summaries.**
* **Model weights.** No checkpoint is released. `results/rl_training_metrics.csv`
  carries the per-checkpoint training reward, validation return and test metrics
  instead, including which checkpoint the paper reports.

Because those inputs are unavailable, this repository **cannot be run end to end
from its own contents.** It is published so the implementation can be read,
audited and adapted, not as a turnkey pipeline.

## The pipeline

### Market states

Each trading day, for each underlying, the encoder emits one `M` row of ten
scale-normalised integers: trailing return, realized volatility, at-the-money
implied volatility, its change, the implied-minus-realized wedge, term slope,
25-delta risk reversal, 25-delta butterfly, flow imbalance, and open-interest
change. A missing field renders an explicit sentinel, never a zero. **The policy
is never shown a price level or a spot quote.** An account row and one row per
open position complete the state. Every field is built under an availability
timestamp `<= t`; that is the leakage boundary.

### News

News enters as `N` rows in the same block. The block is a **changelog, not a
snapshot**: an item is served once, on the step where it first becomes
available, and is not re-served. Each row carries a category, a direction sign,
a horizon and a one-line entity-masked summary. A date whose news partition is
absent renders an explicit "no news feed for this date" row rather than raising,
so the policy is told the absence means nothing instead of inferring a quiet
day. The five numeric datasets do not get that tolerance: a missing partition
there stops the step, because a plausible `na` in a field the policy reads as a
measurement is worse than a halt.

### Supervised trajectories

A frontier teacher runs the full environment over the supervised window, through
the same harness the policy is evaluated in, producing one trajectory per
month-long episode. Trajectories pass an **outcome gate** before any token is
trained on — a risk-adjusted-return and drawdown threshold on the whole
trajectory, not on individual decisions. Episodes exceeding the token budget are
dropped at corpus-build time. `src/portfolio_monkey/training/selection.py` and
`measure.py` implement the gate; `corpus.py` writes the records.

### Supervised fine-tuning

Full fine-tuning of the student on the gated corpus, loss masked to assistant
turns only. The mask is load-bearing: the state block is one to two orders of
magnitude longer than the completion, so an unmasked objective trains mostly on
reproducing the prompt. Settings are in `configs/rl/cfg_sys_sft.txt` and
`cmd_sys_sft.sh`.

### Reinforcement learning

GRPO against the live environment, with no learned value function and no reward
model. At a state the policy samples a group of complete order lists; each is
resolved and executed independently; the reward is the change in log net asset
value after transaction costs. Advantages are group-normalised. Episodes are one
calendar month with the book carried into the next, because the append-only
conversation does not fit a longer trajectory inside the token budget.

`src/pm_train/` is the integration layer: `episode.py` and `procsession.py` run
the environment behind the trainer, `verl_loop.py` is the agent loop,
`reward_shim.py` exposes the trajectory-level reward, `evaluate.py` produces the
metrics. `scripts/rl/campaign27.py` drives the reported campaign.

### Evaluation

Ledgers are reduced to a NAV panel and then to metrics — total and annualized
return, annualized volatility, Sharpe, Sortino, Calmar, maximum drawdown,
skewness, kurtosis, turnover, frictions and cost share — with a calendar-month
block bootstrap for intervals. The split between panel construction
(`eval/builders/`) and metric computation (`eval/metrics.py`) is deliberate: state
variables change across experiments, reported metrics must not move with them.

## Baselines

| paper name | implementation | entry point |
| --- | --- | --- |
| Equal-weighted rule | 140 fixed-strategy arms (10 names × 14 family/orientation cells, 8–30 days to expiry), as a daily-rebalanced equal-weight composite | `scripts/analysis/run_baseline_arms.py`, then `interaction_tables.py` |
| GARCH rule | `GarchVolPremiumPolicy` | `python -m portfolio_monkey.jobs.run_policy_episodes --policy garch_vrp` |
| Threshold rule | `EconomicRulePolicy` | `… --policy econ_rules` |
| GBDT | per-package gradient-boosted classifiers | `… --policy package_model` |
| Logistic | per-package logistic classifiers | `… --policy package_model` |
| Ex-post oracle | hindsight upper reference, not a policy | `scripts/analysis/clairvoyant_oracle.py` |

**Threshold rule.** One fixed cut-off on one scale-free feature per strategy
family, covering all nine families; none is a fitted quantile. The
volatility-level families trade when implied volatility is a set margin away
from realized volatility, and each margin is sized against the family's
round-trip friction as measured on the supervised window. The curvature and skew
families key on a flat smile or skew, or a fixed skew level; the directional
families key on a fixed one-day move.

**GARCH rule.** A pooled GARCH(1,1), fitted on returns dated strictly before the
test window, supplies a causal volatility forecast. The rule sells volatility
through an iron condor when implied volatility is at least 25% above the
forecast, and buys it through a long straddle when implied is at least 10%
below, at 8–30 days to expiry. Both margins are sized against the same friction
measurements.

**GBDT and Logistic.** One binary classifier per family-and-orientation head,
trained on the supervised window to predict whether a package could have been
exited at a profit; the labels come from the ex-post oracle. A package is a
candidate when its predicted probability exceeds 0.5 and exceeds its head's own
training base rate. Candidates are ranked by that ratio, the lift, and opened as
the position caps allow. The lift cut-off of 1.0 was chosen on the
reinforcement-learning window. The rule uses lift rather than raw probability
because a single probability cut-off shared across heads ranks heads by their
base rates rather than by their signal.

No baseline parameter is selected on the test window. The rule constants are in
`configs/baselines/baseline_constants.json`; the GARCH fit and its calibration
windows are in `configs/baselines/params_fitted.json`; each classifier set's base
rates, validation threshold scan and training windows are in
`configs/baselines/package_models_*.meta.json`. The per-date GARCH forecast series,
which is computed from licensed returns, and the trained classifiers are not
included. The scripts that fit them are: `scripts/analysis/fit_baseline_params.py`
and `scripts/analysis/fit_package_models.py`.

## Results

Test window 2025-03-03 to 2025-08-29 (124 trading dates). Every row is computed
by `scripts/analysis/build_table1.py` from the run directories and written to
[`results/table1.csv`](results/table1.csv), which also records each row's
environment fingerprint, trade counts and metric path.

| policy | TR (%) | ASR | ACR | ASoR | AVOL (%) | MDD (%) | WR (%) | PLR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOTA | 18.32 | 1.60 | 4.59 | 4.77 | 21.52 | 8.96 | 29.17 | 4.68 |
| Equal-weighted rule | −5.22 | −13.46 | −1.98 | −11.54 | 0.81 | 5.22 | 31.11 | 1.08 |
| GARCH rule | −49.24 | −6.27 | −1.52 | −6.25 | 22.17 | 49.42 | 29.85 | 0.75 |
| Threshold rule | −28.46 | −1.95 | −1.50 | −3.50 | 35.20 | 33.16 | 41.31 | 1.89 |
| GBDT | −49.78 | −8.55 | −1.52 | −8.18 | 16.51 | 49.78 | 24.07 | 1.23 |
| Logistic | −55.45 | −11.63 | −1.46 | −9.70 | 14.24 | 55.47 | 20.42 | 1.00 |

TR total return; ASR, ACR, ASoR annualized Sharpe, Calmar and Sortino ratios
(risk-free rate 0); AVOL annualized volatility; MDD maximum drawdown; WR win rate
and PLR profit/loss ratio over closed positions. A position's result is its
round trip: realized P&L over the opening and the closing fill, so the entry cost
counts as well as the exit cost; delta-hedge fills are excluded. The CSV also
carries the ex-post oracle as a reference row. It is a hindsight bound on this
window, not an attainable result, and win rate and profit/loss ratio are
undefined for it.

**One environment for every row.** Every baseline ran in the environment the
language policy was trained and evaluated in: the same position sizing, no cash
accrual, vendor mark quotes, and the same hedging band, costs and decision grid.
Its configuration differs from the SOTA run's in exactly two keys, and both are
view-layer flags that change what the observation shows, not how orders execute:

* `flags.anonymize` relabels the underlyings in the observation so that a
  language model cannot use what it has memorized about a ticker. No baseline
  has anything memorized. The three rule baselines look names up by ticker, so
  under anonymization they would find no names and never trade. The classifiers
  score each market row without looking up its name. All five baselines ran
  with the flag off, so they share one configuration.
* `flags.suppress_textual_context` removes the news rows, which no baseline reads.

`build_table1.py` compares each row's configuration with the SOTA run's key by
key and writes the differing keys into the table, so this is checked, not
asserted.

## How the reported results were produced

[`docs/provenance.md`](docs/provenance.md) maps each reported table to the script,
configuration and run directory behind it. `results/rl_training_metrics.csv` gives,
per optimizer step and per arm, the mean training reward, the validation return on
the tuning window, and the test-window metrics of every evaluated checkpoint —
with the reported checkpoint flagged.

Every run directory records its own `env_fingerprint` in `manifest.json`, so any
comparison can be checked rather than assumed. The SOTA run's fingerprint is
`ab1abc60…`; every baseline run's is `621604cf…`; the two configurations differ
only in the two view-layer flags described under [Results](#results).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT, for the code in this repository. The datasets it reads are governed by their
own vendor licences and are not covered by it.
