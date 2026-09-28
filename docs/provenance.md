# What produced each reported result

Every run directory records its own configuration in `manifest.json`, including
an `env_fingerprint`. Where a number is quoted below it is recomputed from the
run's ledger or read from its `metrics.json`, not restated from the paper.

## Main comparison

`results/table1.csv` is written by

```bash
python scripts/analysis/build_table1.py --runs "$PM_ARM_ROOT" \
  --dates configs/dates_eval.txt \
  --oracle "$PM_ARM_ROOT/oracle_S0.json" \
  --out results/table1.csv
```

from these runs, all on the test window 2025-03-03 .. 2025-08-29:

| row | run directory | `env_fingerprint` | produced by |
| --- | --- | --- | --- |
| SOTA | `c27_s4_none_step20` | `ab1abc60…` | the RL campaign's evaluation of the `s4_none` arm at `global_step_20` |
| Equal-weighted rule | `baselines_pername_rlenv`, the 140 arms at `8_30` | `621604cf…` | `run_baseline_arms.py` |
| GARCH rule | `b9_garch_vrp_rlenv` | `621604cf…` | `run_policy_episodes --policy garch_vrp` |
| Threshold rule | `b9_econ_rules_rlenv` | `621604cf…` | `run_policy_episodes --policy econ_rules` |
| GBDT | `pkgB_rlenv` | `621604cf…` | `run_policy_episodes --policy package_model` |
| Logistic | `pkgLogit_rlenv` | `621604cf…` | `run_policy_episodes --policy package_model` |
| Ex-post oracle | `oracle_S0.json` | — | `clairvoyant_oracle.py` |

Full fingerprints are in the CSV. `621604cf…` is the SOTA configuration with
`flags.anonymize` and `flags.suppress_textual_context` switched off; the CSV's
`env_config_keys_differing_from_SOTA` column lists, per row, every key on which
the run's recorded configuration differs from the SOTA run's.

**Code.** Each baseline job logged the source revision it ran from and checksums
of its key files. Every module those jobs imported is byte-identical to the one
in this repository; that covers `src/portfolio_monkey/`, `run_baseline_arms.py`,
`sweep_algorithmic_policies.py` and `clairvoyant_oracle.py`. The GARCH and
Threshold jobs also read a parameter artifact; its fitted values are in
`configs/baselines/params_fitted.json`. The SOTA row
comes from the language policy's own evaluation run. `build_table1.py`
recomputes its metrics from that run's ledger, and they agree with the sealed
`results/paper_results/s4_none.metrics.json`.

### Commands

Equal-weighted rule. All 560 cells are generated; the reported row is the 140 at
8–30 days to expiry. The original ran as four shards (`--shard i --shards 4`),
each with its own mark-quote cache.

```bash
python scripts/analysis/run_baseline_arms.py \
  --start 2025-03-03 --end 2025-08-29 --out "$PM_ARM_ROOT/baselines_pername_rlenv" \
  --data-root "$PM_DATA_ROOT" --hedge volatility \
  --tenors 0_7,8_30,31_90,91_180 \
  --names AAPL,AMZN,GOOGL,META,MSFT,MU,NVDA,PLTR,TSLA,SPY \
  --per-name --seeds 1 --mode fixed \
  --mark-cache "$PM_ARM_ROOT/baselines_pername_rlenv/marks/marks.jsonl" \
  --anonymize 0 \
  --target-fingerprint 621604cfc6b33214b9844ddba0dc8c3a13172b32a01c849cfc81ce7175880a7b
```

`--target-fingerprint` makes the runner refuse to start unless the configuration
it built is that one. The usage line in the script's docstring predates the
switch to this environment: `--mark-cache` is now required, and
`--risk-free-rate` is ignored because this environment accrues no cash.

GARCH and Threshold rules. The first command writes the complete parameter
artifact, including the per-date forecast that `configs/baselines/params_fitted.json`
omits. The reported runs read the artifact built on 2026-09-25, whose fitted
values that file records.

```bash
python scripts/analysis/fit_baseline_params.py configs/baselines/params.json \
  --data-root "$PM_DATA_ROOT" \
  --calibration-dates configs/dates_sft.txt configs/dates_rl.txt \
  --test-dates configs/dates_eval.txt

for P in garch_vrp econ_rules; do
  python -m portfolio_monkey.jobs.run_policy_episodes \
    --arm "b9_${P}_rlenv" --policy "$P" --out "$PM_ARM_ROOT/b9_${P}_rlenv" \
    --start 2025-03-03 --end 2025-08-29 \
    --size-rule scenario --no-rf --hedge volatility --quote-channel text \
    --mark-quote-cache "$PM_ARM_ROOT/marks_${P}.jsonl" \
    --baseline-params configs/baselines/params.json
done
```

`run_policy_episodes` refuses `--anonymize` for these two policies: they read a
per-ticker table, and under anonymized labels every lookup would miss.

GBDT and Logistic. The classifiers are fitted on the supervised window, with
labels from the oracle's candidate dump (`clairvoyant_oracle.py
--dump-candidates`) and features from a `--policy hold` run over the same dates.
The threshold is chosen on the reinforcement-learning window.

```bash
python scripts/analysis/fit_package_models.py "$PM_WORK_ROOT/package_models_v2" \
  --train-features "$PM_WORK_ROOT/feature_runs/feat_sft" \
  --train-dump "$PM_WORK_ROOT/oracle_labels/candidates_sft.jsonl" \
  --val-features "$PM_WORK_ROOT/feature_runs/feat_rl" \
  --val-dump "$PM_WORK_ROOT/oracle_labels/candidates_rl.jsonl" \
  --backend gbdt                 # --backend logistic for the Logistic row

python -m portfolio_monkey.jobs.run_policy_episodes \
  --arm pkgB_rlenv --policy package_model \
  --package-models "$PM_WORK_ROOT/package_models_v2" \
  --out "$PM_ARM_ROOT/pkgB_rlenv" --start 2025-03-03 --end 2025-08-29 \
  --no-rf --hedge volatility --size-rule scenario --quote-channel text \
  --mark-quote-cache "$PM_ARM_ROOT/marks_pkgB.jsonl"
```

The fitted models' metadata is in `configs/baselines/package_models_gbdt.meta.json`
and `package_models_logistic.meta.json`: heads, per-head training base rates, the
validation threshold scan, and the windows the classifiers and the threshold were
fitted on.

`baselines/policy.py` selects by **lift**, a head's predicted probability over its
own training base rate, with an absolute floor of 0.5. Raw probability against
one global cut-off would rank heads by base rate instead. At 0.70 it selects only
the four highest-base-rate heads, each with validation AUC between 0.50 and 0.55,
and excludes every head with real signal. The metadata's `threshold_units` field
records this.

Ex-post oracle, the environment's position sizing and hedging, PM decisions:

```bash
python scripts/analysis/clairvoyant_oracle.py \
  --start 2025-03-03 --end 2025-08-29 --out "$PM_ARM_ROOT/oracle_S0.json" \
  --data-root "$PM_DATA_ROOT" --decision-sessions PM --nav 1000000 \
  --half-spread-multiplier 1.0 --size-rule scenario \
  --target-scenario-risk 0.005 --vol-shock 0.10 --nav-fraction 0.10 \
  --hedge volatility --audit-bounds
```

The oracle chooses its trades with hindsight and applies no risk controls, so its
row is a bound on what this window admitted, not a result any policy could
attain. Its JSON has a `performance` block computed from simple returns; that
block is not used. `build_table1.py` rebuilds the oracle's `nav_curve` into a NAV
panel and runs the same metric code as every other row.

### Metric conventions

* **NAV metrics.** Single-run rows and the oracle use
  `eval.metrics.compute` on the PM-grid NAV panel: log returns, risk-free rate 0.
  The panel starts at the first PM close, which is after that day's fills, so TR
  excludes the first day's move. The equal-weighted row is a composite rather
  than a ledger. `interaction_tables.py` builds it as the daily-rebalanced mean
  of the constituents' simple returns, anchored at `manifest.start_nav`. The
  `n_daily_returns` column shows the one-day difference (123 against 124).
* **Win rate and profit/loss ratio.** Per closed position, realized P&L summed
  over its opening and closing fills, so both costs count; hedge fills are
  excluded. `WR_close_only_pct` and `PLR_close_only` count the closing fill only,
  which drops the entry cost. They are the definition `interaction_tables.trades`
  uses, and they are in the CSV for comparison.

### The equal-weighted composite

Two scripts in the source repository computed "equal-weighted composites" and
they answer different questions. The reported numbers come from
`interaction_tables.py`, which composites **across** tenor buckets unless
`--tenor-only` is given. The reported row is the `8_30` horizon marginal.

```bash
python scripts/analysis/interaction_tables.py --runs "$PM_ARM_ROOT/baselines_pername_rlenv" \
  --dates configs/dates_eval.txt
```

Four properties of that reduction worth stating, each of which produced a wrong
number once before being pinned down:

1. **The health gate is `manifest.failure` plus the NAV-panel span**, not
   `decisions[].error`. A run truncated by a missing data partition reports zero
   decision errors and a self-consistent ledger covering 19 of 124 dates.
   `build_table1.py` adds a third gate: every constituent must have opened at
   least one position.
2. **Anchor the NAV series at `manifest.start_nav`.** The panel's first row is
   written after that step's fills, so a series built from the panel alone loses
   the opening trade.
3. **Key arms by the full cell** (family *and* orientation), not the family code.
   Keying on the family alone collapses two orientations and drops 200 of 560 arms.
4. **A composite is the daily-rebalanced arithmetic mean of constituent simple
   returns** — not the mean of levels, and not the mean of log returns.

Win rate and profit/loss ratio come from `fills.realized`, which is net of cost
and ties to the NAV movement. Per-step mark-to-market is not a substitute: it
telescopes, and its overnight rows are degenerate.

"Horizon" in the marginals is **contract days-to-expiry, not holding period**. The
exit rule is identical in every bucket, so mean holding time is 3–4 days
throughout; a long-dated bucket is not a long-horizon strategy.

## News ablation

Both arms branch from the same supervised checkpoint and run the identical
recipe, differing only in whether the news changelog is retained during RL:
`s4_none` (state only) and `sys_sft_rl` (state + news), both at
`global_step_20`.

## Checkpoint curve

`results/rl_training_metrics.csv`, one row per optimizer step per arm:
mean training reward, per-month validation return on the tuning window, and the
test-window metrics of each evaluated checkpoint. Checkpoints were written every
10 steps; training ran to step 40.

Two gaps are visible in the file rather than smoothed over. The state-plus-news
arm's step-40 checkpoint was never evaluated on the test window. And validation
metrics for steps 30 and 40 are missing one month, though the underlying NAV
panels exist.

## Hyperparameters

`configs/rl/` holds the command lines and agent-loop configurations as executed,
with machine-specific absolute paths replaced by `${PM_*}` variables.
`configs/baselines/` holds the rule constants, the fitted GARCH parameters and
the classifier metadata. `configs/evaluation/paper_base_config.json` is the
resolved environment configuration, dumped from the code.
